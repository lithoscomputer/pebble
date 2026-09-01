//! Fetching a page, and turning it into something a model can read.

use std::error::Error as StdError;
use std::sync::Arc;

use lithos_llm::Client;
use lithos_llm::types::{Request, ToolDefinition};
use serde_json::Value;

use crate::char_boundary::floor_char_boundary;
use crate::environment::ExecRequest;
use crate::tool::{RegisteredTool, ToolError, required_str};
use crate::types::{ToolErrorKind, ToolSource};

mod markdown;

use self::markdown::html_to_markdown;

/// How much of a fetched page reaches the model.
const MAX_WEB_FETCH_BYTES: usize = 100 * 1024;

/// What `web_fetch` calls itself when it asks for a page.
const WEB_FETCH_USER_AGENT: &str = "pebble/0.1";

/// The model that answers a `web_fetch` prompt about a page.
///
/// A session that has one lets `web_fetch` answer a question about what it
/// fetched instead of returning the whole page; a session without one returns
/// the page and says the summary was unavailable. Configure it with
/// [`SessionBuilder::web_fetch_summarizer`](crate::SessionBuilder::web_fetch_summarizer).
///
/// The model is named the way every other selector in pebble is — a catalog
/// id, an alias, or a `provider/model` pair — and is resolved by the client
/// when the call is made, so a page can be summarized by a smaller and cheaper
/// model than the one running the session.
#[derive(Clone)]
pub struct WebFetchSummarizer {
    client: Client,
    model:  String,
}

impl WebFetchSummarizer {
    /// A summarizer that asks `model` through `client`.
    #[must_use]
    pub fn new(client: Client, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
        }
    }

    /// The model selector this summarizer resolves.
    #[must_use]
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Answers `prompt` about the content of `url`.
    async fn summarize(&self, url: &str, content: &str, prompt: &str) -> Result<String, ToolError> {
        let request = Request::builder()
            .model(self.model.clone())
            .user(format!(
                "Content from {url}:\n---\n{content}\n---\n\n{prompt}\n\nRespond concisely based \
                 only on the content above."
            ))
            .build()
            .map_err(|error| self.failed(error))?;

        let response = self
            .client
            .complete(request)
            .await
            .map_err(|error| self.failed(error))?;
        Ok(response.text())
    }

    /// What the model is told when the summarizing call itself failed.
    ///
    /// The failure is named in the message as well as kept as the error's
    /// cause: the model reads only the message, and "summarization failed" on
    /// its own tells it nothing about whether asking again would help.
    fn failed(&self, error: impl StdError + Send + Sync + 'static) -> ToolError {
        ToolError::with_source(
            ToolErrorKind::Execution,
            format!(
                "web_fetch summarization (model={}) failed: {error}",
                self.model
            ),
            error,
        )
    }
}

/// Fetches a URL through the environment and returns it as Markdown.
///
/// The fetch runs as a `curl` command inside the session's environment rather
/// than from pebble's own process, so a page is retrieved from wherever the
/// session's work happens — inside the container, on the remote workspace —
/// and under whatever network policy that place has.
#[must_use]
pub fn make_web_fetch_tool() -> RegisteredTool {
    RegisteredTool {
        definition: ToolDefinition::function(
            "web_fetch",
            "Fetch content from a URL that starts with http:// or https://. Pass a prompt to \
             extract specific information or summarize the page; omit prompt to return the page \
             content.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch (must be http:// or https://)"},
                    "prompt": {"type": "string", "description": "A question or instruction about the page content. When provided, returns a concise answer instead of the full page."},
                    "timeout_ms": {"type": "integer", "description": "Timeout in milliseconds (default 30000, max 60000)"}
                },
                "required": ["url"]
            }),
        ),
        executor:   Arc::new(|args, ctx| {
            Box::pin(async move {
                let url = required_str(&args, "url")?;
                let prompt = args.get("prompt").and_then(Value::as_str);
                let timeout_ms = args
                    .get("timeout_ms")
                    .and_then(Value::as_u64)
                    .unwrap_or(30_000)
                    .min(60_000);

                if !url.starts_with("http://") && !url.starts_with("https://") {
                    return Err(ToolError::invalid_arguments(
                        "URL must start with http:// or https://",
                    ));
                }

                let timeout_secs = timeout_ms.div_ceil(1000);
                let command = format!(
                    "curl -sL --max-time {timeout_secs} -H 'User-Agent: {WEB_FETCH_USER_AGENT}' {}",
                    single_quoted(url)
                );

                let tool_env = ctx.resolve_tool_env().await?;
                let outcome = ctx
                    .env
                    .exec(ExecRequest {
                        timeout_ms: Some(timeout_ms),
                        env_vars: tool_env.as_ref(),
                        cancel_token: Some(ctx.cancel.clone()),
                        ..ExecRequest::new(&command)
                    })
                    .await?;

                let result = outcome.result;
                if !result.is_success() {
                    return Err(ToolError::execution(format!(
                        "curl failed (exit code {}): {}",
                        result.display_exit_code(),
                        result.stderr.trim()
                    )));
                }

                let content = bounded(html_to_markdown(&result.stdout));

                match (prompt, ctx.web_fetch_summarizer.as_ref()) {
                    (Some(prompt), Some(summarizer)) => {
                        summarizer.summarize(url, &content, prompt).await
                    }
                    (Some(_), None) => Ok(format!(
                        "[Note: prompt summarization unavailable, returning full \
                         content]\n\n{content}"
                    )),
                    (None, _) => Ok(content),
                }
            })
        }),
        source:     ToolSource::Native,
    }
}

/// Cuts a fetched page to what a model is given, saying so when it had to.
fn bounded(mut content: String) -> String {
    if content.len() > MAX_WEB_FETCH_BYTES {
        content.truncate(floor_char_boundary(&content, MAX_WEB_FETCH_BYTES));
        content.push_str("\n\n[Output truncated at 100KB]");
    }
    content
}

/// One shell word holding `value` literally.
///
/// Everything is quoted rather than only what looks dangerous, because the
/// value is a URL the model chose and a quoting rule with exceptions is a
/// quoting rule with a way through it. A single quote inside the value ends
/// the quoted run, escapes itself, and opens a new one, which is the only
/// escape single quotes admit.
fn single_quoted(value: &str) -> String {
    format!("'{}'", value.replace('\'', r"'\''"))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use lithos_llm::adapter::ProviderAdapter;
    use lithos_llm::catalog::Catalog;
    use lithos_llm::types::ErrorKind as LlmErrorKind;
    use serde_json::json;

    use super::*;
    use crate::environment::ExecResult;
    use crate::test_support::{
        MockEnvironment, ScriptedCompletion, ScriptedFailure, ScriptedProvider, client_from,
        message_text, text_response,
    };
    use crate::tool::{StaticEnvProvider, ToolContext};
    use crate::tools::testing::{context, context_for};
    use crate::types::{CommandTermination, ToolErrorKind};

    fn fetched(body: &str) -> MockEnvironment {
        MockEnvironment {
            exec_result: ExecResult {
                stdout:      body.to_owned(),
                stderr:      String::new(),
                exit_code:   Some(0),
                termination: CommandTermination::Exited,
                duration_ms: 100,
            },
            ..MockEnvironment::default()
        }
    }

    /// A context whose summarizer answers one call with `text`, and the
    /// provider that answered it.
    fn summarizing(text: &str) -> (ToolContext, Arc<ScriptedProvider>) {
        let (client, provider) = client_from(
            ScriptedProvider::new(Vec::new())
                .completing(vec![ScriptedCompletion::response(text_response(text))]),
        );
        (
            context(fetched("<html><body><p>Page content</p></body></html>"))
                .with_web_fetch_summarizer(Arc::new(WebFetchSummarizer::new(client, "test/model"))),
            provider,
        )
    }

    #[tokio::test]
    async fn the_curl_command_carries_the_timeout_the_agent_and_the_url() {
        let tool = make_web_fetch_tool();
        let environment = Arc::new(fetched("<html><body><h1>hello</h1></body></html>"));

        let output = (tool.executor)(
            json!({"url": "https://example.com"}),
            context_for(Arc::clone(&environment)),
        )
        .await
        .expect("the page is fetched");

        assert!(
            output.contains("# hello"),
            "HTML should become Markdown, got: {output}"
        );
        assert!(!output.contains("<html>"), "got: {output}");
        let command = environment
            .captured_command
            .lock()
            .expect("captured_command lock is not poisoned")
            .clone()
            .expect("a command ran");
        assert!(
            command.starts_with("curl -sL --max-time 30 "),
            "got: {command}"
        );
        assert!(command.contains("https://example.com"), "got: {command}");
        assert!(command.contains("User-Agent: pebble/0.1"), "got: {command}");
    }

    #[tokio::test]
    async fn a_url_with_another_scheme_is_refused_before_anything_runs() {
        let tool = make_web_fetch_tool();
        let environment = Arc::new(MockEnvironment::default());

        let error = (tool.executor)(
            json!({"url": "ftp://example.com/file"}),
            context_for(Arc::clone(&environment)),
        )
        .await
        .expect_err("only http and https are fetched");

        assert_eq!(error.kind(), ToolErrorKind::InvalidArguments);
        assert_eq!(error.message(), "URL must start with http:// or https://");
        assert!(
            environment
                .captured_command
                .lock()
                .expect("captured_command lock is not poisoned")
                .is_none()
        );
    }

    #[tokio::test]
    async fn the_models_timeout_reaches_both_curl_and_the_environment() {
        let tool = make_web_fetch_tool();
        let environment = Arc::new(MockEnvironment::default());

        let _ = (tool.executor)(
            json!({"url": "https://example.com", "timeout_ms": 15_000}),
            context_for(Arc::clone(&environment)),
        )
        .await;

        assert_eq!(
            *environment
                .captured_timeout
                .lock()
                .expect("captured_timeout lock is not poisoned"),
            Some(15_000)
        );
        assert!(
            environment
                .captured_command
                .lock()
                .expect("captured_command lock is not poisoned")
                .as_ref()
                .expect("a command ran")
                .contains("--max-time 15")
        );
    }

    #[tokio::test]
    async fn a_timeout_beyond_a_minute_is_capped() {
        let tool = make_web_fetch_tool();
        let environment = Arc::new(MockEnvironment::default());

        let _ = (tool.executor)(
            json!({"url": "https://example.com", "timeout_ms": 120_000}),
            context_for(Arc::clone(&environment)),
        )
        .await;

        assert_eq!(
            *environment
                .captured_timeout
                .lock()
                .expect("captured_timeout lock is not poisoned"),
            Some(60_000)
        );
        assert!(
            environment
                .captured_command
                .lock()
                .expect("captured_command lock is not poisoned")
                .as_ref()
                .expect("a command ran")
                .contains("--max-time 60")
        );
    }

    #[tokio::test]
    async fn a_page_beyond_the_budget_is_cut_and_says_so() {
        let tool = make_web_fetch_tool();

        let output = (tool.executor)(
            json!({"url": "https://example.com"}),
            context(fetched(&"x".repeat(150 * 1024))),
        )
        .await
        .expect("the page is fetched");

        assert!(output.len() < 110 * 1024, "{}", output.len());
        assert!(output.ends_with("[Output truncated at 100KB]"));
    }

    /// The cut lands wherever the budget falls, which on a page of text is
    /// usually inside a character.
    #[tokio::test]
    async fn a_page_is_cut_at_a_character_boundary() {
        let tool = make_web_fetch_tool();
        // Four-byte characters, so the 100 KiB mark is inside one of them.
        let page = "😀".repeat(40 * 1024);

        let output = (tool.executor)(
            json!({"url": "https://example.com"}),
            context(fetched(&page)),
        )
        .await
        .expect("the page is fetched");

        assert!(output.ends_with("[Output truncated at 100KB]"));
        assert!(
            output.starts_with('😀'),
            "the kept text is still valid characters"
        );
    }

    #[tokio::test]
    async fn a_curl_that_failed_reports_its_exit_code_and_message() {
        let tool = make_web_fetch_tool();
        let environment = MockEnvironment {
            exec_result: ExecResult {
                stdout:      String::new(),
                stderr:      "curl: (6) Could not resolve host".to_owned(),
                exit_code:   Some(6),
                termination: CommandTermination::Exited,
                duration_ms: 100,
            },
            ..MockEnvironment::default()
        };

        let error = (tool.executor)(
            json!({"url": "https://nonexistent.example.com"}),
            context(environment),
        )
        .await
        .expect_err("curl failed");

        assert_eq!(
            error.message(),
            "curl failed (exit code 6): curl: (6) Could not resolve host"
        );
    }

    #[tokio::test]
    async fn the_calls_environment_variables_reach_curl() {
        let tool = make_web_fetch_tool();
        let environment = Arc::new(fetched("fetched content"));
        let tool_env = HashMap::from([("API_KEY".to_owned(), "secret".to_owned())]);

        let _ = (tool.executor)(
            json!({"url": "https://example.com"}),
            context_for(Arc::clone(&environment))
                .with_tool_env_provider(Arc::new(StaticEnvProvider(tool_env.clone()))),
        )
        .await;

        assert_eq!(
            *environment
                .captured_env_vars
                .lock()
                .expect("captured_env_vars lock is not poisoned"),
            Some(tool_env)
        );
    }

    #[tokio::test]
    async fn a_prompt_is_answered_by_the_summarizer() {
        let tool = make_web_fetch_tool();
        let (context, provider) = summarizing("Rust is a systems programming language.");

        let output = (tool.executor)(
            json!({"url": "https://example.com", "prompt": "What is Rust?"}),
            context,
        )
        .await
        .expect("the summarizer answers");

        assert_eq!(output, "Rust is a systems programming language.");
        let requests = provider.completion_requests();
        assert_eq!(requests.len(), 1);
        let asked = message_text(&requests[0].messages()[0]);
        assert!(
            asked.starts_with("Content from https://example.com:"),
            "{asked}"
        );
        assert!(asked.contains("Page content"), "{asked}");
        assert!(asked.contains("What is Rust?"), "{asked}");
        assert!(
            asked.ends_with("Respond concisely based only on the content above."),
            "{asked}"
        );
    }

    #[tokio::test]
    async fn a_prompt_without_a_summarizer_returns_the_page_and_says_why() {
        let tool = make_web_fetch_tool();

        let output = (tool.executor)(
            json!({"url": "https://example.com", "prompt": "What is Rust?"}),
            context(fetched(
                "<html><body><p>Rust is a systems programming language.</p></body></html>",
            )),
        )
        .await
        .expect("the page is fetched");

        assert_eq!(
            output,
            "[Note: prompt summarization unavailable, returning full content]\n\nRust is a \
             systems programming language."
        );
    }

    #[tokio::test]
    async fn a_summarizer_that_fails_says_which_model_it_asked() {
        let tool = make_web_fetch_tool();
        let (client, _provider) = client_from(ScriptedProvider::new(Vec::new()).completing(vec![
            ScriptedCompletion::Failure(ScriptedFailure::terminal(
                LlmErrorKind::Provider,
                "the model is overloaded",
            )),
        ]));

        let error = (tool.executor)(
            json!({"url": "https://example.com", "prompt": "What is Rust?"}),
            context(fetched("<html><body><p>content</p></body></html>"))
                .with_web_fetch_summarizer(Arc::new(WebFetchSummarizer::new(client, "test/model"))),
        )
        .await
        .expect_err("the summarizing call failed");

        assert_eq!(error.kind(), ToolErrorKind::Execution);
        assert!(
            error
                .message()
                .starts_with("web_fetch summarization (model=test/model) failed: "),
            "{}",
            error.message()
        );
        assert!(
            error.message().contains("the model is overloaded"),
            "{}",
            error.message()
        );
    }

    /// The summarizer's own selector decides which provider answers, so a
    /// session can summarize with a model it is not itself running.
    #[tokio::test]
    async fn the_summarizer_asks_the_model_its_selector_names() {
        /// Two providers, so the selector has something to choose between.
        const CATALOG: &str = r#"
schema_version = 1

[providers.running]
display_name = "Running"
adapter = "test-adapter"
codec = "test-codec"
base_url = "http://127.0.0.1"
default_model = "big"

[providers.running.auth]
type = "none"

[providers.running.models.big]
display_name = "Big"
api_model = "big"
capabilities = { text = true }

[providers.summarizing]
display_name = "Summarizing"
adapter = "test-adapter"
codec = "test-codec"
base_url = "http://127.0.0.1"
default_model = "small"

[providers.summarizing.auth]
type = "none"

[providers.summarizing.models.small]
display_name = "Small"
api_model = "small"
capabilities = { text = true }
"#;

        let refusing = Arc::new(ScriptedProvider::new(Vec::new()));
        let answering = Arc::new(ScriptedProvider::new(Vec::new()).completing(vec![
            ScriptedCompletion::response(text_response("summarized content")),
        ]));
        let catalog = Catalog::builder()
            .overlay_toml(CATALOG)
            .expect("the catalog layer parses")
            .build()
            .expect("the catalog validates");
        let client = Client::builder()
            .catalog(catalog)
            .adapter_arc("running", Arc::clone(&refusing) as Arc<dyn ProviderAdapter>)
            .adapter_arc(
                "summarizing",
                Arc::clone(&answering) as Arc<dyn ProviderAdapter>,
            )
            .build()
            .expect("the client builds")
            .client;

        let tool = make_web_fetch_tool();
        let output = (tool.executor)(
            json!({"url": "https://example.com", "prompt": "Summarize this"}),
            context(fetched("<html><body><p>Page content</p></body></html>"))
                .with_web_fetch_summarizer(Arc::new(WebFetchSummarizer::new(
                    client,
                    "summarizing/small",
                ))),
        )
        .await
        .expect("the named provider answers");

        assert_eq!(output, "summarized content");
        assert_eq!(answering.completion_count(), 1);
        assert_eq!(refusing.completion_count(), 0);
    }

    #[test]
    fn a_url_is_quoted_whole() {
        assert_eq!(
            single_quoted("https://example.com"),
            "'https://example.com'"
        );
        assert_eq!(
            single_quoted("https://example.com/?q=a b&x=1"),
            "'https://example.com/?q=a b&x=1'"
        );
        assert_eq!(
            single_quoted("https://example.com/'; rm -rf /"),
            r"'https://example.com/'\''; rm -rf /'"
        );
    }
}
