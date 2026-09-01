//! The template engine the built-in system prompts are written against.
//!
//! Pebble's prompts are checked-in `.md.j2` files, and each one is a Jinja
//! template in the narrow sense fabro's own prompts used: a value substituted
//! into a sentence, a section that appears only when its tool is registered,
//! and a comment carrying an attribution. Nothing else — no loops, no filters,
//! no includes, no macros, no arithmetic.
//!
//! Fabro rendered them with minijinja through `fabro-template`. Pebble cannot
//! take that dependency: no version of minijinja appears in the lockfile pebble
//! is aligned to, and adding one would put packages there that nothing else
//! resolves (the same rule that kept `serde_with`, `jsonschema` and `htmd`
//! out). So this module implements exactly the subset the checked-in prompts
//! use, matching minijinja where it is observable — which the prompt snapshots
//! measure rather than assume, being byte-for-byte the ones fabro rendered:
//!
//! - **Strict.** Naming an input the caller did not supply is an error, not an
//!   empty string. A prompt that silently loses a sentence is worse than one
//!   that fails a test.
//! - **No escaping.** Rendered text is prose for a model, never HTML, so `<`
//!   and `&` pass through unchanged (minijinja's `AutoEscape::None`).
//! - **The template's last newline is dropped**, which is minijinja's default
//!   (`keep_trailing_newline = false`) and what the prompt snapshots were taken
//!   under. Unlike fabro, a template with no tags at all is treated the same
//!   way rather than returned verbatim: there is one rule here, not two.
//!
//! # Syntax
//!
//! ```text
//! {{ inputs.name }}                          an input's value
//! {% if inputs.flag %}…{% else %}…{% endif %} a section, optionally negated
//! {% if inputs.name == "value" %}…{% endif %} compared with a string literal
//! {# a comment, which renders as nothing #}
//! ```
//!
//! Every tag may carry `-` next to either delimiter (`{%-`, `-%}`, `-#}`),
//! which strips the whitespace on that side, as Jinja spells it. Blocks do not
//! trim anything on their own: the newline after `{% endif %}` is part of the
//! output, which is what the prompts are written around.

use std::collections::BTreeMap;

use thiserror::Error;

/// A value a template may name.
///
/// Fabro passed `toml::Value` and used only strings and booleans of it. Pebble
/// takes those two, so a prompt cannot name a number or a table that no
/// template reads and no test covers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum TemplateValue {
    /// Text, substituted as it stands.
    Text(String),
    /// A condition, true or false.
    Flag(bool),
}

impl TemplateValue {
    /// How this value reads inside `{{ }}`.
    fn render(&self) -> &str {
        match self {
            Self::Text(text) => text,
            Self::Flag(true) => "true",
            Self::Flag(false) => "false",
        }
    }

    /// Whether `{% if %}` takes this value's branch.
    ///
    /// A flag answers itself; text answers whether it says anything, which is
    /// Jinja's own rule for a string.
    const fn is_true(&self) -> bool {
        match self {
            Self::Text(text) => !text.is_empty(),
            Self::Flag(flag) => *flag,
        }
    }
}

impl From<bool> for TemplateValue {
    fn from(flag: bool) -> Self {
        Self::Flag(flag)
    }
}

impl From<String> for TemplateValue {
    fn from(text: String) -> Self {
        Self::Text(text)
    }
}

impl From<&str> for TemplateValue {
    fn from(text: &str) -> Self {
        Self::Text(text.to_owned())
    }
}

/// What a template is rendered against.
///
/// Fabro's context exposed `goal`, `inputs` and `vars`; the prompts only ever
/// read `inputs`, so that is all this carries. A template naming anything else
/// is an error rather than an empty string.
#[derive(Debug, Clone, Default)]
pub(crate) struct TemplateContext {
    inputs: BTreeMap<String, TemplateValue>,
}

impl TemplateContext {
    /// An empty context.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// The same context, reading `inputs`.
    pub(crate) fn with_inputs(mut self, inputs: BTreeMap<String, TemplateValue>) -> Self {
        self.inputs = inputs;
        self
    }

    /// What `name` resolves to, if anything.
    fn get(&self, name: &str) -> Option<&TemplateValue> {
        self.inputs.get(name)
    }
}

/// Why a template did not render.
///
/// Every case is a mistake in a checked-in template or in the code that fills
/// it, which is why the callers of [`render_named`] treat one as a bug rather
/// than a condition to recover from.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("template `{template}` line {line}: {message}")]
pub(crate) struct TemplateError {
    /// The template's name, as its caller spells it.
    pub(crate) template: String,
    /// Which line of the template the trouble is on, counting from one.
    pub(crate) line:     usize,
    /// What is wrong with it.
    pub(crate) message:  String,
}

/// A failure before it knows which template it happened in.
#[derive(Debug)]
struct Failure {
    /// Where in the source it happened.
    at:      usize,
    /// What is wrong.
    message: String,
}

impl Failure {
    fn new(at: usize, message: impl Into<String>) -> Self {
        Self {
            at,
            message: message.into(),
        }
    }

    /// The same failure, named and placed in `source`.
    fn locate(self, template: &str, source: &str) -> TemplateError {
        let before = &source[..self.at.min(source.len())];
        TemplateError {
            template: template.to_owned(),
            line:     before.matches('\n').count() + 1,
            message:  self.message,
        }
    }
}

/// Renders `source` against `context`, calling it `name` in any failure.
///
/// The template's own trailing newline is dropped, which is what minijinja does
/// by default and what the prompts are written for: a prompt is a paragraph the
/// caller goes on appending sections to.
pub(crate) fn render_named(
    name: &str,
    source: &str,
    context: &TemplateContext,
) -> Result<String, TemplateError> {
    let source = strip_trailing_newline(source);
    let nodes = parse(source).map_err(|failure| failure.locate(name, source))?;
    let mut rendered = String::with_capacity(source.len());
    render_nodes(&nodes, context, &mut rendered).map_err(|failure| failure.locate(name, source))?;
    Ok(rendered)
}

/// `source` without the one line ending it ends with, if it ends with one.
fn strip_trailing_newline(source: &str) -> &str {
    source.strip_suffix('\n').map_or(source, |trimmed| {
        trimmed.strip_suffix('\r').unwrap_or(trimmed)
    })
}

/// One piece of a parsed template.
#[derive(Debug)]
enum Node {
    /// Text, copied out as it stands.
    Text(String),
    /// An input, substituted.
    Input { name: String, at: usize },
    /// A section, taken or skipped.
    Branch {
        condition: Condition,
        taken:     Vec<Self>,
        skipped:   Vec<Self>,
    },
}

/// What an `{% if %}` asks about one input.
#[derive(Debug, Clone)]
struct Condition {
    /// The input the question is about.
    name:   String,
    /// The text it must equal, where the question is an equality.
    equals: Option<String>,
    /// Where the tag is, for a failure to point at.
    at:     usize,
}

impl Condition {
    /// Whether `context` takes this branch.
    fn holds(&self, context: &TemplateContext) -> Result<bool, Failure> {
        let value = context.get(&self.name).ok_or_else(|| {
            Failure::new(
                self.at,
                format!(
                    "`inputs.{}` is not one of this template's inputs",
                    self.name
                ),
            )
        })?;
        Ok(match &self.equals {
            // A flag never equals a string, which is what Jinja answers when
            // the two types meet.
            Some(literal) => matches!(value, TemplateValue::Text(text) if text == literal),
            None => value.is_true(),
        })
    }
}

/// One tag or run of text, before the branches are nested.
#[derive(Debug)]
enum Token {
    Text(String),
    Input { name: String, at: usize },
    If(Condition),
    Else { at: usize },
    EndIf { at: usize },
}

/// Which tag closed a run of nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Delimiter {
    Else,
    EndIf,
}

/// The nodes `source` describes.
fn parse(source: &str) -> Result<Vec<Node>, Failure> {
    let tokens = tokenize(source)?;
    let mut cursor = 0;
    let (nodes, delimiter) = parse_sequence(&tokens, &mut cursor)?;
    match delimiter {
        None => Ok(nodes),
        Some(Delimiter::Else) => Err(Failure::new(
            delimiter_position(&tokens, cursor),
            "`{% else %}` without an `{% if %}`",
        )),
        Some(Delimiter::EndIf) => Err(Failure::new(
            delimiter_position(&tokens, cursor),
            "`{% endif %}` without an `{% if %}`",
        )),
    }
}

/// Where the delimiter the cursor just passed sits in the source.
fn delimiter_position(tokens: &[Token], cursor: usize) -> usize {
    match tokens.get(cursor.saturating_sub(1)) {
        Some(Token::Else { at } | Token::EndIf { at }) => *at,
        _ => 0,
    }
}

/// Reads nodes until this level's `{% else %}` or `{% endif %}`, or the end.
fn parse_sequence(
    tokens: &[Token],
    cursor: &mut usize,
) -> Result<(Vec<Node>, Option<Delimiter>), Failure> {
    let mut nodes = Vec::new();
    while let Some(token) = tokens.get(*cursor) {
        *cursor += 1;
        match token {
            Token::Text(text) => nodes.push(Node::Text(text.clone())),
            Token::Input { name, at } => nodes.push(Node::Input {
                name: name.clone(),
                at:   *at,
            }),
            Token::Else { .. } => return Ok((nodes, Some(Delimiter::Else))),
            Token::EndIf { .. } => return Ok((nodes, Some(Delimiter::EndIf))),
            Token::If(condition) => {
                const UNCLOSED: &str = "`{% if %}` was never closed";
                let opened_at = condition.at;
                let (taken, delimiter) = parse_sequence(tokens, cursor)?;
                let skipped = match delimiter {
                    Some(Delimiter::EndIf) => Vec::new(),
                    Some(Delimiter::Else) => {
                        let (skipped, delimiter) = parse_sequence(tokens, cursor)?;
                        match delimiter {
                            Some(Delimiter::EndIf) => skipped,
                            Some(Delimiter::Else) => {
                                return Err(Failure::new(
                                    opened_at,
                                    "`{% if %}` has more than one `{% else %}`",
                                ));
                            }
                            None => return Err(Failure::new(opened_at, UNCLOSED)),
                        }
                    }
                    None => return Err(Failure::new(opened_at, UNCLOSED)),
                };
                nodes.push(Node::Branch {
                    condition: condition.clone(),
                    taken,
                    skipped,
                });
            }
        }
    }
    Ok((nodes, None))
}

/// Writes what `nodes` render to into `rendered`.
fn render_nodes(
    nodes: &[Node],
    context: &TemplateContext,
    rendered: &mut String,
) -> Result<(), Failure> {
    for node in nodes {
        match node {
            Node::Text(text) => rendered.push_str(text),
            Node::Input { name, at } => {
                let value = context.get(name).ok_or_else(|| {
                    Failure::new(
                        *at,
                        format!("`inputs.{name}` is not one of this template's inputs"),
                    )
                })?;
                rendered.push_str(value.render());
            }
            Node::Branch {
                condition,
                taken,
                skipped,
            } => {
                let branch = if condition.holds(context)? {
                    taken
                } else {
                    skipped
                };
                render_nodes(branch, context, rendered)?;
            }
        }
    }
    Ok(())
}

/// The tokens `source` is made of, with whitespace control already applied.
fn tokenize(source: &str) -> Result<Vec<Token>, Failure> {
    let mut tokens = Vec::new();
    let mut cursor = 0;
    // Set by a tag that ended with `-`, and spent on the text after it.
    let mut trim_leading = false;

    while cursor < source.len() {
        let Some(start) = find_tag(&source[cursor..]).map(|offset| cursor + offset) else {
            push_text(&mut tokens, &source[cursor..], trim_leading, false);
            return Ok(tokens);
        };

        let opener = &source[start..start + 2];
        let closer = match opener {
            "{{" => "}}",
            "{%" => "%}",
            _ => "#}",
        };
        let quoted = opener != "{#";
        let mut body_start = start + 2;
        let trim_trailing = source[body_start..].starts_with('-');
        if trim_trailing {
            body_start += 1;
        }
        push_text(
            &mut tokens,
            &source[cursor..start],
            trim_leading,
            trim_trailing,
        );

        let body_end = find_closer(&source[body_start..], closer, quoted)
            .map(|offset| body_start + offset)
            .ok_or_else(|| {
                Failure::new(start, format!("`{opener}` is never closed by `{closer}`"))
            })?;
        let mut body = &source[body_start..body_end];
        trim_leading = body.ends_with('-');
        if trim_leading {
            body = &body[..body.len() - 1];
        }

        match opener {
            "{{" => tokens.push(Token::Input {
                name: parse_input_name(body, start)?,
                at:   start,
            }),
            "{%" => tokens.push(parse_block(body, start)?),
            // A comment renders as nothing at all.
            _ => {}
        }
        cursor = body_end + closer.len();
    }

    Ok(tokens)
}

/// Adds `text` as a token, minus whatever the tags around it trim.
fn push_text(tokens: &mut Vec<Token>, text: &str, trim_start: bool, trim_end: bool) {
    let mut text = text;
    if trim_start {
        text = text.trim_start();
    }
    if trim_end {
        text = text.trim_end();
    }
    if !text.is_empty() {
        tokens.push(Token::Text(text.to_owned()));
    }
}

/// Where the next `{{`, `{%` or `{#` starts.
fn find_tag(source: &str) -> Option<usize> {
    let bytes = source.as_bytes();
    bytes.iter().enumerate().find_map(|(index, byte)| {
        (*byte == b'{' && matches!(bytes.get(index + 1), Some(b'{' | b'%' | b'#'))).then_some(index)
    })
}

/// Where `closer` next appears, skipping anything inside a quoted string.
fn find_closer(source: &str, closer: &str, quoted: bool) -> Option<usize> {
    let bytes = source.as_bytes();
    let closer = closer.as_bytes();
    let mut index = 0;
    let mut quote: Option<u8> = None;
    while index < bytes.len() {
        let byte = bytes[index];
        match quote {
            // Inside a literal, only its own closing quote means anything.
            Some(open) if byte == open => quote = None,
            None if quoted && matches!(byte, b'"' | b'\'') => quote = Some(byte),
            None if bytes[index..].starts_with(closer) => return Some(index),
            // Ordinary text, inside a literal or out.
            _ => {}
        }
        index += 1;
    }
    None
}

/// The input `{{ … }}` names.
fn parse_input_name(body: &str, at: usize) -> Result<String, Failure> {
    let expression = body.trim();
    let name = expression.strip_prefix("inputs.").filter(|name| {
        !name.is_empty()
            && name
                .chars()
                .all(|character| character.is_ascii_alphanumeric() || character == '_')
    });
    name.map(str::to_owned).ok_or_else(|| {
        Failure::new(
            at,
            format!("`{expression}` is not an input; write `inputs.<name>`"),
        )
    })
}

/// The tag `{% … %}` is.
fn parse_block(body: &str, at: usize) -> Result<Token, Failure> {
    let block = body.trim();
    if block == "else" {
        return Ok(Token::Else { at });
    }
    if block == "endif" {
        return Ok(Token::EndIf { at });
    }
    let Some(test) = block.strip_prefix("if ") else {
        return Err(Failure::new(
            at,
            format!("`{block}` is not a block this engine knows; write `if`, `else` or `endif`"),
        ));
    };
    Ok(Token::If(parse_condition(test, at)?))
}

/// The question an `{% if %}` asks.
fn parse_condition(test: &str, at: usize) -> Result<Condition, Failure> {
    let (name, equals) = match test.split_once("==") {
        Some((left, right)) => (left, Some(parse_literal(right.trim(), at)?)),
        None => (test, None),
    };
    Ok(Condition {
        name: parse_input_name(name, at)?,
        equals,
        at,
    })
}

/// The text a quoted literal stands for.
fn parse_literal(literal: &str, at: usize) -> Result<String, Failure> {
    let unquoted = |quote: char| {
        literal
            .strip_prefix(quote)
            .and_then(|rest| rest.strip_suffix(quote))
            .filter(|text| !text.contains(quote))
    };
    unquoted('"')
        .or_else(|| unquoted('\''))
        .map(str::to_owned)
        .ok_or_else(|| {
            Failure::new(
                at,
                format!("`{literal}` is not a quoted string, which is all `==` compares against"),
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A context naming `inputs`.
    fn context(inputs: &[(&str, TemplateValue)]) -> TemplateContext {
        TemplateContext::new().with_inputs(
            inputs
                .iter()
                .map(|(name, value)| ((*name).to_owned(), value.clone()))
                .collect(),
        )
    }

    /// What `source` renders to, which every test here expects to succeed.
    fn render(source: &str, inputs: &[(&str, TemplateValue)]) -> String {
        render_named("test.md.j2", source, &context(inputs)).expect("the template renders")
    }

    /// Why `source` did not render.
    fn failure(source: &str, inputs: &[(&str, TemplateValue)]) -> TemplateError {
        render_named("test.md.j2", source, &context(inputs)).expect_err("the template fails")
    }

    #[test]
    fn text_with_no_tags_is_copied_out() {
        assert_eq!(render("just text", &[]), "just text");
    }

    #[test]
    fn a_dollar_sign_and_a_brace_are_not_a_tag() {
        assert_eq!(
            render("price is $5, home is ${HOME}", &[]),
            "price is $5, home is ${HOME}"
        );
    }

    #[test]
    fn an_input_is_substituted() {
        assert_eq!(
            render("Powered by {{ inputs.provider }}.", &[(
                "provider",
                "Moonshot AI".into()
            )]),
            "Powered by Moonshot AI."
        );
    }

    #[test]
    fn an_input_is_substituted_once_and_not_rendered_again() {
        // A value that looks like a tag is text, not more template.
        assert_eq!(
            render("{{ inputs.echo }}", &[("echo", "{{ inputs.echo }}".into())]),
            "{{ inputs.echo }}"
        );
    }

    #[test]
    fn nothing_is_escaped_on_the_way_out() {
        // Fabro set `AutoEscape::None`, because a prompt is prose for a model
        // rather than a web page.
        assert_eq!(
            render("<{{ inputs.raw }}>", &[("raw", "a & b < c".into())]),
            "<a & b < c>"
        );
    }

    #[test]
    fn a_flag_renders_as_a_word() {
        assert_eq!(
            render("{{ inputs.on }}/{{ inputs.off }}", &[
                ("on", true.into()),
                ("off", false.into())
            ]),
            "true/false"
        );
    }

    #[test]
    fn a_flag_chooses_a_branch() {
        let source = "{% if inputs.flag %}yes{% else %}no{% endif %}";

        assert_eq!(render(source, &[("flag", true.into())]), "yes");
        assert_eq!(render(source, &[("flag", false.into())]), "no");
    }

    #[test]
    fn a_branch_with_no_else_renders_nothing_when_it_is_not_taken() {
        let source = "a{% if inputs.flag %}b{% endif %}c";

        assert_eq!(render(source, &[("flag", true.into())]), "abc");
        assert_eq!(render(source, &[("flag", false.into())]), "ac");
    }

    #[test]
    fn text_chooses_a_branch_by_whether_it_says_anything() {
        let source = "{% if inputs.text %}yes{% else %}no{% endif %}";

        assert_eq!(render(source, &[("text", "something".into())]), "yes");
        assert_eq!(render(source, &[("text", String::new().into())]), "no");
    }

    #[test]
    fn a_string_literal_is_compared_for_equality() {
        let source = r#"{% if inputs.editor == "apply_patch" %}patch{% else %}edit{% endif %}"#;

        assert_eq!(render(source, &[("editor", "apply_patch".into())]), "patch");
        assert_eq!(render(source, &[("editor", "edit_file".into())]), "edit");
    }

    #[test]
    fn single_quotes_name_a_literal_too() {
        let source = "{% if inputs.editor == 'edit_file' %}edit{% else %}patch{% endif %}";

        assert_eq!(render(source, &[("editor", "edit_file".into())]), "edit");
    }

    #[test]
    fn a_flag_never_equals_a_string() {
        let source = r#"{% if inputs.flag == "true" %}yes{% else %}no{% endif %}"#;

        assert_eq!(render(source, &[("flag", true.into())]), "no");
    }

    #[test]
    fn branches_nest() {
        let source = "{% if inputs.outer %}[{% if inputs.inner %}in{% else %}out{% endif %}]\
                      {% else %}none{% endif %}";

        assert_eq!(
            render(source, &[("outer", true.into()), ("inner", true.into())]),
            "[in]"
        );
        assert_eq!(
            render(source, &[("outer", true.into()), ("inner", false.into())]),
            "[out]"
        );
        assert_eq!(
            render(source, &[("outer", false.into()), ("inner", true.into())]),
            "none"
        );
    }

    #[test]
    fn a_comment_renders_as_nothing() {
        assert_eq!(render("a{# not shown #}b", &[]), "ab");
    }

    #[test]
    fn a_comment_may_span_lines_and_hold_punctuation() {
        // The GPT-5.6 prompt opens with a multi-line attribution comment.
        assert_eq!(
            render(
                "{#\n  Adapted from openai/codex (Apache-2.0).\n-#}\nYou are an agent.",
                &[]
            ),
            "You are an agent."
        );
    }

    #[test]
    fn a_block_trims_nothing_of_its_own() {
        // Jinja's `trim_blocks` and `lstrip_blocks` are both off, which is what
        // the checked-in prompts are written around: the newline after
        // `{% endif %}` is the blank line before the next heading.
        assert_eq!(
            render("one\n{% if inputs.flag %}two\n{% endif %}three", &[(
                "flag",
                false.into()
            )]),
            "one\nthree"
        );
        assert_eq!(
            render("one\n{% if inputs.flag %}two\n{% endif %}three", &[(
                "flag",
                true.into()
            )]),
            "one\ntwo\nthree"
        );
    }

    #[test]
    fn a_marker_on_the_right_strips_the_whitespace_after_a_tag() {
        assert_eq!(
            render("a{% if inputs.flag -%}\n   b{% endif %}", &[(
                "flag",
                true.into()
            )]),
            "ab"
        );
    }

    #[test]
    fn a_marker_on_the_left_strips_the_whitespace_before_a_tag() {
        assert_eq!(
            render("a   \n{%- if inputs.flag %}b{% endif %}", &[(
                "flag",
                true.into()
            )]),
            "ab"
        );
    }

    #[test]
    fn an_output_takes_the_markers_too() {
        assert_eq!(
            render("a  {{- inputs.text -}}  b", &[("text", "X".into())]),
            "aXb"
        );
    }

    #[test]
    fn the_templates_own_last_newline_is_dropped() {
        // minijinja's `keep_trailing_newline` is off by default, and the prompt
        // snapshots were taken that way.
        assert_eq!(render("line\n", &[]), "line");
        assert_eq!(render("line\r\n", &[]), "line");
        assert_eq!(render("line\n\n", &[]), "line\n");
        assert_eq!(render("line", &[]), "line");
    }

    #[test]
    fn an_input_the_caller_did_not_supply_is_an_error() {
        let error = failure("hi\n{{ inputs.missing }}", &[]);

        assert_eq!(error.template, "test.md.j2");
        assert_eq!(error.line, 2);
        assert!(
            error.message.contains("inputs.missing"),
            "{}",
            error.message
        );
    }

    #[test]
    fn a_condition_on_an_input_the_caller_did_not_supply_is_an_error() {
        let error = failure("{% if inputs.missing %}x{% endif %}", &[]);

        assert!(
            error.message.contains("inputs.missing"),
            "{}",
            error.message
        );
    }

    #[test]
    fn a_condition_is_only_asked_when_its_branch_is_reached() {
        // A nested `{% if %}` inside a branch that is skipped is never
        // evaluated, so its input need not be supplied.
        assert_eq!(
            render(
                "{% if inputs.outer %}{% if inputs.inner %}x{% endif %}{% endif %}",
                &[("outer", false.into())]
            ),
            ""
        );
    }

    #[test]
    fn only_inputs_can_be_named() {
        let error = failure("{{ goal }}", &[]);

        assert!(error.message.contains("inputs.<name>"), "{}", error.message);
    }

    #[test]
    fn an_unknown_block_is_an_error() {
        let error = failure("{% for item in inputs.list %}{% endfor %}", &[]);

        assert!(error.message.contains("not a block"), "{}", error.message);
    }

    #[test]
    fn an_unclosed_tag_is_an_error() {
        let error = failure("{{ inputs.text", &[("text", "x".into())]);

        assert!(error.message.contains("never closed"), "{}", error.message);
    }

    #[test]
    fn an_unclosed_branch_is_an_error() {
        let error = failure("{% if inputs.flag %}x", &[("flag", true.into())]);

        assert!(error.message.contains("never closed"), "{}", error.message);
    }

    #[test]
    fn a_stray_endif_is_an_error() {
        let error = failure("x{% endif %}", &[]);

        assert!(error.message.contains("without an"), "{}", error.message);
    }

    #[test]
    fn a_stray_else_is_an_error() {
        let error = failure("x{% else %}y", &[]);

        assert!(error.message.contains("without an"), "{}", error.message);
    }

    #[test]
    fn a_second_else_in_one_branch_is_an_error() {
        let error = failure("{% if inputs.flag %}a{% else %}b{% else %}c{% endif %}", &[
        ]);

        assert!(error.message.contains("more than one"), "{}", error.message);
    }

    #[test]
    fn a_literal_that_is_not_quoted_is_an_error() {
        let error = failure("{% if inputs.editor == apply_patch %}x{% endif %}", &[]);

        assert!(error.message.contains("quoted string"), "{}", error.message);
    }

    #[test]
    fn a_closing_delimiter_inside_a_literal_is_not_the_end_of_the_tag() {
        assert_eq!(
            render(
                r#"{% if inputs.text == "a %} b" %}yes{% else %}no{% endif %}"#,
                &[("text", "a %} b".into())]
            ),
            "yes"
        );
    }

    #[test]
    fn a_failure_says_which_template_and_line_it_is_on() {
        let error = failure("a\nb\n{{ inputs.missing }}", &[]);

        assert_eq!(
            error.to_string(),
            "template `test.md.j2` line 3: `inputs.missing` is not one of this template's inputs"
        );
    }
}
