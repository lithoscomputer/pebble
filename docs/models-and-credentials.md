# Models and credentials

Start `pebble` to choose a default model. If no provider is configured, Pebble
offers API-key setup first. Key input is masked and is separate from prompt
history, session files, and exports.

You can also configure a provider from the shell:

```sh
pebble auth login openai
pebble auth status
pebble --model gpt-5.6
```

From a source checkout, replace `pebble` with
`cargo run --locked -p pebble-cli --`.

In the TUI, use `/login [provider]`, `/logout [provider]`, and `/model`.
`/model all` includes unconfigured models and explains missing credentials,
unavailable provider adapters, and unsupported coding profiles. Model selection
does not call a provider. A configured credential source does not verify account
access or billing status.

## Files

Pebble reads configuration from `PEBBLE_HOME`, or `~/.pebble` when that variable
is unset. The TUI, `exec`, and auth commands use the same files.

| File | Purpose |
| --- | --- |
| `settings.json` | Default model, reasoning level, and TUI preferences. |
| `models.toml` | Optional provider and model overrides. |
| `auth.json` | Saved credentials and explicit credential sources. |
| `auth.lock` | Coordinates credential changes across Pebble processes. |
| `sessions/` | Saved interactive sessions. |

`--sessions-dir` changes only session storage. `--no-session` uses temporary
session storage and still reads the normal application configuration. If you
previously kept `settings.json` beside a custom sessions directory, move it to
`PEBBLE_HOME/settings.json` or `~/.pebble/settings.json`.

Missing optional files use defaults. Invalid files produce an error with the
file path. Restart Pebble after manually changing settings or the catalog.
Credentials are read for each provider request, so another Pebble process can
update or remove them while a session remains open.

## Model selection

For a new TUI session, `--model` overrides the saved default. If neither exists,
Pebble offers configured models and saves the chosen default. On resume, the
session's model takes precedence over the saved default. `--model` can override
the resumed model. Missing credentials open setup for that provider without
changing the session's model.

`pebble exec` uses `--model`, then the saved default, then `claude-sonnet-5`.
It never opens a setup prompt. Missing credentials fail before the agent starts.
Specify `--model` in scripts that need a fixed model.

Use `/settings model` to save the current model as the startup preference.
`/settings reasoning` saves the current reasoning level. Both TUI and `exec`
read that reasoning preference; a resumed session keeps its recorded level.

## Credential sources

Pebble resolves credentials in this order:

1. An explicit environment or header source in `auth.json`.
2. The provider's conventional environment variables.
3. An API key saved by `/login` or `pebble auth login`.

An explicit source with a missing value fails. An authentication failure does
not cause Pebble to try a different account. Environment variables override
saved API keys. `pebble auth status [provider]` shows the active source without
printing its value.

| Provider | Conventional variables |
| --- | --- |
| Anthropic | `ANTHROPIC_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| Gemini | `GEMINI_API_KEY`, then `GOOGLE_API_KEY` |
| Moonshot | `MOONSHOT_API_KEY`, then `KIMI_API_KEY` |
| OpenRouter | `OPENROUTER_API_KEY` |
| Fireworks | `FIREWORKS_API_KEY` |
| Venice | `VENICE_API_KEY` |
| Modal | Both `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` |

Provider support also depends on the adapters enabled in this build. Auth
status reports unavailable adapters. Providers configured with
`auth = { type = "none" }` need no key.

`pebble auth login <provider>` reads a key from a masked terminal prompt. Press
Escape or Ctrl+C to cancel. For a pipe, explicitly select standard input:

```sh
your-secret-command | pebble auth login openai --stdin
```

Keys must be nonempty printable ASCII, at most 8192 bytes. A final newline from
a pipe or paste is removed. Keys are not accepted as command-line arguments.

`pebble auth logout <provider>` removes the saved entry. If an environment
variable still supplies credentials, Pebble reports that source. Logout can
also remove an exact provider id that was deleted from `models.toml`.

Credential writes use a private temporary file, an atomic replacement, and a
process lock. New application directories use mode `0700` on Unix; credential
files use mode `0600`. Saved API keys are stored as plaintext with these file
permissions. Pebble does not read Pi or Codex credential files.

To use a custom environment variable, put this in `auth.json`:

```json
{
  "version": 1,
  "providers": {
    "my-proxy": {
      "type": "env",
      "variable": "MY_PROXY_API_KEY"
    }
  }
}
```

The provider id must match the catalog. A saved API key uses
`{"type":"api_key","key":"YOUR_API_KEY"}` as its provider entry; the login
commands write this format for you.

For a provider with `auth = { type = "headers" }`, an explicit header entry can
combine environment references and literal values:

```json
{
  "version": 1,
  "providers": {
    "my-proxy": {
      "type": "headers",
      "headers": {
        "x-proxy-key": { "type": "env", "variable": "MY_PROXY_API_KEY" },
        "x-project": { "type": "literal", "value": "project-id" }
      }
    }
  }
}
```

Explicit headers are also supported for `auth.type = "none"`. An explicit
`authorization` header must include its complete value, including `Bearer `
when required. Keep secret headers in `auth.json`; catalog `default_headers`
are ordinary configuration and are not redacted.

## Custom models and endpoints

Catalog layers apply in this order, with later layers taking precedence:

1. The built-in catalog in the pinned `lithos-llm` dependency.
2. `models.toml` in the Pebble configuration directory.
3. `PEBBLE_<PROVIDER>_BASE_URL` environment overrides.

The TOML file uses the existing `lithos-llm` catalog schema. To move an existing
provider to a compatible endpoint while preserving its model metadata:

```toml
schema_version = 1

[providers.openai]
base_url = "http://localhost:8080/v1"
```

Or set `PEBBLE_OPENAI_BASE_URL`. Existing credentials for that provider apply
to the configured endpoint.

To add a provider using the OpenAI Responses protocol:

```toml
schema_version = 1

[providers.my-proxy]
display_name = "My proxy"
adapter = "openai"
codec = "openai-responses"
base_url = "http://localhost:8080/v1"
auth = { type = "bearer" }
default_model = "coding"

[providers.my-proxy.metadata.pebble]
profile = "openai"

[providers.my-proxy.models.coding]
display_name = "My coding model"
api_model = "your-api-model-id"
capabilities = { text = true, tools = true }
limits = { context_tokens = 128000, max_output_tokens = 8192 }
```

Replace the endpoint, API model id, capabilities, limits, and profile with the
values for your model. Use `pebble auth login my-proxy` or configure its
environment source. Select it with `/model my-proxy/coding` or
`pebble exec --model my-proxy/coding "your prompt"`.

`metadata.pebble.profile` selects the coding tools and instructions the model
expects. Supported identifiers are `anthropic`, `claude-5`, `openai`, `gemini`,
`kimi`, `gpt56`, and `gpt6`. Model metadata overrides provider metadata per field.
Pebble rejects unknown profiles before use. A custom catalog entry cannot add
a provider protocol that the compiled client does not support.

Catalog loading stays local. This version has no automatic catalog refresh,
subscription OAuth login, token refresh, or command execution from `auth.json`.
