# Pebble

Build agents in Rust that call models, run tools, and keep a conversation going across turns.

[![CI](https://github.com/lithoscomputer/pebble/actions/workflows/routine.yml/badge.svg)](https://github.com/lithoscomputer/pebble/actions/workflows/routine.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/lithoscomputer/pebble/blob/main/LICENSE)

```rust,no_run
use std::sync::Arc;
use lithos_llm::Client;
use pebble_coding_agent::environment::LocalEnvironment;
use pebble_coding_agent::tools::PermissionLevel;
use pebble_coding_agent::{CodingAgent, ShutdownReason};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::from_env()?.client;
    let mut agent = CodingAgent::builder(client, Arc::new(LocalEnvironment::new(".")))
        .model("claude-sonnet-5")
        .permission_level(PermissionLevel::ReadOnly)
        .build()
        .await?;
    let report = agent.prompt("Explain how this repository is organized").await;
    let reason = if report.result.is_ok() {
        ShutdownReason::Completed
    } else {
        ShutdownReason::Error
    };
    agent.shutdown(reason).await?;
    if let Some(text) = report.result?.text {
        println!("{text}");
    }
    Ok(())
}
```

This example reads the current directory with a read-only tool policy.
It needs `ANTHROPIC_API_KEY`; see [installation](#install) for the dependencies.

Part of [Lithos](#part-of-lithos), a family of Rust building blocks for AI tools.

## Why does this exist?

An agent needs more than a model request. It must execute tool calls, keep their
results in conversation history, and handle new input and cancellation.
Pebble provides that loop and a coding-agent layer, so applications can reuse
them with their own interface, storage, and execution environment.

## Who is it for?

**Use it if** you're building a coding assistant, a workflow that runs agents,
or a service that gives models access to your own tools.

**Look elsewhere if** you only need model requests; use
[lithos-llm](https://github.com/lithoscomputer/lithos-llm) directly.
For a ready-to-run terminal coding agent, this repository also ships the
[`pebble` command](#use-it-from-the-command-line).

## What does it do?

- Runs model turns and tool calls through a provider-neutral agent loop.
- Accepts steering, queued follow-ups, and cancellation while an agent works.
- Adds coding tools, model-specific prompts, project instructions, and skills.
- Summarizes long conversations and supports saved sessions and subagents.
- Emits lifecycle events, tool results, and usage for application views and storage.

**Your application supplies** the model client and credentials, the environment
where tools run, permission policy, and any persistent storage or user interface.
The libraries do not provide process isolation. The CLI supplies a local
application; containers and remote workspaces need an environment adapter.
See [embedding responsibilities](https://github.com/lithoscomputer/pebble/blob/main/docs/embedding.md#what-an-application-has-to-supply).

| Crate | Use it when |
| --- | --- |
| `pebble-agent` | You supply the tools and need the generic agent loop. |
| `pebble-coding-agent` | You need coding tools, profiles, memory, compaction, and subagents. |
| `pebble-cli-core` | You want to reuse the CLI's sessions, terminal rendering, and approvals. |
| `pebble-cli` | You want to install the `pebble` binary. |

Optional features on `pebble-coding-agent` are off by default:

| Feature | Enables |
| --- | --- |
| `mcp` | Tools from Model Context Protocol (MCP) servers. |
| `search-providers` | Brave and Venice implementations of the web-search interface. |
| `sandbox-driver` | An environment adapter for [sandbox-driver](https://github.com/lithoscomputer/sandbox-driver). |
| `test-util` | Test support for embedding applications. |
| `sandbox-driver-test-util` | Test support for the sandbox adapter. |

## How do I use it?

### Install

The workspace is not configured for crates.io publication. Add Git dependencies
to your application's `Cargo.toml`. These dependencies run the opening example:

```toml
[dependencies]
pebble-coding-agent = { git = "https://github.com/lithoscomputer/pebble", branch = "main" }
lithos-llm = { git = "https://github.com/lithoscomputer/lithos-llm", branch = "main" }
tokio = { version = "1.49.0", features = ["macros", "rt-multi-thread"] }
```

Use `branch = "main"` for `lithos-llm`, matching Pebble's dependency source.
Your application's `Cargo.lock` records the commits it uses.

Save the opening example as `src/main.rs`, set `ANTHROPIC_API_KEY` in your
environment, then run `cargo run` from the directory you want the agent to inspect.
The example uses a Tokio runtime and calls a live model.

### Use it as a library

Start with `CodingAgent` for coding tasks, or `pebble_agent::Agent` for your own
tool set. The opening example uses a minimal client. For application use, add
retry middleware and subscribe to events as shown in the
[embedding guide](https://github.com/lithoscomputer/pebble/blob/main/docs/embedding.md#a-coding-agent).

Continue with these guides:

- [Custom tools and the generic agent](https://github.com/lithoscomputer/pebble/blob/main/docs/embedding.md#the-generic-agent-api)
- [Permissions and approvals](https://github.com/lithoscomputer/pebble/blob/main/docs/embedding.md#tool-permissions)
- [Events, durable storage, and resuming sessions](https://github.com/lithoscomputer/pebble/blob/main/docs/embedding.md#events-and-durability)
- [Architecture and coding profiles](https://github.com/lithoscomputer/pebble/blob/main/docs/architecture.md)

For API documentation, run `cargo doc --open -p pebble-coding-agent` from your
application. Use `-p pebble-agent` for the generic loop.

### Use it from the command line

Install from Git, then start an interactive session or run a single prompt:

```sh
cargo install --git https://github.com/lithoscomputer/pebble --branch main --locked pebble-cli
pebble
pebble exec --permission read-only "explain this repository"
```

On first interactive use, Pebble offers provider setup and model selection.
`pebble exec` uses saved settings or environment credentials and does not open
a setup prompt. Permissions control tool access; they do not create an
operating-system sandbox.

See the [CLI guide](https://github.com/lithoscomputer/pebble/blob/main/docs/cli.md),
[interactive guide](https://github.com/lithoscomputer/pebble/blob/main/docs/interactive.md),
and [models and credentials](https://github.com/lithoscomputer/pebble/blob/main/docs/models-and-credentials.md).

## Status and compatibility

- **Version:** 0.1.0; consumed through Git dependencies.
- **Minimum Rust version:** 1.88, edition 2024.
- **CI platforms:** macOS arm64, Linux x86_64, and Linux arm64.
- **Public formats:** coding-agent events and session records. The current
  session-record format is version 5; older records are rejected. See the
  [compatibility contract](https://github.com/lithoscomputer/pebble/blob/main/docs/embedding.md#stability).

## Part of Lithos

| Project | Role |
| --- | --- |
| [lithos-llm](https://github.com/lithoscomputer/lithos-llm) | Model access: one client and catalog across LLM providers. |
| **[pebble](https://github.com/lithoscomputer/pebble)** | **Agent loop, plus a coding agent and CLI built on it.** |
| [sandbox-driver](https://github.com/lithoscomputer/sandbox-driver) | Sandboxes behind one interface. |
| [petri](https://github.com/lithoscomputer/petri) | Workflow engine built on the libraries above. |

## Contributing

Development setup, tests, and conventions are in
[DEVELOPING.md](https://github.com/lithoscomputer/pebble/blob/main/DEVELOPING.md).
README and embedding-guide Rust examples are checked by the documentation tests.

## License

[MIT](https://github.com/lithoscomputer/pebble/blob/main/LICENSE).
