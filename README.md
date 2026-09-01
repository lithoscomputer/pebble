# Pebble

Pebble is a coding agent as a library. The workspace also contains
`pebble-agent`, its provider-neutral agent loop.

It owns the turn loop between a model and a machine: it calls the model,
streams what comes back, runs the tools the model asks for, keeps the history
those calls have to stay paired in, summarizes that history before it outgrows
the context window, and publishes an event for everything it does. It does not
own transport, storage, credentials, or process isolation. Those belong to the
application that embeds it.

## How the pieces sit

Four layers, each with one job.

At the bottom is [lithos-llm](https://github.com/lithoscomputer/lithos-llm):
the `Client`, the provider adapters and codecs, the catalog of models, and the
retry middleware. It knows how to talk to a provider.

Above it is `pebble-agent`: model turns, generic tool execution, conversation
history, steering, follow-up, cancellation, and a small lifecycle event stream.
It accepts a model service and tools from its caller. It knows how to run an
agent, but it knows nothing about coding.

Above that is `pebble`: coding profiles, filesystem and shell tools, memory,
skills, context compaction policy, subagents, and the durable coding event
stream. It knows how to turn an agent into a coding agent.

On top is the application — a CLI, a server, a workflow engine, the example in
this repository. It builds the `Client`, supplies the `Environment` the tools
act through, subscribes to events or installs a sink for them, decides what the
agent is allowed to do, and drives the input. It knows what the agent is for.

Dependencies point down. `pebble` depends on `pebble-agent`, which depends on
the `lithos-llm` runtime without enabling provider features. The coding layer
also uses `lithos-llm` directly for model selection and coding-specific summary
calls. Pebble never builds a client and never opens a socket.

## A session

```rust,no_run
use std::error::Error;
use std::sync::Arc;

use lithos_llm::Client;
use lithos_llm::catalog::Catalog;
use lithos_llm::credentials::EnvironmentCredentials;
use lithos_llm::middleware::{RetryMiddleware, RetryPolicy};
use pebble::events::RetryEventObserver;
use pebble::{CodingSession, LocalEnvironment, ShutdownReason};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let policy = RetryPolicy::exponential().max_attempts(4);
    let client = Client::builder()
        .catalog(Catalog::builder().with_builtin().build()?)
        .credentials(EnvironmentCredentials::conventional())
        .middleware(RetryMiddleware::new(policy).observer(RetryEventObserver))
        .build()?
        .client;

    let mut session = CodingSession::builder(
        client,
        Arc::new(LocalEnvironment::new("/path/to/work")),
    )
        .model("claude-sonnet-5")
        .build()
        .await?;

    let mut events = session.subscribe();
    let outcome = session.prompt("fix the failing test").await?;
    session.shutdown(ShutdownReason::Completed).await?;

    println!("{:?} — first event: {:?}", outcome.text(), events.try_recv());
    Ok(())
}
```

`CodingSessionBuilder::build().await` returns a ready session. Resource loading
and system-prompt construction happen inside the build. There is no separate
initialization step to remember. `prompt` returns a `PromptOutcome` with the final
message, text, token usage, cost, and timing.

The crate root contains the normal coding-session path and the environment
contract. Durable event types are in `pebble::events`. Tool contracts and
built-in tools are in `pebble::tools`. History and loaded resources are in
`pebble::resources`. Lower-level session construction is in
`pebble::advanced`.

Take a `CodingSessionControlHandle` before calling `prompt` when another task
must `steer`, `follow_up`, `abort`, or `wait_for_idle` while the prompt holds
the mutable session borrow. These are the same control verbs as the generic
agent API.

`examples/coding_agent.rs` is the same thing at full size: it renders the event
stream, steers one prompt while it works, interrupts the next, and reports what
the session used. Run it with `mise run dev`.

## What an application has to supply

**A client, with retry middleware.** Pebble takes a built `lithos_llm::Client`
and cannot add middleware to one, so the application installs the retry
middleware when it builds the client, with pebble's `RetryEventObserver` on it.
The observer is what puts the client's own retries onto the session's event
stream; without it a session still runs correctly and simply never reports one.
`CodingSessionOptions::turn_replay` controls the separate replay that happens
after a response stream opens. The two policies can share settings, but they
have different ownership and do not have to match.

The model layer is available under `pebble::advanced::llm`. The same module
exports the `async_trait` attribute used by the seams and `CancellationToken`,
so an application can use `pebble` alone in its manifest. A direct
`lithos-llm` dependency is also appropriate when the application configures
providers.

**Credentials.** They belong to the client, and lithos-llm resolves them per
call — `EnvironmentCredentials::conventional()` reads the usual variables
(`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, and the rest), and an application with
a vault implements lithos-llm's `CredentialProvider` instead. Pebble never sees
a key.

**An `Environment`.** Every tool acts through this one seam: reading and
writing files, listing a directory, searching by content or by name, and
running a command as Bash source. `LocalEnvironment` does that on this machine.
An application working in a container, a VM, or a remote workspace implements
the trait over that instead, and nothing else in the crate changes.
`pebble::test_support::MockEnvironment` stands in for a machine in tests,
behind the `test-util` feature.

**Somewhere for the events to go, if they matter.**
`CodingSession::subscribe` hands out a bounded broadcast receiver, which is
lossy for a reader that falls behind: right for a terminal, wrong for a ledger.
The stream ends with the session rather than with the session value: once
`CodingSession::shutdown` has returned, a reader looping until
`RecvError::Closed` finishes, so a renderer task can be joined before the
session is dropped. An application that must see every event installs a
`pebble::events::EventSink` instead. Each event is recorded there, in sequence,
before any subscriber sees it. A sink that refuses one stops the prompt,
because a session that cannot record what it did is worse than one that stops.

**A policy, if the agent should not do everything.** Pebble installs none: with
no `pebble::tools::ToolAccessPolicy` and no
`pebble::tools::ToolHookCallback`, every registered tool is exposed and every
call runs. `pebble::tools::PermissionLevel` and its table are there to build a
policy out of, not a policy pebble applies.

Optional seams follow the same rule. Pebble ships no implementation and
advertises no tool without one: an `advanced::HumanInputProvider` (no provider,
no question tool), an `advanced::SearchProvider` (no provider, no
`web_search`), an `advanced::Redactor` for process output, and an
`advanced::SessionFactory` for subagents.

## The generic agent API

Use `pebble-agent` directly when the application supplies its own tools and
does not need coding profiles or environment policy. Pebble also exposes the
exact crate as `pebble::advanced::agent`:

```rust,no_run
use pebble::advanced::agent::{Agent, Tool};
use serde_json::json;

# async fn example(client: lithos_llm::Client) -> Result<(), Box<dyn std::error::Error>> {
let inspect = Tool::function(
    "inspect",
    "Inspect a named value",
    json!({"type": "object"}),
    |_context, arguments| async move {
        Ok(format!("inspected {}", arguments["name"]).into())
    },
);

let mut agent = Agent::builder(client, "provider/model")
    .system_prompt("Use tools when they help.")
    .tools([inspect])
    .build()?;

agent.follow_up("Now explain the result");
let outcome = agent.prompt("Inspect the parser").await?;
println!("{}", outcome.text());
# Ok(())
# }
```

The public verbs are `prompt`, `steer`, `follow_up`, `abort`, and
`wait_for_idle`. `Agent::snapshot` returns an owned immutable view instead of
exposing mutable agent state. A `ContextTransform` can summarize or rewrite
history before a model turn without adding coding policy to the agent crate.

The coding layer has the same concise path for application tools:

```rust
use pebble::tools::RegisteredTool;
use serde_json::json;

let inspect = RegisteredTool::function(
    "inspect",
    "Inspect an application value",
    json!({"type": "object"}),
    |_context, arguments| async move {
        Ok(format!("inspected {}", arguments["name"]))
    },
);

# let _ = inspect;
```

Pass it to `CodingSessionBuilder::tools`. Pebble records its source as
`ToolSource::Application` and keeps coding-layer policy and event behavior
around its execution.

## Which harness a session runs

Pebble never guesses. The model selector resolves through the client's catalog,
and the resolved entry's `metadata.pebble.profile` names one of the six
harnesses pebble ships — Claude, Claude 5, Gemini CLI, OpenAI, Codex, Kimi
Code. That profile decides the system prompt, the starting tools, and the
vocabulary those tools are named in, because a model trained inside a coding
harness expects that harness back. A model whose catalog entry names no profile
is a build error, not a session that runs with the wrong prompt.

## Stability

The serialized form of `pebble::events::CodingSessionEvent` and
`pebble::events::CodingEvent` is public API. So is
`pebble::resources::SessionRecord`, which carries a format version. Evolution
is additive: new variants and new optional fields. Consumers should ignore
members they do not know and tolerate variants they do not know.

Ignoring an unknown member is free; tolerating an unknown *variant* is the
reader's own work, because a variant a build has never heard of fails the whole
envelope with it. The crate documentation shows the pattern: read the envelope
with the event held as raw JSON, then parse the payload on its own.

## Setup

Pebble depends on `lithos-llm` through a path dependency. Clone `lithos-llm`
as a sibling directory of this repository first.

Install the locked tools and prepare the repository:

```sh
mise trust
mise install --locked --jobs=1
mise run setup
```

Use the repository tasks for development and verification:

```sh
mise run dev     # run the coding-agent example (needs a provider key)
mise run test    # the test suite, which needs neither key nor network
mise run check   # the complete routine gate
```

See [DEVELOPING.md](DEVELOPING.md) for the complete development workflow.

Routine and nightly checks run on macOS arm64, Linux x86_64, and Linux arm64.
GitHub cannot run them yet, because the workflows do not check out the
`lithos-llm` path dependency. See [DEVELOPING.md](DEVELOPING.md).
