# Pebble

Pebble is the loop a coding agent runs on, as a library.

It owns the turn loop between a model and a machine: it calls the model,
streams what comes back, runs the tools the model asks for, keeps the history
those calls have to stay paired in, summarizes that history before it outgrows
the context window, and publishes an event for everything it does. It does not
own transport, storage, credentials, or process isolation. Those belong to the
application that embeds it.

## How the pieces sit

Three layers, each with one job.

At the bottom is [lithos-llm](https://github.com/lithoscomputer/lithos-llm):
the `Client`, the provider adapters and codecs, the catalog of models, and the
retry middleware. It knows how to talk to a provider.

In the middle is pebble: the session loop, history and compaction, the context
window accounting, the tool registry and dispatch, the event stream, the
steering and interrupt control plane, subagents, the coding tools, skills,
memory, and the six agent profiles. It knows how to run an agent.

On top is the application — a CLI, a server, a workflow engine, the example in
this repository. It builds the `Client`, supplies the `Environment` the tools
act through, subscribes to events or installs a sink for them, decides what the
agent is allowed to do, and drives the input. It knows what the agent is for.

Nothing reaches around a layer. Pebble never builds a client and never opens a
socket; an application never sees a provider's wire format.

## A session

```rust,no_run
use std::error::Error;
use std::sync::Arc;

use lithos_llm::Client;
use lithos_llm::catalog::Catalog;
use lithos_llm::credentials::EnvironmentCredentials;
use lithos_llm::middleware::{RetryMiddleware, RetryPolicy};
use pebble::{LocalEnvironment, RetryEventObserver, Session, ShutdownReason};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let policy = RetryPolicy::exponential().max_attempts(4);
    let client = Client::builder()
        .catalog(Catalog::builder().with_builtin().build()?)
        .credentials(EnvironmentCredentials::conventional())
        .middleware(RetryMiddleware::new(policy).observer(RetryEventObserver))
        .build()?
        .client;

    let mut session = Session::builder(client)
        .model("claude-sonnet-5")
        .environment(Arc::new(LocalEnvironment::new("/path/to/work")))
        .build()?;

    let mut events = session.subscribe();
    session.initialize().await?;
    let answer = session.run("fix the failing test").await?;
    session.shutdown(ShutdownReason::Completed).await?;

    println!("{answer:?} — first event: {:?}", events.try_recv());
    Ok(())
}
```

`examples/coding_agent.rs` is the same thing at full size: it renders the event
stream, steers one run while it works, interrupts the next, and reports what
the session used. Run it with `mise run dev`.

## What an application has to supply

**A client, with retry middleware.** Pebble takes a built `lithos_llm::Client`
and cannot add middleware to one, so the application installs the retry
middleware when it builds the client, with pebble's `RetryEventObserver` on it.
The observer is what puts the client's own retries onto the session's event
stream; without it a session still runs correctly and simply never reports one.
Give `SessionOptions::retry_policy` the same policy: pebble replays a turn
itself only when a stream fails *after* the model produced visible output,
which no middleware can reconnect underneath a reader, and one failure should
be spaced the same way whichever layer handles it.

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

**Somewhere for the events to go, if they matter.** `Session::subscribe` hands
out a bounded broadcast receiver, which is lossy for a reader that falls
behind: right for a terminal, wrong for a ledger. An application that must see
every event installs an `EventSink` instead — each event is recorded there, in
sequence, before any subscriber sees it, and a sink that refuses one stops the
run, because a session that cannot record what it did is worse than one that
stops.

**A policy, if the agent should not do everything.** Pebble installs none: with
no `ToolAccessPolicy` and no `ToolHookCallback`, every registered tool is
exposed and every call runs. `PermissionLevel` and its table are there to build
a policy out of, not a policy pebble applies.

Optional seams follow the same rule — pebble ships no implementation and
advertises no tool without one: a `HumanInputProvider` (no provider, no
question tool), a `SearchProvider` (no provider, no `web_search`), a `Redactor`
for the process output the event stream carries, and a `SessionFactory` for
subagents.

## Which harness a session runs

Pebble never guesses. The model selector resolves through the client's catalog,
and the resolved entry's `metadata.pebble.profile` names one of the six
harnesses pebble ships — Claude, Claude 5, Gemini CLI, OpenAI, Codex, Kimi
Code. That profile decides the system prompt, the starting tools, and the
vocabulary those tools are named in, because a model trained inside a coding
harness expects that harness back. A model whose catalog entry names no profile
is a build error, not a session that runs with the wrong prompt.

## Stability

The serialized form of `SessionEvent` and `AgentEvent` is public API, and so is
`SessionRecord`, which carries a format version. Evolution is additive: new
variants and new optional fields. Consumers should ignore members they do not
know and tolerate variants they do not know.

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
