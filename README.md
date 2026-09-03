# Pebble

Pebble is a two-package agent library. `pebble-agent` is the provider-neutral
agent loop. `pebble-coding-agent` builds a coding agent on top of it.

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

Above that is `pebble-coding-agent`: coding profiles, filesystem and shell
tools, memory, skills, context compaction policy, subagents, and the durable
coding event stream. Its primary type is `CodingAgent`.

On top is the application — a CLI, a server, a workflow engine, the example in
this repository. It builds the `Client`, supplies the `Environment` the tools
act through, subscribes to events or installs a sink for them, decides what the
agent is allowed to do, and drives the input. It knows what the agent is for.

Dependencies point down. `pebble-coding-agent` depends on `pebble-agent`, which
depends on the `lithos-llm` runtime without enabling provider features. The
coding layer also uses `lithos-llm` directly for model selection and
coding-specific summary calls. Neither package builds a client or opens a
socket.

## A coding agent

```rust,no_run
use std::error::Error;
use std::sync::Arc;

use lithos_llm::Client;
use lithos_llm::catalog::Catalog;
use lithos_llm::credentials::EnvironmentCredentials;
use lithos_llm::middleware::{RetryMiddleware, RetryPolicy};
use pebble_coding_agent::events::RetryEventObserver;
use pebble_coding_agent::environment::LocalEnvironment;
use pebble_coding_agent::{CodingAgent, ShutdownReason};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let policy = RetryPolicy::exponential().max_attempts(4);
    let client = Client::builder()
        .catalog(Catalog::builder().with_builtin().build()?)
        .credentials(EnvironmentCredentials::conventional())
        .middleware(RetryMiddleware::new(policy).observer(RetryEventObserver))
        .build()?
        .client;

    let mut agent = CodingAgent::builder(
        client,
        Arc::new(LocalEnvironment::new("/path/to/work")),
    )
        .model("claude-sonnet-5")
        .build()
        .await?;

    let mut events = agent.subscribe();
    let outcome = agent.prompt("fix the failing test").await?;
    agent.shutdown(ShutdownReason::Completed).await?;

    println!("{:?} — first event: {:?}", outcome.text(), events.try_recv());
    Ok(())
}
```

`CodingAgentBuilder::build().await` returns a ready coding agent. Resource loading
and system-prompt construction happen inside the build. There is no separate
initialization step to remember. `prompt` returns a `PromptOutcome` with the final
message, text, token usage, cost, and timing.

The crate root contains the normal coding-agent path and the environment
contract. The environment a session acts through is in
`pebble_coding_agent::environment`. Durable event types are in
`pebble_coding_agent::events`. Tool contracts, built-in tools, and the standalone
tool runner are in `pebble_coding_agent::tools`. Durable session state is in
`pebble_coding_agent::state`. Optional application services are in
`pebble_coding_agent::extensions`. Subagent configuration is in
`pebble_coding_agent::subagents`. The internal runtime is not public.

Take a `CodingAgentControlHandle` before calling `prompt` when another task
must steer, follow up, cancel compaction, abort a prompt, close the agent, or
wait for idle while the prompt holds the mutable agent borrow. `abort` ends the
active prompt and leaves the agent reusable. `close` is permanent. The task
that owns the agent then calls `shutdown` to publish the terminal event and
join owned tasks.

The control handle also exposes queued input. `pending_input` clones steering
and follow-ups in queue order. `take_pending_input` removes and returns them.
An application can use this to restore unsent text to an editor after it aborts
a prompt.

`CodingInput` and `SteeringMessage::from_content` accept ordered
`lithos_llm::types::ContentPart` values. Pebble keeps those parts in history,
stored records, queued input, and input events. This includes images, audio,
and documents. The stable `text` event member remains available to text-only
renderers. `InputSource` distinguishes a direct prompt, a follow-up,
agent-generated input, and input from another application integration.

Use `CodingAgent::observe` when a view needs a complete starting point and
live updates. It returns a `CodingAgentSnapshot` and an event receiver at one
committed event cursor. Apply the snapshot first. Discard queued events at or
below `snapshot.committed_event_seq()`. Then apply later events in sequence.
The snapshot includes history, pending input, state, route, profile, memory,
skills, tools, and the latest context-window measurement.

Automatic context compaction remains policy-driven. An application can also
call `CodingAgent::compact(CompactionOptions)` while the agent is idle. The
operation returns a structured `CompactionOutcome`. A completed summary is a
dedicated `Message::Compaction` turn with its reason, counts, usage, and cost.
Each started compaction ends with a completed, failed, or cancelled event.
Pass a token to `compact_with_cancellation`, or use
`CodingAgentControlHandle::cancel_compaction` from another task.

`crates/pebble-coding-agent/examples/coding_agent.rs` is the same thing at full
size. It renders the event stream, steers one prompt while it works, interrupts
the next, and reports what the agent used. Run it with `mise run dev`.

## What an application has to supply

**A client, with retry middleware.** Pebble takes a built `lithos_llm::Client`
and cannot add middleware to one, so the application installs the retry
middleware when it builds the client, with Pebble's `RetryEventObserver` on it.
The observer is what puts the client's own retries onto the agent's event
stream. Without it, an agent still runs correctly and does not report retries.
`CodingAgentOptions::turn_replay` controls the separate replay that happens
after a response stream opens. Its default waits one second before the first
replay, doubles the wait each time, caps a wait at sixty seconds, and jitters
each wait. The two policies can share settings, but they have different
ownership and do not have to match.

Applications use `lithos-llm` directly to build clients and use its public
model types. The Pebble packages do not re-export their dependencies.

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
`pebble_coding_agent::test_support::MockEnvironment` stands in for a machine in tests,
behind the `test-util` feature.

**Somewhere for the events to go, if they matter.**
`CodingAgent::subscribe` hands out a bounded broadcast receiver, which is
lossy for a reader that falls behind: right for a terminal, wrong for a ledger.
The stream ends with the agent rather than with the agent value: once
`CodingAgent::shutdown` has returned, a reader looping until
`RecvError::Closed` finishes, so a renderer task can be joined before the
agent is dropped. An application that must see every event installs a
`pebble_coding_agent::events::EventSink` instead.

One root agent and all its subagents write directly to one ordered stream. Each
`CodingAgentEvent` names that stream with `stream_id`, names its producer with
`session_id`, and has a contiguous `seq` within the stream. The stream identity
and numbering continue when the root session resumes. A sink receives every
event, including streaming deltas, in sequence before a live subscriber sees
it. Treat `(event.stream_id(), event.seq)` as the idempotency key because a
process can stop after storage commits an event but before its latest session
record is saved. If the log and record do not share one transaction, read the
log's highest sequence and call `SessionRecord::advance_event_cursor` before
resume. This prevents the resumed stream from reusing an already committed
position.

The producer queue is bounded by `event_capacity`. A full queue, a refused sink
write, or a write longer than `event_sink_timeout` stops the stream, cancels
active work, closes the session tree, and returns an `event_stream` error. It
never lets the agent continue with a partial ledger. Building an agent and
finishing a prompt both wait for their events to reach the sink. Use
`CodingAgent::flush_events` for an explicit durability boundary and
`CodingAgent::committed_event_seq` to read its committed cursor.

**Tool middleware, if the agent should not do everything.** Pebble installs no
application policy. With no middleware, every registered tool is exposed and
every valid call runs. Add each layer with `CodingAgentBuilder::tool_middleware`.
The first added layer is outermost.

Permissions use the same middleware path as logging, metering, or other tool
behavior. `PermissionMiddleware` filters discovery and checks every call again.
The call check sees the validated arguments. Several permission layers compose
by narrowing access.

```rust
# use std::sync::Arc;
# use pebble_coding_agent::{CodingAgentBuilder, CodingAgentOptions};
use pebble_coding_agent::tools::{
    PermissionLevel, PermissionLevelPolicy, PermissionMiddleware,
};

# fn configure(builder: CodingAgentBuilder) -> CodingAgentBuilder {
let level = PermissionLevel::ReadWrite;
let permissions = PermissionMiddleware::new(Arc::new(
    PermissionLevelPolicy::new(level),
));

builder
    .options(CodingAgentOptions::default().with_permission_level(level))
    .tool_middleware(Arc::new(permissions))
# }
```

`CodingAgentOptions::with_permission_level` records the selected value in the
session. The middleware enforces it. Add a `ToolApprovalService` with
`PermissionMiddleware::with_approval` when calls that are not auto-approved
should remain visible and ask for approval. Without an approval service, those
tools are hidden and direct attempts are denied.

Optional seams follow the same rule. Pebble ships no implementation and
advertises no tool without one: an `extensions::HumanInputProvider` (no
provider, no question tool), an `extensions::SearchProvider` (no provider, no
`web_search`), an `extensions::Redactor` for process output and failed tool
call messages, and a
`subagents::SubagentOptions` for subagents.

## The generic agent API

Use `pebble-agent` directly when the application supplies its own tools and
does not need coding profiles or environment policy:

```rust,no_run
use pebble_agent::{Agent, Tool};
use serde_json::json;

# async fn example(client: lithos_llm::Client) -> Result<(), Box<dyn std::error::Error>> {
let inspect = Tool::function(
    "inspect",
    "Inspect a named value",
    json!({"type": "object"}),
    |_context, arguments| async move {
        Ok(format!("inspected {}", arguments["name"]).into())
    },
)?;

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
exposing mutable agent state. Its control handle can also close the agent and
clone or drain pending steering and follow-ups.

The coding layer has the same concise path for application tools:

```rust
use pebble_coding_agent::tools::RegisteredTool;
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

Pass it to `CodingAgentBuilder::tools`. Pebble records its source as
`ToolSource::Application`. The call then uses the same middleware, event,
output, and cancellation path as a built-in tool.

## Which harness a session runs

Pebble never guesses. The model selector resolves through the client's catalog,
and the resolved entry's `metadata.pebble.profile` names one of the six
harnesses pebble ships — Claude, Claude 5, Gemini CLI, OpenAI, Codex, Kimi
Code. That profile decides the system prompt, the starting tools, and the
vocabulary those tools are named in, because a model trained inside a coding
harness expects that harness back. A model whose catalog entry names no profile
is a build error, not a session that runs with the wrong prompt.

## Stability

The serialized form of `pebble_coding_agent::events::CodingAgentEvent` and
`pebble_coding_agent::events::CodingEvent` is public API. Event evolution is
additive: new variants and new optional fields. Consumers should ignore members
they do not know and tolerate variants they do not know.

`pebble_coding_agent::state::SessionRecord` is also public API. Its format
version changes when a stored shape changes. Version 2 stores user and steering
content as ordered content parts and stores compaction as its own turn kind.
`SessionRecord::migrate` upgrades version 1 records before resume.

Ignoring an unknown member is free; tolerating an unknown *variant* is the
reader's own work, because a variant a build has never heard of fails the whole
envelope with it. The crate documentation shows the pattern: read the envelope
with the event held as raw JSON, then parse the payload on its own.

## Setup

Pebble depends on `lithos-llm` as a git dependency pinned to one commit. The
repository is private and fetched over ssh, so load an ssh key that can read
it first.

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
GitHub cannot run them yet, because the runners have no key for the private
`lithos-llm` repository. See [DEVELOPING.md](DEVELOPING.md).
