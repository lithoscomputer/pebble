# Architecture

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
depends on `lithos-llm` for model access. The
coding layer also uses `lithos-llm` directly for model selection and
coding-specific summary calls. The application builds the model client.
Optional coding-agent integrations can open connections for search and
Model Context Protocol (MCP) servers.

## The command-line application

`pebble-cli-core` contains the interactive session, batch execution, terminal
rendering, settings, and credential storage. `pebble-cli` parses arguments and
starts that application. See the [CLI guide](cli.md#embedding-the-command-line)
for reuse from another Rust application.

## Coding profiles

A coding profile supplies the system prompt, starting tools, and tool names
for a model. The model selector resolves through the client's catalog, and
`metadata.agent.profile` selects the profile. A missing or unsupported profile
is a build error.

See [models and credentials](models-and-credentials.md#custom-models-and-endpoints)
for supported profile identifiers and custom catalog entries.

Return to the [README](../README.md).
