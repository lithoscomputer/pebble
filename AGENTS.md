# Repository Instructions

## Project purpose

Pebble contains two agent library crates, a CLI library, and a binary,
built on `lithos-llm`. `pebble-agent` owns the provider-neutral turn loop:
model calls, generic tool execution, conversation history, lifecycle events,
steering,
follow-up, and cancellation. `pebble-coding-agent` is the coding-agent layer.
It adds coding profiles, environment-backed tools, memory, skills, compaction
policy, durable events, and subagents. `pebble-cli-core` owns the command-line
application; `pebble-cli` is its `pebble` binary. The application builds the client, supplies the environment,
and drives user input. The libraries do not own transport, storage, credentials,
or process isolation, which belong to the application that embeds them.

Start with `README.md` for purpose, installation, scope, and compatibility.
`docs/embedding.md` describes application responsibilities and API behavior;
`docs/architecture.md` describes the layers. `DEVELOPING.md` owns contributor
setup, tasks, tests, and policy.

## README.md

Keep the README focused on what Pebble does, who it serves, how to start,
and where to go next. Use a complete library example near the top.

- Put API detail in rustdoc and guides in `docs/`; contributor setup belongs in
  `DEVELOPING.md`.
- Update the README when its description, examples, installation, scope,
  compatibility, or related-project links change.
- Use absolute GitHub URLs for repository-file links in the README.
- Keep Markdown examples complete, with no rustdoc-only hidden lines.
- README and embedding-guide Rust examples are doctests. Run
  `mise run test:doc` after changing them.

## Rust style

Before changing Rust code, configuration, project structure, or tests:

1. Run `bin/style-guides prepare`.
2. Read `.ai/style-guides/rust-style-guide/SKILL.md` completely.
3. Read each workflow and policy page that the skill routes for the task.

Project requirements and accepted architecture decisions override general
style-guide defaults.

## Repository tasks

- Use `mise run dev` for the normal development path.
- Use `mise run test` for the routine test suite.
- Use `mise run check` for the complete routine verification gate.
- Use `mise run check:nightly` for the extended verification gate.
- Use `mise run fmt` to format Rust with the pinned nightly formatter.

## Safety

- Never install packages less than 24 hours old.
- Never force push, including with `--force-with-lease`.
- Never amend commits. Create a new commit instead.

## Working documents

Save plans under `.ai/plans/` and reviews under `.ai/reviews/`.
