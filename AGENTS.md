# Repository Instructions

## Project purpose

Pebble is the coding-agent loop as a library crate, built on `lithos-llm`. It
owns the turn loop between a model and a machine: model calls, tool execution,
history and compaction, the event stream, the control plane, and subagents. It
does not own transport, storage, credentials, or process isolation, which
belong to the application that embeds it.

The authoritative documents are `README.md` (what pebble is and what an
application supplies) and `DEVELOPING.md` (setup, tasks, tests, and policy).

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
