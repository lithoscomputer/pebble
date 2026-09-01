# Repository Instructions

## Project purpose

Pebble is a two-crate library workspace built on `lithos-llm`.
`pebble-agent` owns the provider-neutral turn loop: model calls, generic tool
execution, conversation history, lifecycle events, steering, follow-up, and
cancellation. `pebble` is the coding-agent facade. It adds coding profiles,
environment-backed tools, memory, skills, compaction policy, durable events,
and subagents. The workspace does not own transport, storage, credentials, or
process isolation, which belong to the application that embeds it.

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
