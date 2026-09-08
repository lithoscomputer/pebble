# Developing

Pebble is a workspace of two library crates, one example, and one binary.
`crates/pebble-agent` is the provider-neutral agent loop.
`crates/pebble-coding-agent` is the coding-agent layer and owns the example.
`crates/pebble-cli` is the `pebble` command, the smallest application that
embeds the coding agent. Most coding-agent work is a change to
`crates/pebble-coding-agent/src/`, its unit tests beside it, and the contract
tests in `crates/pebble-coding-agent/tests/`. Generic turn-loop work belongs
in `crates/pebble-agent/src/`.

The CLI's private `interactive` modules own terminal input and output, prompt
editing, session files, questions, and approvals. The normal-screen renderer
uses Crossterm and Termimad. Terminal and storage concerns stay in the CLI.
Run it with `cargo run --locked -p pebble-cli`.

The private `application`, `settings`, `credentials`, and `storage` modules
provide common configuration for the TUI, `exec`, and auth commands. Login
input uses a separate `secret_input` buffer with no history. The CLI layers
optional `models.toml` over the upstream catalog and supplies an application-owned
`CredentialProvider`. These concerns stay outside the agent libraries.

## Setup

Pebble is a library crate. It depends on `lithos-llm` as a git dependency
pinned to one commit (`rev`) in the workspace `Cargo.toml`. The repository is
public, so Cargo fetches it over HTTPS with no credentials.

To move the pin: push the lithos-llm commit (a branch under review is fine),
change `rev` in `Cargo.toml` to the full sha, run `cargo update lithos-llm`,
and commit `Cargo.toml` and `Cargo.lock` together. When the lithos-llm change
merges, re-pin to the merge commit the same way. Do not use a `[patch]`
section; the pinned sha is the single source of truth.

Install [Mise](https://mise.jdx.dev/), then install the locked tools and prepare
the pinned Rust Style Guide:

```sh
mise trust
mise install --locked --jobs=1
mise run setup
```

## Common tasks

| Command | Purpose |
| --- | --- |
| `mise run dev` | Run the coding-agent example against a live provider |
| `mise run exec -- <PROMPT>` | Run `pebble exec` against a live provider |
| `mise run fmt` | Format Rust code |
| `mise run fmt:check` | Check formatting without changing files |
| `mise run lint` | Run Clippy with warnings denied |
| `mise run test` | Run the test suite |
| `mise run test:doc` | Run the documentation tests |
| `mise run doc` | Build the API documentation with warnings denied |
| `mise run check` | Run the routine verification gate |
| `mise run check:nightly` | Run the extended verification gate |

Run `mise run check` before opening a pull request.

## Running the example

`crates/pebble-coding-agent/examples/coding_agent.rs` runs a real coding agent.
It builds a client, works in a new directory under the system temporary
directory, steers one prompt, interrupts another, and prints what the agent
used. It is the only thing in the repository that calls a provider.

```sh
# The default model, claude-sonnet-5.
mise run dev

# Any selector the built-in catalog knows.
cargo run --locked -p pebble-coding-agent --example coding_agent -- gpt-5.6
```

It needs a key for whichever provider the model resolves to, in the variable
lithos-llm reads for that provider: `ANTHROPIC_API_KEY` for the default model,
`OPENAI_API_KEY`, `GEMINI_API_KEY`, and so on. Credentials are resolved per
call, so a missing key fails the first model call rather than the build.

The example prints the directory it worked in and leaves it behind, so the
files the model wrote can be read afterwards. Nothing removes them; they are
under the system temporary directory.

## Running the tests

```sh
# Every test, through Nextest.
mise run test

# The documentation tests, which Nextest does not run.
mise run test:doc

# Just the end-to-end suite.
cargo nextest run --locked -p pebble-coding-agent --all-features --test e2e
```

Unit tests live beside the code they cover. `tests/` holds the contract tests:
the serialized event stream and record format (`event_contract.rs`,
`record_contract.rs`), message conversion, tool dispatch, the event pipeline,
and `e2e.rs` — the example's own flow, driven through the scripted provider
and a real temporary directory, so it needs no credentials and no network.

Tests that reach `pebble_coding_agent::test_support` need the `test-util` feature, which is
why the test tasks pass `--all-features`.

The `pebble` binary is tested with `trycmd`: each case under
`crates/pebble-cli/tests/cmd/` names the arguments, the expected output on
each stream, and the exit status. A case that runs a prompt talks to
[twin-openai](https://github.com/lithoscomputer/twins), a deterministic
OpenAI-compatible server the harness starts in the test process on an
ephemeral port and hands to the binary through `PEBBLE_<PROVIDER>_BASE_URL`.
The real codec and transport run; only the far end is scripted. Each case's
API key, `OPENAI_API_KEY`, `MOONSHOT_API_KEY`, or `VENICE_API_KEY`, names its
scenario namespace
in `tests/cmd/scenarios.json`, and its working directory is a sandbox copy of
its `.in/` directory. A subagent's id is random, so a scripted parent waits
with no `agent_id`, which waits for every child, and fixtures elide the id
with `[..]`. Run
`TRYCMD=overwrite mise run test` to accept changed output after reviewing it,
and `TRYCMD=dump` to write actual output to a `dump/` directory instead.

`crates/pebble-cli/tests/tui.rs` drives the actual interactive binary through a
pseudoterminal on Unix. A small VTE-based terminal emulator checks normal
scrollback, cursor placement, bracketed paste, resize, and terminal-mode
restoration. The same deterministic provider covers saved sessions,
cancellation, and approvals. Run it with
`cargo nextest run --locked -p pebble-cli --test tui`.

`tests/configuration.rs` checks saved credentials, environment precedence,
custom catalog entries, and concurrent auth commands through the real binary.
CLI and terminal tests set an isolated `PEBBLE_HOME`. Terminal tests also check
masked login, credential setup on resume, and that keys never appear in terminal
output, the session journal, or exports.

## Rust policy

This project follows the pinned Brynary Rust Style Guide. Run
`mise run setup`, then read `.ai/style-guides/rust-style-guide/SKILL.md` before
changing Rust code, configuration, project structure, or tests.

The project uses Rust 2024 and declares Rust 1.88 as its minimum supported
version. Mise pins the development compiler and the nightly formatter.

## Continuous integration

Routine checks run for pull requests and pushes to `main`. Extended checks run
each night. Both workflows test these native platforms:

- macOS arm64;
- Linux x86_64;
- Linux arm64.

The workflows check out only this repository. `lithos-llm` is public, so the
dependency fetch needs no credentials.

## Cargo.lock policy

Every version in `Cargo.lock` matches the version in the `lithos-llm`
lockfile. Pebble builds `lithos-llm` from source, so the two projects must
agree. Pin new dependencies to the version that the `lithos-llm` or `fabro`
lockfile already contains, and use `cargo update --precise` to keep the
versions aligned.

The rule covers every entry in the lockfile, not only the packages this
project builds today. Cargo records the optional dependencies of a dependency
even when no feature enables them, and enabling that feature later would build
whatever version the lockfile named. A dependency that drags in entries no
`lithos-llm` version covers is the wrong dependency.

Test tooling follows the same rule. `trycmd` and `twin-openai` are
dev-dependencies of `lithos-llm` too, so pebble's lockfile takes their
entries from there. When one of them needs to move, move it in `lithos-llm`
first, then re-pin here. The `twin-openai` revision may run ahead of
`lithos-llm`'s while a twin feature pebble needs is still landing; the crate
version stays the same and the two revisions meet at the next re-pin.

## Releases

Both crates are libraries. Neither has a release pipeline. Consumers depend on
the repository directly.
