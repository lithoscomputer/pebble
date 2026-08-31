# Developing

TODO: Replace this introduction with project-specific development notes.

## Setup

Pebble is a library crate. It depends on `lithos-llm` through a path
dependency, so clone `lithos-llm` as a sibling directory of this repository:

```text
parent/
├── lithos-llm/
└── pebble/
```

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
| `mise run dev` | Check the library |
| `mise run fmt` | Format Rust code |
| `mise run fmt:check` | Check formatting without changing files |
| `mise run lint` | Run Clippy with warnings denied |
| `mise run test` | Run the test suite |
| `mise run check` | Run the routine verification gate |
| `mise run check:nightly` | Run the extended verification gate |

Run `mise run check` before opening a pull request.

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

The workflows check out only this repository. They cannot build until the
`lithos-llm` path dependency is available on the runner, because `lithos-llm`
is a separate private repository. Both workflows fail at manifest load until
the port adds a second checkout step (or the dependency changes form). Verify
changes locally with `mise run check` in the meantime.

## Cargo.lock policy

Every version in `Cargo.lock` matches the version in the `lithos-llm`
lockfile. Pebble builds `lithos-llm` from source, so the two projects must
agree. Pin new dependencies to the version that the `lithos-llm` or `fabro`
lockfile already contains, and use `cargo update --precise` to keep the
versions aligned.

## Releases

Pebble is a library. It has no release pipeline. Consumers depend on the
repository directly.
