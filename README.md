# Pebble

TODO: Replace this text with the project description.

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
mise run dev
mise run test
mise run check
```

See [DEVELOPING.md](DEVELOPING.md) for the complete development workflow.

Routine and nightly checks run on macOS arm64, Linux x86_64, and Linux arm64.
GitHub cannot run them yet, because the workflows do not check out the
`lithos-llm` path dependency. See [DEVELOPING.md](DEVELOPING.md).
