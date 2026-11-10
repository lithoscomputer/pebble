# Third-party licenses

Pebble is licensed under the [MIT License](../../LICENSE). Some files contain text adapted from other open-source projects. Those parts stay under their original licenses, which are listed here.

Each adapted file has a header that names its source, the upstream commit, and the license. A header in a prompt template (`.md.j2`) is a template comment, so it is not sent to the model.

## OpenAI Codex

- Source: [openai/codex](https://github.com/openai/codex)
- License: Apache License 2.0, see [Apache-2.0.txt](Apache-2.0.txt)
- Notice: [openai-codex-NOTICE.txt](openai-codex-NOTICE.txt)

| Pebble file | Upstream file |
|---|---|
| `crates/pebble-coding-agent/src/profiles/prompts/openai.md.j2` | `codex-rs/core/prompt.md` and the `gpt_5_*_prompt.md` files at `87f7226cca` |
| `crates/pebble-coding-agent/src/profiles/prompts/gpt56.md.j2` | `codex-rs/models-manager/models.json` at `4c43465133` |
| `crates/pebble-coding-agent/src/profiles/prompts/gpt6.md.j2` | `codex-rs/models-manager/models.json` at `ed391d4dd2` |
| `crates/pebble-coding-agent/src/tools/apply_patch.rs` | the `apply_patch` tool |
| `crates/pebble-coding-agent/src/tools/apply_patch/apply_patch.lark` | the `apply_patch` grammar |

## Gemini CLI

- Source: [google-gemini/gemini-cli](https://github.com/google-gemini/gemini-cli)
- License: Apache License 2.0, see [Apache-2.0.txt](Apache-2.0.txt)
- Copyright 2026 Google LLC

| Pebble file | Upstream file |
|---|---|
| `crates/pebble-coding-agent/src/profiles/prompts/gemini.md.j2` | `packages/core/src/prompts/snippets.ts` at `87f5dd15d6` |

## Kimi Code

- Source: [MoonshotAI/kimi-code](https://github.com/MoonshotAI/kimi-code)
- License: MIT License, see [kimi-code-MIT.txt](kimi-code-MIT.txt)
- Copyright (c) 2026 Moonshot AI

| Pebble file | Upstream file |
|---|---|
| `crates/pebble-coding-agent/src/profiles/prompts/kimi.md.j2` | `packages/agent-core-v2/src/app/agentProfileCatalog/system.md` at `ce0e3ceb04` |
| `crates/pebble-coding-agent/src/profiles/kimi.rs` | `packages/agent-core-v2/src/app/edit/tools/edit.md` at `ceb158dc54` |
| `crates/pebble-coding-agent/src/profiles/kimi_tools.rs` | `packages/agent-core-v2/src/os/backends/node-local/tools/{read,write,bash}.md` at `ce0e3ceb04` and `ceb158dc54` |

## Pi

- Source: [earendil-works/pi](https://github.com/earendil-works/pi)
- License: MIT License, see [pi-MIT.txt](pi-MIT.txt)
- Copyright (c) 2025 Mario Zechner

| Pebble file | Upstream file |
|---|---|
| `crates/pebble-coding-agent/src/compaction.rs` | `packages/coding-agent/src/core/compaction/compaction.ts` at `d35935200f` |

The compaction prompt follows the section layout of Pi's prompt. Its wording is pebble's own. The notice is included as a precaution.

## Test snapshots

The prompt snapshots in `crates/pebble-coding-agent/src/profiles/snapshots/` are rendered copies of the prompts above. Each snapshot has the license of the prompt it renders.
