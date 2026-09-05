# Interactive Pebble

Run `pebble` with terminal input and output. For scripts and pipes, use
`pebble exec`. Both commands use the same model catalog, provider credentials,
and `PEBBLE_<PROVIDER>_BASE_URL` overrides.

On first use, Pebble offers provider setup and saves your chosen model.
Use `/login` to save a masked API key, or set the provider's environment variable.
See [models and credentials](models-and-credentials.md) for configuration and
custom providers.

```sh
cargo run --locked -p pebble-cli -- --model gpt-5.6
pebble "explain this repository"
pebble --cwd ../service --name "Fix the build"
pebble --continue
pebble --resume <session-id>
```

Pebble uses the terminal's normal screen. Scroll, select, copy, and search with
your terminal's controls. Completed output stays above the prompt. Pickers
and current activity share a small area around the editor. Pebble does not
capture the mouse or clear terminal scrollback.

Markdown paragraphs, code blocks, and tables receive terminal styling. The
unfinished response stays near the editor until it can be printed. Very long
unfinished blocks use plain text. Code blocks do not yet have language-specific
syntax colors. Old output keeps its printed layout after a resize. `/tools`
can print a saved tool result again at the current width.

## Input

| Key | Action |
| --- | --- |
| Enter | Send a prompt, or queue steering while the agent works. |
| Alt+Enter | Queue a follow-up while the agent works. |
| Shift+Enter or Ctrl+J | Insert a newline. |
| Escape | Close a picker, or cancel active work. |
| Alt+Up | Return pending steering and follow-ups to the editor. |
| Up / Down | Recall inputs at the first or last editor line. |
| Ctrl+Left / Ctrl+Right | Move by words. |
| Ctrl+A / Ctrl+E | Move to the start or end of a line. |
| Ctrl+U / Ctrl+K | Delete to the start or end of a line. |
| Ctrl+W | Delete the previous word. |
| Ctrl+_ | Undo a draft change. |
| Tab | Insert the selected command, skill, path, or `@` file reference. |
| Ctrl+L | Open the model picker. |
| Ctrl+P / Ctrl+Shift+P | Cycle configured models forward / backward. |
| Shift+Tab | Cycle reasoning effort. |
| Ctrl+G | Open the draft in an external editor. |
| Ctrl+O | Toggle details for running tools. |
| Ctrl+T | Toggle visible reasoning. |
| Ctrl+Z | Suspend Pebble; return with `fg`. |
| Ctrl+C | Clear the draft. Press twice within one second to exit. |
| Ctrl+D | Exit when the draft is empty. |

Modified Enter keys depend on terminal support. Ctrl+J always inserts a
newline. Bracketed paste never submits its contents. New pastes and external
editor files are limited to 1 MiB.

Cancellation waits for the agent and tools to settle. Pending input returns
after the existing draft. A full input queue also returns the evicted input
to the editor. Image placeholders preserve their original content and order.

## Commands

| Command | Action |
| --- | --- |
| `/help` | Show commands and shortcuts. |
| `/new` | Start a new session with the current model and permissions. |
| `/resume [id]` | Choose or resume a saved session. |
| `/name <name>` | Rename the current session. |
| `/session` | Show the session id and storage directory. |
| `/model [model]` | Choose or set a configured model. `/model all` shows setup requirements for other models. |
| `/login [provider]` | Save an API key through a dedicated masked input. |
| `/logout [provider]` | Remove a saved credential or explicit credential source. |
| `/thinking [level]` | Choose reasoning effort. The model must support the selected level. |
| `/compact [instructions]` | Summarize model context and keep the saved transcript. |
| `/tools [tool-call-id]` | Choose a recent tool call or print a call from the journal. |
| `/agents [id]` | Choose a subagent and print its saved transcript. |
| `/skills` | List available skills. Invoke one with `/skill:<name> [input]`. |
| `/attach <image-path>` | Attach a PNG, JPEG, GIF, or WebP image up to 5 MiB. |
| `/copy` | Copy the last assistant answer through the system clipboard command. |
| `/export [path]` | Save Markdown, or the complete event log when the path ends in `.jsonl`. |
| `/editor` | Open the draft in the external editor. |
| `/settings` | Save the current model, reasoning effort, and display preferences. |
| `/suspend` | Suspend on Unix. |
| `/quit` | Save and exit. |

The tool picker lists the most recent 200 calls. A known tool-call id can
select an older call. Tool details include retained streamed output and the
final result. Bytes that the tool or environment never retained cannot be
recovered. Subagents require `--subagents` at startup.

Session changes, model changes, and compaction require idle work. Stop the
current turn with Escape first. A model change rebuilds the coding profile
through the library's resume path. A failed model selection restores the
previous agent. An unsent draft moves with you when changing sessions.

Typing `/` or `@` opens suggestions automatically. Typing continues to edit the
prompt. Up and Down select a suggestion; Tab inserts it; Enter submits the
prompt; Escape dismisses suggestions without changing the draft.

Other pickers have a separate search field with its own cursor. Search words
can appear in either order. Results rank close matches first and show the
selection position and result count. Arrow keys wrap; Page Up and Page Down
move by ten entries. The model picker marks and prioritizes the current model
and saved default. Ctrl+S selects a model and saves it as the startup default.
Model cycling uses the configured catalog order and keeps the unsent draft.

File completion after `@` searches tracked and unignored files in the current
Git repository. It inserts a reference for the model. It does not automatically
read the whole file into the prompt. Path completion works outside Git too.

## Storage and recovery

Sessions live in `~/.pebble/sessions/`. Set `PEBBLE_HOME` to change the Pebble
directory, or use `--sessions-dir <directory>` to change only session storage.
`--no-session` uses a temporary directory that is removed on normal exit. It
still reads the normal application configuration and cannot resume a session.

Each session contains an ordered `events.jsonl` journal, an atomic
`checkpoint.json`, an advisory `session.lock`, and any attachment files.
Checkpoints contain model history and the session's model, reasoning,
instructions, permission level, and approval setting. The application saves
them after each completed prompt or command and during orderly shutdown.
Only one Pebble process can write a session at a time on supported Unix systems.

On resume, Pebble prints the saved transcript and restores the last checkpoint.
If a crash left newer events, it reports that boundary. It does not rerun tools
or add the unfinished work to model history. An incomplete final journal line
is saved in an `events.partial-*` file before the journal is repaired. Complete
malformed records produce an error.
Pebble also refuses a journal that ends before its saved checkpoint.

Compaction changes model context and leaves the full journal in place. Session
forks and a conversation tree are not part of this version.

## Preferences

Preferences live in `~/.pebble/settings.json`, or `PEBBLE_HOME/settings.json`.
`--sessions-dir` does not move them. `/settings` saves common choices. Edit
this file to set an external editor or remap keys:

```json
{
  "model": "openai/gpt-5.6",
  "reasoning": "high",
  "show_reasoning": false,
  "expand_tools": false,
  "external_editor": "code --wait",
  "keybindings": {
    "ctrl+e": "external-editor",
    "alt+enter": "follow-up"
  }
}
```

Key names use modifiers in this order: `ctrl+`, `alt+`, `shift+`, then a
character or a name such as `enter`, `esc`, `up`, or `tab`. Supported actions
are `submit`, `follow-up`, `newline`, `cancel`, `quit`, `external-editor`,
`toggle-tools`, `toggle-reasoning`, `recover-input`, `complete`, `history-up`,
`history-down`, `undo`, `suspend`, `model-picker`, `next-model`,
`previous-model`, and `cycle-thinking`. Remapping applies to the main editor;
approval and question controls keep their displayed bindings.

Without an editor preference, Pebble uses `VISUAL`, then `EDITOR`, then `vi`.
The editor setting accepts a shell command with arguments. Pebble restores
terminal modes during the handoff. Set `NO_COLOR` to disable styling.
