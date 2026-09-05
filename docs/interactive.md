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
unfinished blocks use plain text. Code blocks use basic lexical colors for Rust, JavaScript/TypeScript, Python,
shell, JSON, TOML, YAML, Go, C/C++, and Java. Unknown languages stay plain.
Edit previews show removed and added lines; failed edits show the error.
Tool headings show the file or command, with ten retained output lines by default.
`/tools` prints the retained result and any saved output chunks with the same styling.
`NO_COLOR` disables syntax and diff colors. Old output keeps its printed layout after a resize. `/tools`
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
| Ctrl+V | Paste an image from the clipboard. |
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
| `/fork [input-event or @bookmark]` | Branch before an earlier prompt and restore it for editing. |
| `/clone` | Copy the current model context into a new session. |
| `/tree` | Navigate saved sessions, their ancestry, and current conversation boundaries. |
| `/bookmark [name]` | Name the current boundary, or choose a saved bookmark. |
| `/name <name>` | Rename the current session. |
| `/session` | Show the session id and storage directory. |
| `/favorites [model or clear]` | Choose a saved shortlist for Ctrl+P. A model argument toggles it; `clear` restores cycling through all configured models. |
| `/model [model]` | Choose or set a configured model. `/model all` shows setup requirements for other models. |
| `/login [provider]` | Save an API key through a dedicated masked input. |
| `/logout [provider]` | Remove a saved credential or explicit credential source. |
| `/thinking [level]` | Choose reasoning effort. The model must support the selected level. |
| `/compact [instructions]` | Summarize model context and keep the saved transcript. |
| `/tools [tool-call-id]` | Choose a recent tool call or print a call from the journal. |
| `/agents [id]` | Choose a subagent and print its saved transcript. |
| `/skills` | List available skills. Invoke one with `/skill:<name> [input]`. |
| `/attach <image-path>` | Attach a PNG, JPEG, GIF, or WebP image up to 5 MiB. |
| `/paste` | Paste a clipboard image. |
| `/shells [clear]` | Show saved shell results, or drop pending shell context. |
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
Model cycling keeps the unsent draft. `/favorites` opens a searchable picker;
Enter toggles a model and saves immediately. The picker keeps its search and
selection so you can choose several models. The model picker also links to it.

Ctrl+P and Ctrl+Shift+P cycle favorites in the order you added them. Unavailable
favorites are skipped. When the current model is outside the shortlist, forward
cycling starts at the first favorite and backward cycling at the last. An empty
shortlist uses all configured models in catalog order. `/model` continues to show
all configured models; `/model all` includes setup requirements for the rest.
Favorites are saved globally as `favorite_models` in `settings.json`.

File completion after `@` searches tracked and unignored files in the current
Git repository. It inserts a reference for the model. It does not automatically
read the whole file into the prompt. Path completion works outside Git too.

## Images

Copy an image, then press Ctrl+V or enter `/paste`. macOS uses the system
clipboard through AppleScript. Linux uses `wl-paste` or `xclip`. You can set
`PEBBLE_CLIPBOARD_COMMAND` to a command that writes image bytes to stdout.
For example, this supports a custom clipboard bridge over SSH. Ordinary text
paste still uses your terminal's paste shortcut.

`/attach <path>` reads an image file. Pebble checks its format header, size,
and dimensions. Images must be no larger than 5 MiB and 64 million pixels.
Image loading runs while you edit. Escape cancels it. The placeholder enters
the draft only after loading succeeds. Delete the placeholder to omit the image.

Previews use the [Kitty graphics protocol](https://sw.kovidgoyal.net/kitty/graphics-protocol/)
in Kitty and Ghostty, and the [iTerm2 image protocol](https://iterm2.com/documentation-images.html)
in iTerm2 and WezTerm. Other terminals show dimensions and a text label.
Previews are disabled by default inside tmux and screen. Set
`PEBBLE_IMAGE_PROTOCOL=kitty`, `iterm2`, or `none` to override detection.
Graphics stay in normal scrollback above the editor. The terminal controls how
long it retains them. Very small terminal windows use text labels.

PNG previews need no converter. For other formats, macOS uses its `sips` tool;
Linux can use an installed ImageMagick `magick` command. A conversion failure
keeps the original attachment usable. iTerm2 can display the original format;
Kitty needs a PNG preview. Preview conversion may resize to 1024 pixels; the
original bytes sent to the model stay unchanged.

Submitted images and cached previews survive resume and branching. Markdown
exports embed the original image as a data URL; JSONL exports retain the original
content parts. Remote image URLs are not fetched for previews. Clipboard and
conversion helpers have a three-second timeout per invocation and bounded output.
Tests use fixture clipboard commands and never read your clipboard.

## Shell commands

Enter `!command` to run a command with `/bin/sh` in the session directory.
The command and its result join the next model prompt. `!!command` runs without
adding model context. Neither form sends a model request by itself. Commands
require an idle agent. A draft stays editable while a command runs.

Shell commands ask for approval unless permission is `full`. With
`--no-approvals`, lower permission levels reject commands. Escape cancels the
command and its process group. Exit status and cancellation appear below the
output. Child processes cannot outlive the command.

Pebble retains up to 64 KiB of combined output per command. `/shells` shows
saved results, including after a restart. `/shells clear` drops pending context
without deleting those results. Pending context is limited to 512 KiB.
Markdown exports include shell results; `.jsonl` exports contain agent events
only. Completed results are saved; a hard crash can lose an active command's
partial output. Commands use the current process environment, and shell state
such as `cd` does not carry over to the next command.

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

Compaction changes model context and leaves the full journal in place. New
checkpoints are also retained in `checkpoints/`. This uses more disk space than
keeping only the latest context, but permits branching across compaction.
Older sessions remain readable; prompts from before checkpoint retention cannot
be selected as fork points.

`/fork` offers earlier prompts. Selecting one creates a new session with the
context from before that prompt and restores the prompt in the editor. Nothing
runs until you submit it. `/clone` copies the current context. Both preserve the
original session and copy attachments, applicable shell results, and bookmarks.
An existing draft moves with you. A fork appends the restored prompt to that draft.

Use `/bookmark working` to name the current context. `/fork @working` creates
a branch there. Reusing a bookmark name moves the bookmark. `/tree` shows session
ancestry and saved boundaries; selecting a history boundary creates a branch.
Selecting a session resumes that session. Branches record their parent session
and event number in the checkpoint metadata.

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
`previous-model`, `cycle-thinking`, and `paste-image`. Remapping applies to the main editor;
approval and question controls keep their displayed bindings.

Without an editor preference, Pebble uses `VISUAL`, then `EDITOR`, then `vi`.
The editor setting accepts a shell command with arguments. Pebble restores
terminal modes during the handoff. Set `NO_COLOR` to disable styling.
