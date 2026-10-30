# Command-line guide

Install the binary with the [README instructions](../README.md#use-it-from-the-command-line).

## Interactive sessions

Run `pebble` in a terminal for an interactive coding session:

```sh
pebble
pebble --model gpt-5.6 --cwd ../service
pebble --continue
pebble --resume <session-id>
```

The conversation stays in normal terminal scrollback. The prompt accepts
multiple lines, paste, history, and completion while the agent works. Enter
queues steering during a turn; Alt+Enter queues a follow-up. Escape cancels
active work and restores pending input. `/help` lists commands and shortcuts.
Sessions save locally and can resume after exit. See the
[interactive guide](interactive.md) for commands, settings, and recovery.

On first use, Pebble offers provider setup and model selection. `/login` saves
an API key through masked input, and `/model` lists configured models. From the
shell, use `pebble auth login <provider>` and `pebble auth status`. Interactive mode and
`exec` share `~/.pebble/settings.json`, `models.toml`, and `auth.json`, relocated
by `PEBBLE_HOME`. See [models and credentials](models-and-credentials.md)
for precedence, custom providers, and credential storage.

Interactive mode asks before each tool call outside the selected permission
level. Approval starts on **Deny**. `--no-approvals` hides those tools instead.
The permission level is an application policy, not an operating-system sandbox.

## Run one prompt

`pebble exec` runs one prompt to completion without a person in the loop: the
prompt goes in, the tools the model asks for run, and the final answer comes
out on standard output. Events are rendered to standard error as they happen,
and the exit status says how the prompt ended: `0` for an answer, `1` for a
failure, `130` when the prompt was interrupted or ran out of time.

```sh
pebble exec "add a --dry-run flag to bin/deploy"
pebble exec --model gpt-5.6 --cwd ../service --permission full "run the tests and fix what fails"
echo "summarize what this repository does" | pebble exec --quiet
pebble exec --json "..."   # one JSON event per line on standard error
pebble exec --verbose "..." # tool arguments and results, and the model's reasoning
```

The permission flag chooses what the agent may do without asking:
`read-only`, `read-write` (the default), or `full`, which enables commands.
In `pebble exec`, there is no approval path, so a tool the level does not allow is hidden from
the model and refused if called anyway. Credentials come from explicit sources,
the provider's usual environment variables, or saved API keys. Environment
variables override saved keys. `exec` uses `--model`, then the saved default,
then `claude-sonnet-5`, and never opens a setup prompt.
`PEBBLE_<PROVIDER>_BASE_URL`, such as `PEBBLE_OPENAI_BASE_URL` or
`PEBBLE_MOONSHOT_BASE_URL`, points a built-in provider at a compatible
endpoint instead: a proxy, a self-hosted model, or a test double. `--subagents`
lets the agent spawn children for independent work. `--tool-results` prints
each tool call's arguments in full and what the call answered;
`--transcript` prints the reasoning behind each turn when the model reports
it, and the text of a turn that did not stream; `--verbose` is both. The
command is the smallest application pebble ships, and its source is a worked
example of what an embedding application supplies.

## Embedding the command line

`pebble-cli-core` contains batch execution, the interactive session, terminal
rendering, approvals, credentials, and settings. The `pebble` binary parses
arguments and calls this library.

Use `session::run_prompt` to render events, print the final answer and summary,
and shut down the agent. `session::run_prompt_with` accepts rendering options
such as verbose output. `render::Renderer`, `RenderOptions`, and `Summary`
provide the terminal presentation. `approval::TerminalApproval` supplies a
terminal prompt for tool calls that need approval.

An embedding application builds its own agent, client, environment, tools,
and settings, then reuses this session and renderer. The `fabro exec` command
in the Fabro workflow application uses this path.

`TerminalApproval` keeps Pebble's three permission levels. Choosing "yes"
allows only the current call. Choosing "always" grants the least level that
allows the tool: `read-write` for native writes, `full` for shell, web, and
unrecognized tools, including Model Context Protocol (MCP) tools. The prompt states that scope before asking.
The grant applies to every tool allowed by that level for the lifetime of the
approval-service instance, including sessions that share it; a fresh instance
starts at its configured level. Noninteractive approval refuses calls outside
the configured level. Pebble's own `exec` uses a fixed permission level, and
its interactive UI uses separate per-call approvals.
