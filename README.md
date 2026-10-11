# llm-transport

Run a prompt through the **`claude` CLI** in print mode, or straight through the
**Anthropic Messages API** - directly, or via a **LiteLLM gateway** in front of
Bedrock, Vertex and the rest - with one environment switch and one result shape.

```python
import llm_transport

env = llm_transport.run(
    "Read notes.md and reply with a one-line summary.",
    cwd=".",
    allowed_tools="Read,Grep,Glob",
)
if env["is_error"]:
    raise RuntimeError(env["result"])
print(env["result"], env["total_cost_usd"], env["num_turns"])
```

`env` is the JSON envelope `claude -p --output-format json` prints (`result`,
`is_error`, `subtype`, `stop_reason`, `api_error_status`, `num_turns`, `usage`,
`total_cost_usd`) plus a `warnings` list - in **both** modes. Code that already
parses the CLI's output can switch transport without touching its error handling.

## Install

```bash
pip install llm-transport              # CLI mode only: no dependencies
pip install "llm-transport[anthropic]" # adds the anthropic SDK for API mode
```

Python 3.12+.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `LLM_USE_API` | unset → **CLI** | `1` / `true` / `yes` / `on` → the Messages API. Anything else, blank or unset → `claude -p`. |
| `LLM_API_URL` | `ANTHROPIC_BASE_URL`, else `https://api.anthropic.com` | Base URL. A LiteLLM gateway URL works as is: LiteLLM serves `/v1/messages` natively. |
| `LLM_API_KEY` | `ANTHROPIC_API_KEY` | Sent as `x-api-key`. |
| `LLM_MODEL` | `ANTHROPIC_MODEL`, else `claude-opus-5-5` | Model id, or the gateway's alias for it. |
| `LLM_EFFORT` | `high` | `output_config.effort`. Set it **blank** to omit the field for a gateway that rejects it. |
| `LLM_MAX_TOKENS` | `32000` | Per response (requests stream, so large values are safe). |
| `LLM_MAX_TURNS` | `40` | Tool-loop turns per call. |
| `LLM_REQUEST_RETRIES` | `4` | SDK retries of one request on 408/409/429/5xx. |
| `LLM_CA_BUNDLE` | `SSL_CERT_FILE`, else `REQUESTS_CA_BUNDLE` | TLS trust for a gateway with a private CA. |
| `LLM_PRICE_INPUT_PER_MTOK`, `LLM_PRICE_OUTPUT_PER_MTOK` | - | USD per million tokens, for a model the price table does not know. |
| `LLM_WEB_SEARCH_TOOL` | by model | Override the server web-search tool type. |
| `CLAUDE_BIN` | `claude` | CLI mode: the executable, or a quoted multi-token command. |

`LLM_API_KEY` and `LLM_API_URL` drive **both** transports: in CLI mode they are
passed to `claude` as `ANTHROPIC_API_KEY` / `ANTHROPIC_BASE_URL`, unless those are
already set.

## Why the Anthropic Messages format, not OpenAI's

Behind LiteLLM you could speak the OpenAI chat format instead. This package
doesn't, because for Claude that format drops what matters most in a tool loop:
prompt caching (`cache_control`), which keeps a turn from re-paying for the whole
prompt; thinking blocks, which must go back unchanged between tool turns;
server-side web search; `effort`; refusal details; and cache token accounting.

## Tools in API mode

Grants use the CLI's own `--allowed-tools` / `--disallowed-tools` / `--tools`
syntax and meaning, so one grant string works in both modes:

```python
llm_transport.run(prompt, cwd=repo,
                  allowed_tools="Read,Grep,Bash(python scripts/lookup.py*),WebFetch,WebSearch",
                  disallowed_tools="Write,Edit",
                  tools=None)   # None: derive from the grant; "": no tools at all
```

| Tool | API-mode behaviour |
|---|---|
| `Read`, `Grep`, `Glob` | Same input fields as Claude Code. Confined to `cwd`, `$RUN_DIR` (from `env`) and any `read_roots`. |
| `Bash(<prefix>*)` | Runs only commands that start with the prefix, **without a shell**: pipes, redirects, `;`, `&&`, backticks and `$(...)` are refused. `python` maps to the current interpreter. A bare `Bash` grant is **not** honoured. |
| `WebFetch` | GET through `fetch(url) -> (status, text)` - default `default_fetch`, which converts HTML to text and refuses any `blocked_hosts` you pass. |
| `WebSearch` | Anthropic's server-side search tool. Where the endpoint rejects it (e.g. Bedrock behind LiteLLM) the call continues without it and says so in `warnings`. |

Deny beats allow. Tools the CLI has but API mode does not (`Write`, `Edit`, ...)
are left out with a note in `warnings`. **In API mode this package is the
permission boundary**; in CLI mode the CLI is.

API mode does **not** load `.claude/rules/`, `CLAUDE.md` or skills the way the
CLI does from its working directory. Inline what the prompt needs.

## Limits and cost

```python
llm_transport.run(prompt, cwd=".", timeout_s=900, max_usd=2.0)
```

- `timeout_s` bounds the call (CLI: the process tree is killed; API: checked
  between turns and as the request timeout).
- `max_usd` is `--max-budget-usd` in CLI mode (dropped, and listed in
  `dropped_flags`, if the installed build lacks it) and is checked between turns
  in API mode, ending with `subtype: "error_max_budget_usd"`.
- `total_cost_usd` in API mode is LiteLLM's `x-litellm-response-cost` when the
  gateway sends it, otherwise an estimate at first-party list prices
  (`llm_transport.PRICES`). Bedrock and Vertex bill at their own rates. A model
  the table cannot price is recorded at $0 with a warning.

## Errors

`run`, `run_api` and `run_cli` never raise on a model or transport failure: the
envelope has `is_error: true`, `api_error_status` (HTTP status, when there was
one) and a readable `result` such as `API error 401: authentication_error: ...`.
Classify on those. Lower-level pieces (`run_cli_raw`, `spawn_kwargs`,
`kill_process_tree`, `cli_env`) are public for callers that drive the CLI
themselves.

## License

MIT
