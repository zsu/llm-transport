# Changelog

## 0.1.0 - 2026-10-11

First release, extracted from two internal pipelines that shared it as a
vendored file.

- `run()` dispatches on `LLM_USE_API`: the `claude` CLI in print mode (default)
  or the Anthropic Messages API.
- API mode runs a client-side tool loop for grants in the CLI's own syntax:
  Read, Grep, Glob, `Bash(<prefix>*)`, WebFetch, and server-side WebSearch.
- Both transports return the CLI's `--output-format json` envelope.
- `LLM_API_KEY` / `LLM_API_URL` configure both transports; the `ANTHROPIC_*`
  names are accepted as fallbacks.
- Cost from LiteLLM's `x-litellm-response-cost` header, else list prices.
