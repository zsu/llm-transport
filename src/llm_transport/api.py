"""The Anthropic Messages API with a client-side tool loop.

Speaks `POST {LLM_API_URL}/v1/messages` through the `anthropic` SDK, so the
same code reaches api.anthropic.com directly or a LiteLLM gateway (and through
it Bedrock, Vertex and the rest), which serves that endpoint natively.

API mode does not load `.claude/rules/`, CLAUDE.md or skills the way the CLI
does from its working directory. A prompt that relies on them must inline what
it needs.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path
from typing import Any

from llm_transport.config import LLMConfig
from llm_transport.envelope import envelope, usage_dict
from llm_transport.pricing import cost_of
from llm_transport.tools import (
    Fetcher,
    ToolGrant,
    Workspace,
    default_fetch,
    search_tool_type,
    tool_defs,
)


def _make_client(cfg: LLMConfig, timeout_s: float) -> Any:
    import anthropic  # lazy: CLI-only installs do not need the SDK

    http_client = (anthropic.DefaultHttpxClient(verify=cfg.ca_bundle)
                   if cfg.ca_bundle else None)
    return anthropic.Anthropic(
        api_key=cfg.api_key, base_url=cfg.api_url, http_client=http_client,
        max_retries=cfg.request_retries,
        timeout=min(timeout_s, 600.0) if timeout_s > 0 else 600.0,
    )


def api_error(exc: Exception) -> tuple[int | None, str]:
    """(HTTP status, readable message) from an SDK exception.

    Duck-typed, so callers and tests need not import the SDK.
    """
    status = getattr(exc, "status_code", None)
    body = getattr(exc, "body", None)
    etype, message = "", ""
    if isinstance(body, dict):
        err = body.get("error") if isinstance(body.get("error"), dict) else body
        etype = str(err.get("type") or "")
        message = str(err.get("message") or "")
    message = message or str(getattr(exc, "message", "") or exc)
    name = type(exc).__name__
    if status is None:
        if "Timeout" in name:
            return None, f"request timed out: {message}"
        if "Connection" in name:
            return None, f"connection error: {message}"
        return None, f"{name}: {message}"
    return status, f"API error {status}{': ' + etype if etype else ''}: {message}"


def _get(obj: Any, key: str, default: Any = None) -> Any:
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def run_api(prompt: str, *, cwd: str | Path, allowed_tools: str = "",
            disallowed_tools: str = "", tools: str | None = None,
            env: dict | None = None, timeout_s: float = 0, max_usd: float = 0,
            fetch: Fetcher | None = None,
            blocked_hosts: Mapping[str, str] | None = None,
            read_roots: tuple[str | Path, ...] = (), system: str | None = None,
            on_log: Callable[[str], None] | None = None,
            config: LLMConfig | None = None, client: Any = None) -> dict:
    """Run one prompt through the Messages API, executing granted tools.

    `allowed_tools` / `disallowed_tools` / `tools` take the CLI's syntax and
    meaning (`tools=""` grants nothing). `timeout_s` bounds the whole call,
    `max_usd` its spend (0 = no bound). `fetch(url) -> (status, text)` backs
    WebFetch (default: `default_fetch` with `blocked_hosts`). File tools may
    read `cwd`, `$RUN_DIR` and `read_roots`. Never raises on an API or tool
    failure: the outcome is in the returned envelope.
    """
    cfg = config or LLMConfig.from_env()
    log = on_log or (lambda _m: None)
    started = time.monotonic()
    root = Path(cwd).resolve()
    run_env = {**os.environ, "PYTHONUTF8": "1", **(env or {})}
    grant = ToolGrant.parse(allowed_tools, disallowed_tools, tools)
    warnings = list(grant.notes)

    if client is None:
        if not cfg.api_key:
            return envelope(is_error=True, status=401, subtype="error_during_execution",
                            result="authentication_error: no API key - set LLM_API_KEY "
                                   "or ANTHROPIC_API_KEY", warnings=warnings)
        try:
            client = _make_client(cfg, timeout_s)
        except ImportError:
            return envelope(is_error=True, subtype="error_during_execution",
                            result="the anthropic package is not installed; "
                                   "pip install 'llm-transport[anthropic]', or unset "
                                   "LLM_USE_API", warnings=warnings)

    roots = [root] + [Path(r).resolve() for r in read_roots]
    if run_env.get("RUN_DIR"):
        roots.append(Path(run_env["RUN_DIR"]).resolve())
    work = Workspace(root, roots, grant, run_env,
                     fetch or partial(default_fetch, blocked_hosts=blocked_hosts))
    client_tools = tool_defs(grant)
    search = "WebSearch" in grant.names
    search_def = {"type": search_tool_type(cfg.model), "name": "web_search",
                  "max_uses": 10}

    messages: list = [{"role": "user", "content": [
        # Cached: each tool-loop turn re-sends this prefix, and it is usually
        # the bulk of every request.
        {"type": "text", "text": prompt, "cache_control": {"type": "ephemeral"}}]}]
    totals = usage_dict()
    cost = 0.0
    priced = True
    turns = 0
    last_text = ""

    def done(**kw: Any) -> dict:
        return envelope(turns=turns, usage=dict(totals), cost=cost,
                        warnings=warnings, **kw)

    while True:
        if timeout_s > 0 and time.monotonic() - started > timeout_s:
            return done(is_error=True, subtype="error_during_execution",
                        result=f"timed out after {timeout_s:g}s ({turns} turns)")
        if max_usd > 0 and cost >= max_usd:
            return done(is_error=True, subtype="error_max_budget_usd",
                        result=f"stage budget exceeded: ${cost:.2f} spent against a "
                               f"max-budget-usd of ${max_usd:g}")
        if turns >= cfg.max_turns:
            return done(is_error=True, subtype="error_max_turns",
                        result=f"reached LLM_MAX_TURNS ({cfg.max_turns}) without a "
                               f"final answer; last text: {last_text[:500]}")

        request: dict[str, Any] = {"model": cfg.model, "max_tokens": cfg.max_tokens,
                                   "messages": messages}
        if system:
            request["system"] = system
        if client_tools or search:
            request["tools"] = client_tools + ([search_def] if search else [])
        if cfg.effort:
            request["output_config"] = {"effort": cfg.effort}
        try:
            with client.messages.stream(**request) as stream:
                msg = stream.get_final_message()
                headers = getattr(getattr(stream, "response", None), "headers", None) or {}
        except Exception as exc:  # noqa: BLE001 - mapped to the envelope
            status, text = api_error(exc)
            if search and status == 400 and "web_search" in text.lower():
                # A gateway or provider without server-side search (Bedrock
                # through LiteLLM, for one). The call continues without it.
                search = False
                warnings.append(f"web search is not available at {cfg.host}; "
                                f"continuing without it ({text[:200]})")
                log(warnings[-1])
                continue
            return done(is_error=True, status=status,
                        subtype="error_during_execution", result=text)

        turns += 1
        u = _get(msg, "usage")
        turn_usage = usage_dict(
            int(_get(u, "input_tokens", 0) or 0), int(_get(u, "output_tokens", 0) or 0),
            int(_get(u, "cache_read_input_tokens", 0) or 0),
            int(_get(u, "cache_creation_input_tokens", 0) or 0))
        for k in totals:
            totals[k] += turn_usage[k]
        searches = int(_get(_get(u, "server_tool_use"), "web_search_requests", 0) or 0)
        reported = headers.get("x-litellm-response-cost") if hasattr(headers, "get") else None
        try:
            cost += float(reported)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            est = cost_of(cfg.model, turn_usage, searches)
            if est is not None:
                cost += est
            elif priced:
                priced = False
                warnings.append(f"no price known for model {cfg.model} and the "
                                f"endpoint reports none; cost is recorded as $0. "
                                f"Set LLM_PRICE_INPUT_PER_MTOK/LLM_PRICE_OUTPUT_PER_MTOK")
                log(warnings[-1])

        content = list(_get(msg, "content", []) or [])
        texts = [_get(b, "text", "") for b in content if _get(b, "type") == "text"]
        last_text = "".join(texts) or last_text
        stop = _get(msg, "stop_reason", "") or ""
        # Append the whole content, thinking blocks included and unchanged.
        messages.append({"role": "assistant", "content": content})

        if stop == "tool_use":
            results = []
            for block in content:
                if _get(block, "type") != "tool_use":
                    continue
                out, is_err = work.call(_get(block, "name"), _get(block, "input", {}))
                results.append({"type": "tool_result", "tool_use_id": _get(block, "id"),
                                "content": out, "is_error": is_err})
            messages.append({"role": "user", "content": results})
            continue
        if stop == "pause_turn":
            continue  # a long server-tool turn; re-send to let it finish
        if stop == "refusal":
            details = _get(msg, "stop_details")
            return done(is_error=True, subtype="error_during_execution",
                        stop_reason=stop,
                        result=f"the model declined the request (refusal, category "
                               f"{_get(details, 'category')})")
        return done(result="".join(texts), stop_reason=stop)
