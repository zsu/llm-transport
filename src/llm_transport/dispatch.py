"""`run`: one call through whichever transport LLM_USE_API selects."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llm_transport.api import run_api
from llm_transport.cli import run_cli
from llm_transport.config import use_api


def run(prompt: str, *, cwd: str | Path, allowed_tools: str = "",
        disallowed_tools: str = "", tools: str | None = None,
        timeout_s: float = 0, max_usd: float = 0, env: dict | None = None,
        cli_flags_optional: tuple[str, ...] = (), **api_kwargs: Any) -> dict:
    """Run `prompt`; return the CLI-shaped envelope. Never raises on a model
    or transport failure.

    In CLI mode `max_usd` becomes `--max-budget-usd` and `tools` becomes
    `--tools`, each dropped (and listed in `dropped_flags`) when the installed
    build does not advertise it. `api_kwargs` (fetch, blocked_hosts,
    read_roots, system, on_log, ...) apply in API mode only.
    """
    if use_api():
        return run_api(prompt, cwd=cwd, allowed_tools=allowed_tools,
                       disallowed_tools=disallowed_tools, tools=tools,
                       timeout_s=timeout_s, max_usd=max_usd, env=env, **api_kwargs)
    optional = list(cli_flags_optional)
    if tools is not None:
        optional += ["--tools", tools]
    if max_usd > 0:
        optional += ["--max-budget-usd", f"{max_usd:g}"]
    return run_cli(prompt, cwd=cwd, allowed_tools=allowed_tools,
                   disallowed_tools=disallowed_tools, optional_flags=tuple(optional),
                   timeout_s=timeout_s, env=env)
