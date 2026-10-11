"""The `claude` CLI in print mode (`claude -p --output-format json`)."""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

from llm_transport.envelope import envelope


def spawn_kwargs() -> dict:
    """Platform flags that put a child in its own killable process group."""
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}  # type: ignore[attr-defined]
    return {"start_new_session": True}


def kill_process_tree(proc: subprocess.Popen) -> None:
    """Kill `proc` and everything it spawned.

    `claude` is a Node program that spawns children; killing only the parent
    leaves them running - and billing.
    """
    if proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True, check=False, timeout=30)
        else:
            os.killpg(os.getpgid(proc.pid), 9)  # type: ignore[attr-defined]
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        proc.kill()
    except OSError:
        pass


def cli_env(base: dict | None = None) -> dict:
    """The environment for a `claude` child process.

    LLM_API_KEY / LLM_API_URL are the one documented pair for both transports;
    the CLI only reads ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL, so they are
    passed on under those names - but never over a value already set.
    """
    env = {**os.environ, "PYTHONUTF8": "1", **(base or {})}
    for ours, theirs in (("LLM_API_KEY", "ANTHROPIC_API_KEY"),
                         ("LLM_API_URL", "ANTHROPIC_BASE_URL")):
        if env.get(ours, "").strip() and not env.get(theirs, "").strip():
            env[theirs] = env[ours].strip()
    return env


def resolve_cli() -> list[str]:
    """CLAUDE_BIN as an argv prefix: a bare name resolved on PATH (.cmd shims
    on Windows need shutil.which), or a quoted multi-token command."""
    cand = os.environ.get("CLAUDE_BIN", "claude")
    parts = shlex.split(cand, posix=(os.name != "nt"))
    if os.name == "nt":
        parts = [p.strip('"') for p in parts]
    if len(parts) == 1:
        return [shutil.which(parts[0]) or parts[0]]
    return parts


@lru_cache(maxsize=8)
def cli_flags(cmd: tuple[str, ...]) -> frozenset[str]:
    """Long options this CLI build advertises in --help; empty if unreadable."""
    try:
        out = subprocess.run([*cmd, "--help"], capture_output=True, text=True,
                             encoding="utf-8", errors="replace",
                             stdin=subprocess.DEVNULL, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return frozenset()
    return frozenset(re.findall(r"--[a-z0-9][a-z0-9-]+",
                                (out.stdout or "") + (out.stderr or "")))


def filter_flags(cmd: list[str], flags: list[str] | tuple[str, ...]
                 ) -> tuple[list[str], list[str]]:
    """Drop `--flag [value]` groups this build does not advertise.

    An unreadable --help keeps everything: refusing every flag that could not
    be confirmed would silently drop them on any machine whose probe failed.
    """
    known = cli_flags(tuple(cmd))
    kept: list[str] = []
    dropped: list[str] = []
    skipping = False
    for item in flags:
        if item.startswith("--"):
            skipping = bool(known) and item not in known
            (dropped if skipping else kept).append(item)
        elif not skipping:
            kept.append(item)
    return kept, dropped


class CLITimeout(Exception):  # noqa: N818 - mirrors subprocess.TimeoutExpired
    def __init__(self, timeout_s: float, stdout: str, stderr: str):
        super().__init__(f"the claude CLI timed out after {timeout_s:g}s")
        self.stdout, self.stderr = stdout, stderr


def run_cli_raw(prompt: str, *, cwd: str | Path, allowed_tools: str = "",
                disallowed_tools: str = "", optional_flags: tuple[str, ...] = (),
                timeout_s: float = 0, env: dict | None = None
                ) -> tuple[int, str, str, list[str]]:
    """One `claude -p --output-format json` run.

    Returns (returncode, stdout, stderr, dropped optional flags). Raises
    CLITimeout after killing the process tree, and OSError when the CLI cannot
    be spawned. `optional_flags` are filtered against --help first.
    """
    cli = resolve_cli()
    optional, dropped = filter_flags(cli, optional_flags)
    cmd = [*cli, "-p", "--output-format", "json"]
    if allowed_tools:
        cmd += ["--allowed-tools", allowed_tools]
    if disallowed_tools:
        cmd += ["--disallowed-tools", disallowed_tools]
    cmd += optional
    proc = subprocess.Popen(
        cmd, cwd=str(cwd), env=cli_env(env),
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace", **spawn_kwargs(),
    )
    try:
        out, err = proc.communicate(input=prompt,
                                    timeout=timeout_s if timeout_s > 0 else None)
    except subprocess.TimeoutExpired:
        kill_process_tree(proc)
        out, err = proc.communicate()
        raise CLITimeout(timeout_s, out or "", err or "") from None
    return proc.returncode, out or "", err or "", dropped


def run_cli(prompt: str, *, cwd: str | Path, allowed_tools: str = "",
            disallowed_tools: str = "", optional_flags: tuple[str, ...] = (),
            timeout_s: float = 0, env: dict | None = None) -> dict:
    """`run_cli_raw`, normalised to an envelope. Never raises."""
    try:
        rc, out, err, dropped = run_cli_raw(
            prompt, cwd=cwd, allowed_tools=allowed_tools,
            disallowed_tools=disallowed_tools, optional_flags=optional_flags,
            timeout_s=timeout_s, env=env)
    except CLITimeout as exc:
        return envelope(is_error=True, subtype="error_during_execution",
                        result=str(exc))
    except OSError as exc:
        return envelope(is_error=True, subtype="error_during_execution",
                        result=f"the claude CLI could not be started: {exc}. "
                               "Install it, or set CLAUDE_BIN to its full path.")
    warnings = [f"this claude build does not support {f}" for f in dropped]
    try:
        parsed = json.loads(out)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        parsed.setdefault("warnings", [])
        parsed["warnings"] += warnings
        parsed["dropped_flags"] = dropped
        if rc != 0:
            parsed["is_error"] = True
        return parsed
    detail = (out.strip() or err.strip())[-2000:]
    return envelope(is_error=True, subtype="error_during_execution",
                    result=detail or f"the claude CLI exited with status {rc}",
                    warnings=warnings)
