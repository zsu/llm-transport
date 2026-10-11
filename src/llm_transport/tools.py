"""Client-side tools for API mode, granted with the CLI's allow-string syntax.

In API mode this module is the permission boundary, so it is deliberately
strict: file tools are confined to the working directory (plus the run
directory and any extra roots the caller names), Bash runs only an
allowlisted command prefix and never through a shell, and WebFetch goes
through a fetcher the caller can replace.
"""

from __future__ import annotations

import fnmatch
import html
import os
import re
import shlex
import subprocess
import sys
import urllib.request
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from llm_transport.cli import kill_process_tree, spawn_kwargs

SUPPORTED_TOOLS = ("Read", "Grep", "Glob", "Bash", "WebFetch", "WebSearch")

Fetcher = Callable[[str], tuple[object, str]]

# Shell operators. Commands run without a shell, so these would reach the
# program as literal arguments; refusing them says so instead of letting the
# model believe a pipe or redirect happened.
_SHELL_OPS = re.compile(r"[;&|<>`\r\n]|\$\(")

_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".pytest_cache"}
MAX_TOOL_CHARS = 100_000


class ToolError(Exception):
    """A tool call that cannot be honoured; reported back to the model."""


@dataclass
class ToolGrant:
    """What a call may use, parsed from the CLI's allow/deny strings."""
    names: list[str]
    bash_prefixes: list[str]
    notes: list[str] = field(default_factory=list)

    @classmethod
    def parse(cls, allowed: str, disallowed: str = "",
              tools: str | None = None) -> ToolGrant:
        """`allowed` / `disallowed` as in --allowed-tools / --disallowed-tools;
        `tools` as in --tools (None derives it from `allowed`, "" grants none).
        """
        names: list[str] = []
        prefixes: list[str] = []
        notes: list[str] = []
        for spec in (s.strip() for s in (allowed or "").split(",")):
            if not spec:
                continue
            base = spec.split("(", 1)[0].strip()
            if base == "Bash":
                inner = spec[len(base):].strip()
                if not (inner.startswith("(") and inner.endswith(")")):
                    # A bare grant means "any command" to the CLI. Here there
                    # is no classifier behind it, so it is not honoured.
                    notes.append("a bare Bash grant is not honoured in API mode; "
                                 "grant Bash(<command prefix>*) instead")
                    continue
                prefix = re.sub(r":?\*$", "", inner[1:-1].strip()).strip()
                if prefix:
                    prefixes.append(prefix)
            if base not in SUPPORTED_TOOLS:
                notes.append(f"{base} is not available in API mode")
                continue
            if base not in names:
                names.append(base)
        # Deny beats allow, as it does in the CLI.
        denied = {s.split("(", 1)[0].strip() for s in (disallowed or "").split(",")}
        names = [n for n in names if n not in denied]
        if tools is not None:
            keep = {t.strip() for t in tools.split(",") if t.strip()}
            names = [n for n in names if n in keep]
        if "Bash" not in names:
            prefixes = []
        return cls(names, prefixes, notes)


def tool_defs(grant: ToolGrant) -> list[dict]:
    """Client tool definitions for `grant`. WebSearch is a server tool and is
    added by the caller."""
    defs = {
        "Read": {
            "name": "Read",
            "description": "Read a text file. Returns lines numbered like `cat -n`. "
                           "Use offset/limit to page through a long file.",
            "input_schema": {"type": "object", "properties": {
                "file_path": {"type": "string",
                              "description": "Absolute, or relative to the working directory"},
                "offset": {"type": "integer", "description": "1-based line to start at"},
                "limit": {"type": "integer", "description": "Lines to read (default 2000)"},
            }, "required": ["file_path"]},
        },
        "Grep": {
            "name": "Grep",
            "description": "Search file contents with a Python regular expression.",
            "input_schema": {"type": "object", "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string",
                         "description": "File or directory (default: working directory)"},
                "glob": {"type": "string", "description": "Filter files, e.g. *.json"},
                "output_mode": {"type": "string",
                                "enum": ["files_with_matches", "content", "count"]},
                "-i": {"type": "boolean", "description": "Case insensitive"},
                "-n": {"type": "boolean",
                       "description": "Line numbers (content mode, default true)"},
                "-A": {"type": "integer"}, "-B": {"type": "integer"}, "-C": {"type": "integer"},
                "head_limit": {"type": "integer", "description": "Max output lines (default 250)"},
                "multiline": {"type": "boolean"},
            }, "required": ["pattern"]},
        },
        "Glob": {
            "name": "Glob",
            "description": "List files matching a glob pattern such as **/*.json, newest first.",
            "input_schema": {"type": "object", "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string",
                         "description": "Directory to search (default: working directory)"},
            }, "required": ["pattern"]},
        },
        "Bash": {
            "name": "Bash",
            "description": ("Run ONE command, without a shell (no pipes, redirects, `;` "
                            "or `&&`). Only commands beginning with one of these are "
                            "allowed: " + "; ".join(grant.bash_prefixes)),
            "input_schema": {"type": "object", "properties": {
                "command": {"type": "string"},
                "timeout": {"type": "integer", "description": "Milliseconds (max 600000)"},
            }, "required": ["command"]},
        },
        "WebFetch": {
            "name": "WebFetch",
            "description": ("Fetch a URL and return its text (HTML converted to text, "
                            "truncated). Apply `prompt` to the returned text yourself."),
            "input_schema": {"type": "object", "properties": {
                "url": {"type": "string"},
                "prompt": {"type": "string"},
            }, "required": ["url"]},
        },
    }
    return [defs[n] for n in grant.names if n in defs]


def search_tool_type(model: str) -> str:
    """The server-side web search tool type for `model` (LLM_WEB_SEARCH_TOOL
    overrides it)."""
    override = os.environ.get("LLM_WEB_SEARCH_TOOL", "").strip()
    if override:
        return override
    name = model.lower()
    dynamic = ("opus-5", "opus-4-8", "opus-4-7", "opus-4-6", "sonnet-5", "sonnet-4-6",
               "fable-5")
    return "web_search_20260209" if any(m in name for m in dynamic) else "web_search_20250305"


class Workspace:
    """Executes the client-side tools, confined to `roots`."""

    def __init__(self, cwd: Path, roots: list[Path], grant: ToolGrant, env: dict,
                 fetch: Fetcher):
        self.cwd = cwd
        self.roots = roots
        self.grant = grant
        self.env = env
        self.fetch = fetch

    # -- paths --
    def _path(self, raw: str | None, *, default: Path | None = None) -> Path:
        if not raw:
            if default is None:
                raise ToolError("a path is required")
            return default
        p = Path(raw)
        p = (p if p.is_absolute() else self.cwd / p).resolve()
        if not any(p == r or r in p.parents for r in self.roots):
            raise ToolError(f"{raw} is outside the directories this call may read")
        return p

    def _rel(self, p: Path) -> str:
        try:
            return p.relative_to(self.cwd).as_posix()
        except ValueError:
            return str(p)

    def _files(self, base: Path) -> Iterator[Path]:
        if base.is_file():
            yield base
            return
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
            for name in filenames:
                yield Path(dirpath) / name

    @staticmethod
    def _text(p: Path, max_bytes: int = 20_000_000) -> str | None:
        try:
            if p.stat().st_size > max_bytes:
                return None
            data = p.read_bytes()
        except OSError:
            return None
        if b"\x00" in data[:8192]:
            return None
        return data.decode("utf-8", errors="replace")

    # -- tools --
    def Read(self, inp: dict) -> str:  # noqa: N802 - the tool's name
        p = self._path(inp.get("file_path"))
        if not p.is_file():
            raise ToolError(f"no such file: {inp.get('file_path')}")
        text = self._text(p)
        if text is None:
            raise ToolError(f"{inp.get('file_path')} is binary or too large to read")
        lines = text.splitlines()
        start = max(1, int(inp.get("offset") or 1))
        limit = max(1, int(inp.get("limit") or 2000))
        chunk = lines[start - 1:start - 1 + limit]
        out = "\n".join(f"{n:6}\t{line[:2000]}" for n, line in enumerate(chunk, start))
        if start - 1 + limit < len(lines):
            out += (f"\n... ({len(lines)} lines in all; continue with "
                    f"offset={start + limit})")
        return out or "(empty file)"

    def Glob(self, inp: dict) -> str:  # noqa: N802
        base = self._path(inp.get("path"), default=self.cwd)
        pattern = str(inp.get("pattern") or "")
        if not pattern or ".." in Path(pattern).parts:
            raise ToolError("a relative glob pattern is required")
        hits = []
        for p in base.glob(pattern):
            try:
                self._path(str(p))
            except ToolError:
                continue
            if p.is_file() and not (set(p.relative_to(base).parts) & _SKIP_DIRS):
                hits.append(p)
        hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        if not hits:
            return "No files found"
        more = f"\n... ({len(hits) - 200} more)" if len(hits) > 200 else ""
        return "\n".join(self._rel(p) for p in hits[:200]) + more

    def Grep(self, inp: dict) -> str:  # noqa: N802
        flags = re.IGNORECASE if inp.get("-i") else 0
        if inp.get("multiline"):
            flags |= re.MULTILINE | re.DOTALL
        try:
            rx = re.compile(str(inp.get("pattern") or ""), flags)
        except re.error as exc:
            raise ToolError(f"bad regular expression: {exc}") from None
        base = self._path(inp.get("path"), default=self.cwd)
        glob = inp.get("glob")
        mode = inp.get("output_mode") or "files_with_matches"
        numbers = inp.get("-n", True)
        before = int(inp.get("-B") or inp.get("-C") or 0)
        after = int(inp.get("-A") or inp.get("-C") or 0)
        head = int(inp.get("head_limit") or 250)
        out: list[str] = []
        for p in self._files(base):
            rel = self._rel(p)
            if glob and not (fnmatch.fnmatch(p.name, glob) or fnmatch.fnmatch(rel, glob)):
                continue
            text = self._text(p)
            if text is None:
                continue
            if inp.get("multiline"):
                n = len(rx.findall(text))
                if n:
                    out.append(rel if mode != "count" else f"{rel}:{n}")
                continue
            lines = text.splitlines()
            hit = [i for i, line in enumerate(lines) if rx.search(line)]
            if not hit:
                continue
            if mode == "files_with_matches":
                out.append(rel)
            elif mode == "count":
                out.append(f"{rel}:{len(hit)}")
            else:
                shown: set[int] = set()
                for i in hit:
                    for j in range(max(0, i - before), min(len(lines), i + after + 1)):
                        if j not in shown:
                            shown.add(j)
                            sep = ":" if j == i else "-"
                            prefix = f"{rel}{sep}{j + 1}{sep}" if numbers else f"{rel}{sep}"
                            out.append(prefix + lines[j][:2000])
            if len(out) >= head:
                break
        if not out:
            return "No matches found"
        tail = f"\n... (truncated at {head})" if len(out) > head else ""
        return "\n".join(out[:head]) + tail

    def Bash(self, inp: dict) -> str:  # noqa: N802
        command = str(inp.get("command") or "").strip()
        if _SHELL_OPS.search(command):
            raise ToolError("shell operators (pipes, redirects, ;, &&, `...`, $(...)) "
                            "are not supported: run one command")
        if not any(command == pre or command.startswith(pre + " ")
                   for pre in self.grant.bash_prefixes):
            raise ToolError("command not allowed; it must begin with one of: "
                            + "; ".join(self.grant.bash_prefixes))
        try:
            argv = shlex.split(command, posix=True)
        except ValueError as exc:
            raise ToolError(f"could not parse the command: {exc}") from None
        if argv and argv[0] in ("python", "python3", "py"):
            argv[0] = sys.executable
        timeout = min(max(int(inp.get("timeout") or 120_000), 1000), 600_000) / 1000
        proc = subprocess.Popen(argv, cwd=str(self.cwd), env=self.env,
                                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, encoding="utf-8",
                                errors="replace", **spawn_kwargs())
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            kill_process_tree(proc)
            proc.communicate()
            raise ToolError(f"command timed out after {timeout:g}s") from None
        text = (out or "") + (f"\n[stderr]\n{err}" if (err or "").strip() else "")
        if proc.returncode:
            raise ToolError(f"exit code {proc.returncode}\n{text}")
        return text or "(no output)"

    def WebFetch(self, inp: dict) -> str:  # noqa: N802
        url = str(inp.get("url") or "").strip()
        if urlsplit(url).scheme not in ("http", "https"):
            raise ToolError("an http(s) URL is required")
        status, text = self.fetch(url)
        if not (isinstance(status, int) and 200 <= status < 300):
            raise ToolError(f"{url}: {status}\n{text[:2000]}")
        return f"URL: {url}\nStatus: {status}\n\n{text}"

    def call(self, name: str, inp: object) -> tuple[str, bool]:
        """(content, is_error) for one tool_use block."""
        if name not in self.grant.names or name == "WebSearch":
            return f"{name} is not available to this call", True
        try:
            out = getattr(self, name)(inp if isinstance(inp, dict) else {})
            err = False
        except ToolError as exc:
            out, err = str(exc), True
        except Exception as exc:  # a tool bug must not end the call
            out, err = f"{name} failed: {type(exc).__name__}: {exc}", True
        if len(out) > MAX_TOOL_CHARS:
            out = out[:MAX_TOOL_CHARS] + f"\n... (truncated at {MAX_TOOL_CHARS} chars)"
        return out, err


def html_to_text(body: str) -> str:
    body = re.sub(r"(?is)<(script|style|noscript)\b.*?</\1>", " ", body)
    body = re.sub(r"(?i)<br\s*/?>|</(p|div|li|tr|h[1-6])>", "\n", body)
    body = html.unescape(re.sub(r"(?s)<[^>]+>", " ", body))
    body = re.sub(r"[ \t\f\v]+", " ", body)
    return re.sub(r"\n\s*\n+", "\n\n", body).strip()


def default_fetch(url: str, *, timeout: int = 30,
                  blocked_hosts: Mapping[str, str] | None = None) -> tuple[object, str]:
    """GET `url`: (HTTP status or an error string, page text). Never raises.

    `blocked_hosts` maps a hostname to the remedy to report instead of
    fetching it - for sites that refuse automated access, say.
    """
    host = (urlsplit(url).hostname or "").lower()
    if blocked_hosts and host in blocked_hosts:
        return "blocked", f"{host} may not be fetched; {blocked_hosts[host]}"
    req = urllib.request.Request(url, headers={
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"),
        "Accept": "*/*"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read(5_000_000)
            status = getattr(resp, "status", 200) or 200
            charset = resp.headers.get_content_charset() or "utf-8"
            ctype = resp.headers.get_content_type()
    except Exception as exc:  # noqa: BLE001 - the status is the answer
        return getattr(exc, "code", None) or f"error: {exc}"[:200], ""
    text = raw.decode(charset, errors="replace")
    if "html" in ctype:
        text = html_to_text(text)
    return status, text
