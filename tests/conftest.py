import copy
import sys

import pytest


class _Stream:
    def __init__(self, reply, headers):
        self.reply = reply
        self.response = type("R", (), {"headers": headers})()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get_final_message(self):
        return self.reply


class FakeClient:
    """Answers each `messages.stream` with the next scripted reply (plain
    dicts, read the same way as SDK objects); an Exception is raised instead."""

    def __init__(self, *replies, headers=None):
        self.replies = list(replies)
        self.requests = []
        self.headers = headers or {}
        self.messages = self

    def stream(self, **request):
        self.requests.append(copy.deepcopy(request))  # the loop mutates messages
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return _Stream(reply, self.headers)


class APIError(Exception):
    """Shaped like anthropic.APIStatusError."""

    def __init__(self, status, etype, message):
        super().__init__(message)
        self.status_code = status
        self.body = {"type": "error", "error": {"type": etype, "message": message}}


def text(t, stop="end_turn", usage=None):
    return {"content": [{"type": "text", "text": t}], "stop_reason": stop,
            "usage": usage or {"input_tokens": 100, "output_tokens": 10}}


def tool(name, inp, id_="t1"):
    return {"content": [{"type": "tool_use", "id": id_, "name": name, "input": inp}],
            "stop_reason": "tool_use", "usage": {"input_tokens": 100, "output_tokens": 10}}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """No test sees the developer's own LLM / Anthropic configuration."""
    for name in ("LLM_USE_API", "LLM_API_URL", "LLM_API_KEY", "LLM_MODEL", "LLM_EFFORT",
                 "LLM_MAX_TOKENS", "LLM_MAX_TURNS", "LLM_PRICE_INPUT_PER_MTOK",
                 "LLM_PRICE_OUTPUT_PER_MTOK", "LLM_WEB_SEARCH_TOOL", "LLM_CA_BUNDLE",
                 "ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL",
                 "CLAUDE_BIN"):
        monkeypatch.delenv(name, raising=False)


_STUB = """\
import json, os, sys
if "--help" in sys.argv:
    print("--output-format --allowed-tools --permission-mode --tools")
    sys.exit(0)
mode = os.environ.get("FAKE_MODE", "ok")
prompt = sys.stdin.read()
if mode == "crash":
    sys.stderr.write("boom")
    sys.exit(2)
if mode == "sleep":
    import time
    time.sleep(30)
print(json.dumps({"type": "result", "subtype": "success", "is_error": False,
                  "result": prompt, "num_turns": 1, "total_cost_usd": 0.01,
                  "usage": {}, "argv": sys.argv[1:],
                  "key": os.environ.get("ANTHROPIC_API_KEY"),
                  "url": os.environ.get("ANTHROPIC_BASE_URL")}))
"""


@pytest.fixture
def fake_claude(monkeypatch, tmp_path):
    """Point CLAUDE_BIN at a stub that echoes its prompt, argv and auth env."""
    stub = tmp_path / "fake_claude.py"
    stub.write_text(_STUB, encoding="utf-8")
    monkeypatch.setenv("CLAUDE_BIN", f'"{sys.executable}" "{stub}"')
    from llm_transport import cli
    cli.cli_flags.cache_clear()
    yield
    cli.cli_flags.cache_clear()
