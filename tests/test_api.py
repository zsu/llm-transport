"""run_api: the tool loop, its boundaries and its failure envelopes."""

import pytest
from conftest import APIError, FakeClient, text, tool

import llm_transport
from llm_transport import LLMConfig, ToolGrant, run_api

CFG = LLMConfig(use_api=True, api_url="https://gw.example", api_key="k",
                model="claude-opus-5-5", max_tokens=1000, max_turns=5, effort="high",
                request_retries=0, ca_bundle="")


def _results(client, turn=1):
    """The tool_result blocks sent back on request `turn`."""
    return client.requests[turn]["messages"][-1]["content"]


def _cfg(**kw):
    return LLMConfig(**{**CFG.__dict__, **kw})


# ---------------------------------- the loop ---------------------------------

def test_missing_key_is_an_authentication_error(tmp_path):
    env = run_api("p", cwd=tmp_path, config=_cfg(api_key=""))
    assert env["is_error"] and env["api_error_status"] == 401
    assert "authentication_error" in env["result"]


def test_plain_reply_is_a_cli_shaped_envelope(tmp_path):
    client = FakeClient(text('{"ok": 1}'))
    env = run_api("prompt", cwd=tmp_path, config=CFG, client=client)
    assert env["result"] == '{"ok": 1}'
    assert not env["is_error"] and env["subtype"] == "success"
    assert env["stop_reason"] == "end_turn" and env["num_turns"] == 1
    assert env["usage"]["input_tokens"] == 100
    req = client.requests[0]
    assert req["model"] == "claude-opus-5-5" and req["max_tokens"] == 1000
    assert req["output_config"] == {"effort": "high"}
    assert "tools" not in req  # nothing granted
    block = req["messages"][0]["content"][0]
    assert block["text"] == "prompt" and block["cache_control"] == {"type": "ephemeral"}


def test_blank_effort_and_system_prompt(tmp_path):
    client = FakeClient(text("x"))
    run_api("p", cwd=tmp_path, config=_cfg(effort=""), system="be brief", client=client)
    assert "output_config" not in client.requests[0]
    assert client.requests[0]["system"] == "be brief"


def test_tool_loop_reads_a_file_and_sums_usage(tmp_path):
    (tmp_path / "a.txt").write_text("alpha\nbeta\n", encoding="utf-8")
    client = FakeClient(tool("Read", {"file_path": "a.txt"}), text("done"))
    env = run_api("p", cwd=tmp_path, allowed_tools="Read,Grep,Glob",
                  config=CFG, client=client)
    assert env["result"] == "done" and env["num_turns"] == 2
    assert env["usage"]["input_tokens"] == 200
    res = _results(client)[0]
    assert res["tool_use_id"] == "t1" and not res["is_error"]
    assert "1\talpha" in res["content"] and "2\tbeta" in res["content"]
    assert [t["name"] for t in client.requests[0]["tools"]] == ["Read", "Grep", "Glob"]
    # The assistant turn goes back whole, ahead of the tool results.
    assert client.requests[1]["messages"][1]["role"] == "assistant"


def test_read_pages_a_long_file(tmp_path):
    (tmp_path / "long.txt").write_text("\n".join(map(str, range(1, 11))), encoding="utf-8")
    client = FakeClient(tool("Read", {"file_path": "long.txt", "offset": 3, "limit": 2}),
                        text("ok"))
    run_api("p", cwd=tmp_path, allowed_tools="Read", config=CFG, client=client)
    out = _results(client)[0]["content"]
    assert "3\t3" in out and "4\t4" in out and "5\t5" not in out
    assert "offset=5" in out


def test_parallel_tool_calls_answer_in_one_message(tmp_path):
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    both = {"content": [
        {"type": "tool_use", "id": "x", "name": "Read", "input": {"file_path": "a.txt"}},
        {"type": "tool_use", "id": "y", "name": "Glob", "input": {"pattern": "*.txt"}}],
        "stop_reason": "tool_use", "usage": {}}
    client = FakeClient(both, text("ok"))
    run_api("p", cwd=tmp_path, allowed_tools="Read,Glob", config=CFG, client=client)
    assert [r["tool_use_id"] for r in _results(client)] == ["x", "y"]


def test_pause_turn_is_resumed(tmp_path):
    client = FakeClient(text("", stop="pause_turn"), text("final"))
    env = run_api("p", cwd=tmp_path, config=CFG, client=client)
    assert env["result"] == "final" and len(client.requests) == 2


def test_tools_empty_string_grants_nothing(tmp_path):
    client = FakeClient(text("x"))
    run_api("p", cwd=tmp_path, allowed_tools="Read,Grep", tools="", config=CFG,
            client=client)
    assert "tools" not in client.requests[0]


def test_ungranted_tool_call_is_refused(tmp_path):
    client = FakeClient(tool("Bash", {"command": "python -c 1"}), text("ok"))
    run_api("p", cwd=tmp_path, allowed_tools="Read", config=CFG, client=client)
    res = _results(client)[0]
    assert res["is_error"] and "not available" in res["content"]


def test_grep_and_glob(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "x.json").write_text('{"cost": 33}\n{"cost": 32}\n',
                                             encoding="utf-8")
    client = FakeClient(tool("Grep", {"pattern": "cost.*3[23]", "output_mode": "content"}),
                        tool("Grep", {"pattern": "cost", "output_mode": "count"}),
                        tool("Glob", {"pattern": "**/*.json"}), text("ok"))
    run_api("p", cwd=tmp_path, allowed_tools="Grep,Glob", config=CFG, client=client)
    grep = _results(client, 1)[0]["content"]
    assert "sub/x.json:1:" in grep and "sub/x.json:2:" in grep
    assert _results(client, 2)[0]["content"] == "sub/x.json:2"
    assert _results(client, 3)[0]["content"] == "sub/x.json"


# -------------------------------- boundaries ---------------------------------

def test_file_tools_are_confined_to_the_working_directory(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (tmp_path / "secret.txt").write_text("s", encoding="utf-8")
    client = FakeClient(tool("Read", {"file_path": "../secret.txt"}),
                        tool("Read", {"file_path": str(tmp_path / "secret.txt")}),
                        tool("Grep", {"pattern": "s", "path": str(tmp_path)}),
                        tool("Glob", {"pattern": "../*"}),
                        text("ok"))
    run_api("p", cwd=root, allowed_tools="Read,Grep,Glob", config=CFG, client=client)
    for turn in (1, 2, 3, 4):
        assert _results(client, turn)[0]["is_error"]


def test_run_dir_and_read_roots_are_readable(tmp_path):
    root, run, extra = tmp_path / "repo", tmp_path / "run", tmp_path / "extra"
    for d in (root, run, extra):
        d.mkdir()
    (run / "m.json").write_text("{}", encoding="utf-8")
    (extra / "n.json").write_text("{}", encoding="utf-8")
    client = FakeClient(tool("Read", {"file_path": str(run / "m.json")}),
                        tool("Read", {"file_path": str(extra / "n.json")}), text("ok"))
    run_api("p", cwd=root, allowed_tools="Read", env={"RUN_DIR": str(run)},
            read_roots=(extra,), config=CFG, client=client)
    assert not _results(client, 1)[0]["is_error"]
    assert not _results(client, 2)[0]["is_error"]


def test_grant_parsing():
    g = ToolGrant.parse("Read,Bash(python scripts/lookup.py*),Bash(git log:*)", "Write")
    assert g.names == ["Read", "Bash"]
    assert g.bash_prefixes == ["python scripts/lookup.py", "git log"]
    assert ToolGrant.parse("Bash(python x.py*)", "Bash").names == []  # deny wins
    bare = ToolGrant.parse("Bash")
    assert bare.names == [] and any("bare Bash" in n for n in bare.notes)
    assert any("Write" in n for n in ToolGrant.parse("Write").notes)
    assert ToolGrant.parse("Read,Grep", tools="Read").names == ["Read"]


@pytest.mark.parametrize("command,why", [
    ("python -c print(1)", "must begin with"),
    ("python scripts/lookup.pyx", "must begin with"),
    ("python scripts/lookup.py 1; rm -rf /", "shell operators"),
    ("python scripts/lookup.py 1 | head", "shell operators"),
    ("python scripts/lookup.py 1 > out.txt", "shell operators"),
    ("python scripts/lookup.py $(whoami)", "shell operators"),
    ("python scripts/lookup.py `whoami`", "shell operators"),
    ("python scripts/lookup.py 1 && python -c 1", "shell operators"),
    ("python scripts/lookup.py 1\npython -c 1", "shell operators"),
])
def test_bash_refuses_anything_but_the_granted_prefix(tmp_path, command, why):
    client = FakeClient(tool("Bash", {"command": command}), text("ok"))
    run_api("p", cwd=tmp_path, allowed_tools="Bash(python scripts/lookup.py*)",
            config=CFG, client=client)
    res = _results(client)[0]
    assert res["is_error"] and why in res["content"]


def test_bash_runs_the_granted_script_without_a_shell(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "echo.py").write_text(
        "import os, sys; print(sys.argv[1:], os.environ.get('RUN_DIR'))", encoding="utf-8")
    client = FakeClient(tool("Bash", {"command": "python scripts/echo.py 'a b' $HOME"}),
                        text("ok"))
    run_api("p", cwd=tmp_path, allowed_tools="Bash(python scripts/echo.py*)",
            env={"RUN_DIR": "rd"}, config=CFG, client=client)
    res = _results(client)[0]
    assert not res["is_error"]
    # No shell: the quoted argument stays whole and $HOME is not expanded.
    assert "['a b', '$HOME'] rd" in res["content"]


def test_bash_reports_a_failing_command(tmp_path):
    (tmp_path / "fail.py").write_text("import sys; print('out'); sys.exit(3)",
                                      encoding="utf-8")
    client = FakeClient(tool("Bash", {"command": "python fail.py"}), text("ok"))
    run_api("p", cwd=tmp_path, allowed_tools="Bash(python fail.py*)", config=CFG,
            client=client)
    res = _results(client)[0]
    assert res["is_error"] and "exit code 3" in res["content"] and "out" in res["content"]


def test_webfetch_uses_the_injected_fetcher(tmp_path):
    seen = []

    def fetch(url):
        seen.append(url)
        return 200, "page text"

    client = FakeClient(tool("WebFetch", {"url": "https://example.gov/x"}),
                        tool("WebFetch", {"url": "file:///etc/passwd"}), text("ok"))
    run_api("p", cwd=tmp_path, allowed_tools="WebFetch", fetch=fetch, config=CFG,
            client=client)
    assert seen == ["https://example.gov/x"]
    assert "page text" in _results(client, 1)[0]["content"]
    assert _results(client, 2)[0]["is_error"]


def test_blocked_hosts_are_refused_without_a_request(tmp_path):
    client = FakeClient(tool("WebFetch", {"url": "https://www.example.org/p"}), text("ok"))
    run_api("p", cwd=tmp_path, allowed_tools="WebFetch", config=CFG, client=client,
            blocked_hosts={"www.example.org": "use the API instead"})
    res = _results(client)[0]
    assert res["is_error"] and "use the API instead" in res["content"]
    status, body = llm_transport.default_fetch(
        "https://www.example.org/p", blocked_hosts={"www.example.org": "no"})
    assert status == "blocked"


def test_html_is_converted_to_text():
    from llm_transport.tools import html_to_text
    page = "<html><script>x()</script><p>One &amp; two</p><div>three</div></html>"
    out = html_to_text(page)
    assert "x()" not in out and "One & two" in out and "three" in out


def test_web_search_is_a_server_tool_dropped_where_unsupported(tmp_path):
    client = FakeClient(
        APIError(400, "invalid_request_error", "tool type web_search_20260209 not supported"),
        text("ok"))
    env = run_api("p", cwd=tmp_path, allowed_tools="Read,WebSearch", config=CFG,
                  client=client)
    first, second = client.requests
    assert {"type": "web_search_20260209", "name": "web_search", "max_uses": 10} \
        in first["tools"]
    assert all(t.get("name") != "web_search" for t in second["tools"])
    assert not env["is_error"]
    assert any("web search is not available" in w for w in env["warnings"])


def test_older_models_get_the_basic_search_tool(tmp_path):
    client = FakeClient(text("ok"))
    run_api("p", cwd=tmp_path, allowed_tools="WebSearch", config=_cfg(model="claude-haiku-4-5"),
            client=client)
    assert client.requests[0]["tools"][0]["type"] == "web_search_20250305"


# ---------------------------------- failures ---------------------------------

@pytest.mark.parametrize("status,etype,msg", [
    (401, "authentication_error", "invalid x-api-key"),
    (529, "overloaded_error", "Overloaded"),
    (400, "budget_exceeded", "ExceededBudget: Crossed spend within budget"),
])
def test_api_errors_become_error_envelopes(tmp_path, status, etype, msg):
    env = run_api("p", cwd=tmp_path, config=CFG,
                  client=FakeClient(APIError(status, etype, msg)))
    assert env["is_error"] and env["api_error_status"] == status
    assert etype in env["result"] and msg in env["result"]


def test_connection_errors_have_no_status(tmp_path):
    class APIConnectionError(Exception):
        pass

    env = run_api("p", cwd=tmp_path, config=CFG,
                  client=FakeClient(APIConnectionError("reset by peer")))
    assert env["is_error"] and env["api_error_status"] is None
    assert env["result"].startswith("connection error")


def test_turn_cap(tmp_path):
    client = FakeClient(tool("Glob", {"pattern": "*"}), tool("Glob", {"pattern": "*"}))
    env = run_api("p", cwd=tmp_path, allowed_tools="Glob", config=_cfg(max_turns=2),
                  client=client)
    assert env["is_error"] and env["subtype"] == "error_max_turns"


def test_spend_cap_stops_the_loop(tmp_path):
    reply = tool("Glob", {"pattern": "*"})
    reply["usage"] = {"input_tokens": 1_000_000, "output_tokens": 0}
    env = run_api("p", cwd=tmp_path, allowed_tools="Glob", max_usd=1.0, config=CFG,
                  client=FakeClient(reply))
    assert env["is_error"] and env["subtype"] == "error_max_budget_usd"
    assert "max-budget-usd" in env["result"]
    assert env["total_cost_usd"] == pytest.approx(4.0)  # $4 / MTok input


def test_timeout_stops_the_loop(tmp_path):
    env = run_api("p", cwd=tmp_path, allowed_tools="Glob", timeout_s=1e-9, config=CFG,
                  client=FakeClient(text("never")))
    assert env["is_error"] and "timed out" in env["result"]


def test_gateway_reported_cost_wins(tmp_path):
    env = run_api("p", cwd=tmp_path, config=CFG,
                  client=FakeClient(text("x"), headers={"x-litellm-response-cost": "0.5"}))
    assert env["total_cost_usd"] == 0.5


def test_unknown_model_is_priced_at_zero_with_a_warning(tmp_path, monkeypatch):
    cfg = _cfg(model="my-gateway-alias")
    env = run_api("p", cwd=tmp_path, config=cfg, client=FakeClient(text("x")))
    assert env["total_cost_usd"] == 0
    assert any("no price known" in w for w in env["warnings"])
    monkeypatch.setenv("LLM_PRICE_INPUT_PER_MTOK", "1")
    monkeypatch.setenv("LLM_PRICE_OUTPUT_PER_MTOK", "1")
    env = run_api("p", cwd=tmp_path, config=cfg, client=FakeClient(text("x")))
    assert env["total_cost_usd"] == pytest.approx(110 / 1e6)


def test_refusal_is_an_error(tmp_path):
    reply = text("", stop="refusal")
    reply["stop_details"] = {"category": "cyber"}
    env = run_api("p", cwd=tmp_path, config=CFG, client=FakeClient(reply))
    assert env["is_error"] and env["stop_reason"] == "refusal"
    assert "cyber" in env["result"]


def test_max_tokens_is_reported_not_hidden(tmp_path):
    env = run_api("p", cwd=tmp_path, config=CFG,
                  client=FakeClient(text('{"partial', stop="max_tokens")))
    assert env["stop_reason"] == "max_tokens" and env["result"] == '{"partial'
