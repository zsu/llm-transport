import llm_transport
from llm_transport import cli_env, run_cli


def test_cli_returns_the_envelope_and_takes_the_prompt_on_stdin(tmp_path, fake_claude):
    env = run_cli("hello", cwd=tmp_path, allowed_tools="Read")
    assert not env["is_error"] and env["result"] == "hello"
    assert env["argv"][:3] == ["-p", "--output-format", "json"]
    assert env["argv"][3:5] == ["--allowed-tools", "Read"]


def test_unadvertised_flags_are_dropped_and_reported(tmp_path, fake_claude):
    env = llm_transport.run("p", cwd=tmp_path, max_usd=2, tools="",
                            cli_flags_optional=("--permission-mode", "manual"))
    assert "--max-budget-usd" not in env["argv"]
    assert env["dropped_flags"] == ["--max-budget-usd"]
    assert any("--max-budget-usd" in w for w in env["warnings"])
    assert env["argv"][env["argv"].index("--tools") + 1] == ""
    assert "--permission-mode" in env["argv"]


def test_cli_failures_are_error_envelopes(tmp_path, fake_claude, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "crash")
    env = run_cli("p", cwd=tmp_path)
    assert env["is_error"] and "boom" in env["result"]
    monkeypatch.setenv("FAKE_MODE", "sleep")
    env = run_cli("p", cwd=tmp_path, timeout_s=1)
    assert env["is_error"] and "timed out" in env["result"]


def test_missing_cli_is_an_error_envelope(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_BIN", str(tmp_path / "no-such-claude"))
    env = run_cli("p", cwd=tmp_path)
    assert env["is_error"] and "could not be started" in env["result"]


def test_llm_key_and_url_reach_the_cli(tmp_path, fake_claude, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "llm-key")
    monkeypatch.setenv("LLM_API_URL", "https://gw.example")
    env = run_cli("p", cwd=tmp_path)
    assert (env["key"], env["url"]) == ("llm-key", "https://gw.example")


def test_cli_env_never_overrides_anthropic_names(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "llm-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "theirs")
    env = cli_env({"X": "1"})
    assert env["ANTHROPIC_API_KEY"] == "theirs" and env["X"] == "1"
    assert env["PYTHONUTF8"] == "1"


def test_run_dispatches_on_the_switch(monkeypatch, tmp_path):
    from llm_transport import dispatch
    seen = {}
    monkeypatch.setattr(dispatch, "run_api", lambda *a, **k: seen.setdefault("api", k))
    monkeypatch.setattr(dispatch, "run_cli", lambda *a, **k: seen.setdefault("cli", k))
    llm_transport.run("p", cwd=tmp_path, max_usd=2, tools="")
    assert seen["cli"]["optional_flags"] == ("--tools", "", "--max-budget-usd", "2")
    monkeypatch.setenv("LLM_USE_API", "1")
    llm_transport.run("p", cwd=tmp_path, max_usd=2, read_roots=("x",))
    assert seen["api"]["max_usd"] == 2 and seen["api"]["read_roots"] == ("x",)
