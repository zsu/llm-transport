import pytest

import llm_transport
from llm_transport import LLMConfig


@pytest.mark.parametrize("value,expected", [
    (None, False), ("", False), ("0", False), ("false", False), ("no", False),
    ("1", True), ("true", True), ("TRUE", True), ("yes", True), ("on", True),
])
def test_switch_defaults_to_cli(monkeypatch, value, expected):
    if value is not None:
        monkeypatch.setenv("LLM_USE_API", value)
    assert llm_transport.use_api() is expected


def test_url_key_and_model_precedence(monkeypatch):
    cfg = LLMConfig.from_env()
    assert cfg.api_url == "https://api.anthropic.com"
    assert cfg.model == "claude-opus-5-5"
    assert cfg.api_key == ""
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://gw.example/")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    cfg = LLMConfig.from_env()
    assert (cfg.api_url, cfg.api_key) == ("https://gw.example", "a")
    monkeypatch.setenv("LLM_API_URL", "https://other.example")
    monkeypatch.setenv("LLM_API_KEY", "b")
    cfg = LLMConfig.from_env()
    assert (cfg.api_url, cfg.api_key, cfg.host) == ("https://other.example", "b",
                                                    "other.example")


def test_blank_effort_omits_the_field(monkeypatch):
    assert LLMConfig.from_env().effort == "high"
    monkeypatch.setenv("LLM_EFFORT", "")
    assert LLMConfig.from_env().effort == ""


def test_bad_numbers_fall_back_to_defaults(monkeypatch):
    monkeypatch.setenv("LLM_MAX_TURNS", "lots")
    monkeypatch.setenv("LLM_MAX_TOKENS", "")
    cfg = LLMConfig.from_env()
    assert (cfg.max_turns, cfg.max_tokens) == (40, 32000)


def test_describe_never_prints_the_key(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "sk-secret")
    assert "sk-secret" not in llm_transport.describe()
    monkeypatch.setenv("LLM_USE_API", "1")
    line = llm_transport.describe()
    assert "sk-secret" not in line and "llm=api" in line


def test_price_table():
    assert llm_transport.price_for("claude-opus-5-5") == (4.0, 20.0)
    assert llm_transport.price_for("claude-opus-5") == (5.0, 25.0)  # longest match
    assert llm_transport.price_for("us.anthropic.claude-sonnet-5-5-v1:0") == (2.0, 10.0)
    assert llm_transport.price_for("my-alias") is None
    usage = {"input_tokens": 1_000_000, "cache_read_input_tokens": 1_000_000,
             "cache_creation_input_tokens": 1_000_000, "output_tokens": 1_000_000}
    assert llm_transport.cost_of("claude-opus-5-5", usage, searches=2) == pytest.approx(
        4 + 0.4 + 5 + 20 + 0.02)
