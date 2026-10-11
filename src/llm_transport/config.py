"""Configuration from the environment.

    LLM_USE_API   unset, blank, 0, false -> the `claude` CLI (the default)
                  1, true, yes, on       -> the Anthropic Messages API
    LLM_API_URL   base URL; else ANTHROPIC_BASE_URL; else https://api.anthropic.com
    LLM_API_KEY   else ANTHROPIC_API_KEY
    LLM_MODEL     else ANTHROPIC_MODEL; else claude-opus-5-5

`LLM_API_KEY` / `LLM_API_URL` drive both transports: in CLI mode they are
handed to `claude` as ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL when those are
unset (see `cli.cli_env`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlsplit

DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_API_URL = "https://api.anthropic.com"
_TRUE = {"1", "true", "yes", "on"}


def env_first(*names: str, default: str = "") -> str:
    """The first of `names` set to a non-blank value."""
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return default


def env_num(name: str, default: float) -> float:
    """Blank or unparseable reads as the default rather than raising: an empty
    variable exported by a compose file must not take a caller down."""
    try:
        return float(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


@dataclass(frozen=True)
class LLMConfig:
    use_api: bool
    api_url: str
    api_key: str
    model: str
    max_tokens: int
    max_turns: int
    effort: str
    request_retries: int
    ca_bundle: str

    @classmethod
    def from_env(cls) -> LLMConfig:
        return cls(
            use_api=os.environ.get("LLM_USE_API", "").strip().lower() in _TRUE,
            api_url=env_first("LLM_API_URL", "ANTHROPIC_BASE_URL",
                              default=DEFAULT_API_URL).rstrip("/"),
            api_key=env_first("LLM_API_KEY", "ANTHROPIC_API_KEY"),
            model=env_first("LLM_MODEL", "ANTHROPIC_MODEL", default=DEFAULT_MODEL),
            max_tokens=int(env_num("LLM_MAX_TOKENS", 32000)),
            max_turns=max(1, int(env_num("LLM_MAX_TURNS", 40))),
            # Blank is meaningful (omit the field), so only an UNSET variable
            # takes the default.
            effort=os.environ.get("LLM_EFFORT", "high").strip(),
            request_retries=max(0, int(env_num("LLM_REQUEST_RETRIES", 4))),
            ca_bundle=env_first("LLM_CA_BUNDLE", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"),
        )

    @property
    def host(self) -> str:
        return urlsplit(self.api_url).hostname or self.api_url


def use_api() -> bool:
    """True when LLM_USE_API selects the direct API; the CLI otherwise."""
    return LLMConfig.from_env().use_api


def describe() -> str:
    """One line for a run log naming the transport in use. Never the key."""
    cfg = LLMConfig.from_env()
    if cfg.use_api:
        return f"llm=api model={cfg.model} host={cfg.host}"
    return f"llm=cli bin={os.environ.get('CLAUDE_BIN', 'claude')}"
