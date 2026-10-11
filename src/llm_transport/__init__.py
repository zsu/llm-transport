"""Run a prompt through the `claude` CLI or the Anthropic Messages API.

    import llm_transport

    env = llm_transport.run("Summarize README.md", cwd=".", allowed_tools="Read")
    if env["is_error"]:
        raise RuntimeError(env["result"])
    print(env["result"], env["total_cost_usd"])

LLM_USE_API picks the transport (default: the CLI); LLM_API_URL and
LLM_API_KEY point it at Anthropic or a LiteLLM gateway. Either way the result
is the envelope `claude -p --output-format json` prints.
"""

from llm_transport.api import api_error, run_api
from llm_transport.cli import (
    CLITimeout,
    cli_env,
    kill_process_tree,
    resolve_cli,
    run_cli,
    run_cli_raw,
    spawn_kwargs,
)
from llm_transport.config import LLMConfig, describe, use_api
from llm_transport.dispatch import run
from llm_transport.envelope import envelope
from llm_transport.pricing import PRICES, cost_of, price_for
from llm_transport.tools import SUPPORTED_TOOLS, ToolGrant, default_fetch

__version__ = "0.1.0"

__all__ = [
    "PRICES",
    "SUPPORTED_TOOLS",
    "CLITimeout",
    "LLMConfig",
    "ToolGrant",
    "__version__",
    "api_error",
    "cli_env",
    "cost_of",
    "default_fetch",
    "describe",
    "envelope",
    "kill_process_tree",
    "price_for",
    "resolve_cli",
    "run",
    "run_api",
    "run_cli",
    "run_cli_raw",
    "spawn_kwargs",
    "use_api",
]
