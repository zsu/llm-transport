"""The result shape both transports return.

It is the JSON envelope `claude -p --output-format json` prints, key for key,
plus a `warnings` list, so code that already parses the CLI's output can
switch transport without touching its error handling.
"""

from __future__ import annotations


def usage_dict(i: int = 0, o: int = 0, cr: int = 0, cw: int = 0) -> dict:
    return {"input_tokens": i, "output_tokens": o,
            "cache_read_input_tokens": cr, "cache_creation_input_tokens": cw}


def envelope(*, result: str = "", is_error: bool = False, subtype: str = "success",
             stop_reason: str = "", status: int | None = None, turns: int = 0,
             usage: dict | None = None, cost: float = 0.0,
             warnings: list[str] | None = None) -> dict:
    return {
        "type": "result",
        "subtype": subtype,
        "is_error": is_error,
        "result": result,
        "stop_reason": stop_reason,
        "api_error_status": status,
        "num_turns": turns,
        "usage": usage or usage_dict(),
        "total_cost_usd": round(cost, 6),
        "warnings": list(warnings or []),
    }
