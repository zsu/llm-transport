"""Cost estimates for when the endpoint reports none.

A LiteLLM gateway reports what each request cost in `x-litellm-response-cost`,
and that figure wins. Without it the cost is estimated from first-party list
prices; Bedrock and Vertex bill at their own rates, so behind a gateway that
omits the header this is an estimate.
"""

from __future__ import annotations

from llm_transport.config import env_num

# USD per million tokens (input, output), first-party list prices as of
# 2026-10. Cache reads bill at 0.1x input, 5-minute cache writes at 1.25x.
# Longest matching key wins, so "claude-opus-5-5" beats "claude-opus-5".
PRICES = {
    "claude-fable-5-1": (10.0, 50.0),
    "claude-fable-5": (10.0, 50.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-haiku-4-5": (1.0, 5.0),
}
WEB_SEARCH_USD = 0.01  # per search ($10 per 1,000)


def price_for(model: str) -> tuple[float, float] | None:
    """(input, output) USD per MTok for `model`, or None if unknown.

    LLM_PRICE_INPUT_PER_MTOK / LLM_PRICE_OUTPUT_PER_MTOK override the table -
    for a gateway alias it cannot recognise, or negotiated rates.
    """
    override_in = env_num("LLM_PRICE_INPUT_PER_MTOK", -1)
    override_out = env_num("LLM_PRICE_OUTPUT_PER_MTOK", -1)
    if override_in >= 0 and override_out >= 0:
        return override_in, override_out
    # Bedrock-style ids ("us.anthropic.claude-opus-5-5-v1:0") still match.
    name = model.lower().replace(".", "-")
    hits = [k for k in PRICES if k in name]
    return PRICES[max(hits, key=len)] if hits else None


def cost_of(model: str, usage: dict, searches: int = 0) -> float | None:
    price = price_for(model)
    if price is None:
        return None
    p_in, p_out = price
    return (usage.get("input_tokens", 0) * p_in
            + usage.get("cache_read_input_tokens", 0) * p_in * 0.1
            + usage.get("cache_creation_input_tokens", 0) * p_in * 1.25
            + usage.get("output_tokens", 0) * p_out) / 1e6 + searches * WEB_SEARCH_USD
