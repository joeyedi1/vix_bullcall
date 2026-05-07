"""Liquidity gates for spread fills.

ARCHITECTURE §5.3. A spread that fails any gate is REJECTED — it is not
silently filled at a worse theoretical price. Rejections are reported as
a first-class category in backtest output, alongside P&L, so the
"attractive equity curve with a 60% rejection rate is not a strategy"
failure mode is observable.

First-pass evaluation set (Phase-4):
  - reject_no_bid_short_leg: short leg has bid <= 0
  - reject_locked_or_crossed: bid == ask  /  bid > ask  on either leg
  - max_quote_age_seconds: quote_age_seconds <= threshold per leg
  - min_displayed_size: smallest leg/side displayed size >= threshold
  - max_leg_spread_pct: (ask - bid) / mid <= threshold per leg
  - max_order_size_pct_of_displayed: order_size <= pct * displayed (the
    side being crossed: long.ask_size on entry-buy, short.bid_size on
    entry-sell)

Gates NOT yet wired (need data not on OptionQuote):
  - min_leg_open_interest, min_leg_volume_today (require daily chain ref)
  - max_order_size_pct_of_oi (same)
These will be wired when the daily-chain side-channel reaches FillEngine.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LiquidityGates:
    """Per-leg and per-spread acceptance criteria for spread fills."""

    max_leg_spread_pct: float = 0.15
    min_leg_open_interest: int = 0
    min_leg_volume_today: int = 0
    min_displayed_size: int = 0
    max_quote_age_seconds: float = 30.0
    reject_locked_or_crossed: bool = True
    reject_no_bid_short_leg: bool = True
    max_order_size_pct_of_displayed: float = 0.5
    max_order_size_pct_of_oi: float = 0.05
