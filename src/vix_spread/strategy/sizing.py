"""Position sizing — `FixedRiskSizer`.

ARCHITECTURE §7.1 / §10.1. The Phase-5 default sizer: every entry risks
a fixed percentage of equity. Picked because:
  - It's the simplest reproducible rule that bounds per-trade max-loss.
  - It's the least-coupled-to-signal sizing — better for regime-stability
    debugging than Kelly or filtered-prob-scaled sizing.
  - It surfaces "the budget is too small for this debit" cleanly as
    `size = 0`, which propagates up as a no-entry decision instead of
    a silent fractional position.

For a bull call spread, max-loss equals the debit paid (you cannot lose
more than the premium put up).  Solve for integer contracts:

    risk_dollars   = equity * risk_per_trade_pct
    max_loss_per   = debit_per_spread * option_multiplier
    size           = floor(risk_dollars / max_loss_per)

`option_multiplier` is read from the spread's long leg
(`Product.option_multiplier()`), so VIX-index ($100) and VX-future
($1000) products size correctly without the caller passing the value
explicitly.
"""
from __future__ import annotations

from vix_spread.products.spread import BullCallSpread


class FixedRiskSizer:
    """Sizes a bull call spread to risk a fixed % of equity per trade.

    Returns 0 on:
      - non-positive equity
      - non-positive debit (a credit / zero-cost spread can't lose
        anything; sizing is undefined for the fixed-risk rule)
      - debit larger than the dollar risk budget (one contract would
        exceed the per-trade risk cap)
    """

    def __init__(self, risk_per_trade_pct: float) -> None:
        if not (0.0 < risk_per_trade_pct < 1.0):
            raise ValueError(
                f"risk_per_trade_pct must be in (0, 1); got {risk_per_trade_pct}."
            )
        self.risk_per_trade_pct = float(risk_per_trade_pct)

    def size(
        self,
        *,
        spread: BullCallSpread,
        debit_per_spread: float,
        equity: float,
    ) -> int:
        """Compute integer contracts. See module docstring for math."""
        if equity <= 0.0:
            return 0
        if debit_per_spread <= 0.0:
            return 0
        risk_dollars = float(equity) * self.risk_per_trade_pct
        max_loss_per_spread = (
            float(debit_per_spread) * spread.long_leg.option_multiplier()
        )
        if max_loss_per_spread <= 0.0:
            return 0
        return max(0, int(risk_dollars // max_loss_per_spread))
