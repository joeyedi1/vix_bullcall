"""SpreadSelector — picks (long_strike, short_strike, expiry) for a bull call spread.

ARCHITECTURE §7.1. The selector is the thin layer between regime/curve
signals and the executable spread. It NEVER produces a spread that the
FillEngine would reject for `no_bid_short` — the validity filter runs
HERE, pre-selection. That distinction is structural: a strategy whose
selector silently produces unfillable spreads would inflate the rejection
rate and bleed signal-to-fill latency in the backtest.

Selection rule (Phase-5 first-pass, configurable)
-------------------------------------------------
1. Index `market.options_quotes` by (expiry SOQ-Wed, strike) for calls.
2. Pick the nearest expiry whose DTE falls in `[dte_min, dte_max]` and
   that has at least two valid strikes.
3. Read the forward from `market.vx_curve[expiry]` — that IS the ATM
   reference (NOT spot VIX, per validation memo §12.2).
4. `long_target  = forward + long_offset`
   `long_eligible = {strike : ask > 0}`            (so we can buy the ask)
   `long_strike   = closest in long_eligible`
5. `short_target  = long_strike + short_offset`
   `short_eligible = {strike > long_strike : bid > 0}`  (so we can sell the bid)
   `short_strike   = closest in short_eligible`
6. Construct `BullCallSpread(long_leg, short_leg)`. Construction-time
   invariants (same product type, same settlement, K_long < K_short) are
   enforced by `BullCallSpread.__post_init__`.

Returns `None` when no spread is constructible (no eligible expiry, no
eligible long, no eligible short above long_strike, no forward in curve).
The caller's Strategy layer interprets `None` as "no entry this minute"
and reports it as such.

Out of scope for this first pass
--------------------------------
  - Signal-conditional strike rules (e.g. contrarian_tail picks ITM,
    breakout_momentum picks OTM). The signal is accepted on the
    interface so future variants can branch on `signal.hypothesis_tag`
    or `signal.curve_features` — currently unused.
  - Locked / crossed / staleness pre-filters (left to FillEngine —
    selecting around them would pre-empt the gate audit).
  - PCP / interpolated forward sources (ForwardSelector is not consulted
    here; we read the VX curve directly because settlement-date match
    is the only Phase-2 wired branch and the selector doesn't need
    forward-method audit).
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import TYPE_CHECKING

from vix_spread.data.expiry_calendar import vx_settlement_date
from vix_spread.data.vix_index_options import parse_vix_option_ticker
from vix_spread.execution.quote import OptionQuote
from vix_spread.products.spread import BullCallSpread
from vix_spread.products.vix_index_option import VIXIndexOption

if TYPE_CHECKING:
    from vix_spread.data.snapshot import VIXSnapshot
    from vix_spread.regime.base import RegimeSignal


class SpreadSelector:
    """Picks `(long_strike, short_strike, expiry)` for a `BullCallSpread`.

    Phase-5 first-pass implementation: forward-relative strike offsets,
    nearest-eligible-expiry within a DTE window, validity-filtered strikes.
    """

    def __init__(
        self,
        long_offset: float,
        short_offset: float,
        dte_min: int,
        dte_max: int,
    ) -> None:
        if long_offset < 0:
            raise ValueError(f"long_offset must be non-negative; got {long_offset}.")
        if short_offset <= 0:
            raise ValueError(
                f"short_offset must be positive (short strike above long); "
                f"got {short_offset}."
            )
        if dte_min < 0 or dte_max < dte_min:
            raise ValueError(
                f"DTE window invalid: dte_min={dte_min}, dte_max={dte_max}."
            )
        self.long_offset = float(long_offset)
        self.short_offset = float(short_offset)
        self.dte_min = int(dte_min)
        self.dte_max = int(dte_max)

    def select(
        self,
        market: "VIXSnapshot",
        signal: "RegimeSignal | None",
        as_of: datetime,
    ) -> BullCallSpread | None:
        """Return a `BullCallSpread` or `None` if no executable spread exists.

        `signal` is accepted on the interface for ARCH §7.1 compatibility
        and future signal-conditional rules; this first-pass selector
        does not branch on it.
        """
        by_expiry = self._index_calls_by_expiry(market.options_quotes)

        # Find nearest expiry in the DTE window with at least two strikes.
        as_of_date = as_of.date()
        eligible_expiries: list[date] = []
        for soq_wed, strikes_dict in by_expiry.items():
            dte = (soq_wed - as_of_date).days
            if (
                self.dte_min <= dte <= self.dte_max
                and len(strikes_dict) >= 2
                and soq_wed in market.vx_curve
            ):
                eligible_expiries.append(soq_wed)
        if not eligible_expiries:
            return None
        soq_wed = sorted(eligible_expiries)[0]

        forward = float(market.vx_curve[soq_wed])
        if forward <= 0:
            return None

        strike_quotes = by_expiry[soq_wed]

        # Long leg: ask > 0; closest strike to (forward + long_offset).
        long_target = forward + self.long_offset
        long_eligible = sorted(
            s for s, q in strike_quotes.items() if q.ask > 0
        )
        if not long_eligible:
            return None
        long_strike = min(long_eligible, key=lambda s: abs(s - long_target))

        # Short leg: bid > 0 AND strike > long_strike; closest to (long + short_offset).
        # The bid > 0 filter is the validation-memo critical guard against
        # selecting a short leg the FillEngine would reject as no_bid_short.
        short_target = long_strike + self.short_offset
        short_eligible = sorted(
            s for s, q in strike_quotes.items()
            if q.bid > 0 and s > long_strike
        )
        if not short_eligible:
            return None
        short_strike = min(short_eligible, key=lambda s: abs(s - short_target))

        # Construct the spread. expiry = settlement_event by VIX-option
        # convention (cash-settled to VRO at the SOQ Wednesday).
        expiry_dt = datetime.combine(
            soq_wed, time(14, 30), tzinfo=timezone.utc,
        )
        long_leg = VIXIndexOption(
            contract_root="VIX",
            expiry=expiry_dt,
            settlement_event=expiry_dt,
            strike=float(long_strike),
            right="call",
        )
        short_leg = VIXIndexOption(
            contract_root="VIX",
            expiry=expiry_dt,
            settlement_event=expiry_dt,
            strike=float(short_strike),
            right="call",
        )
        return BullCallSpread(long_leg=long_leg, short_leg=short_leg)

    @staticmethod
    def _index_calls_by_expiry(
        quotes: dict[str, OptionQuote],
    ) -> dict[date, dict[float, OptionQuote]]:
        """Index option quotes by `(SOQ_Wed_expiry, strike)` for calls.

        Tickers parse to EITHER Tue (active) OR Wed (settled) date forms.
        We always normalize to the SOQ Wednesday via the expiry calendar
        so downstream `vx_curve` and `Product.expiry.date()` lookups
        (which both key on the canonical SOQ Wed) succeed regardless of
        which form the ticker carried.
        """
        out: dict[date, dict[float, OptionQuote]] = {}
        for cid, q in quotes.items():
            try:
                parsed = parse_vix_option_ticker(cid)
            except ValueError:
                continue
            if parsed.right != "C":
                continue
            soq_wed = vx_settlement_date(
                parsed.expiry_date.year, parsed.expiry_date.month,
            )
            out.setdefault(soq_wed, {})[parsed.strike] = q
        return out
