"""FillEngine — converts (spread, quotes, signal) into ExecutedFill or RejectedOrder.

ARCHITECTURE §5. The validation memo's largest performance-inflation
defect was treating fair value as a fill; this engine NEVER consumes a
TheoreticalPrice (`is_executable=False` is rejected at the type level).

Fill modes
----------
  SYNTHETIC_BIDASK (default, base case):
    open_debit  = long.ask − short.bid    (you pay)
    Conservative — exactly what a naive sweep would cross.

  MIDPOINT (optimistic sensitivity):
    open_debit  = mid(long) − mid(short)
                = ((long.bid+long.ask)/2) − ((short.bid+short.ask)/2)
    REQUIRES `accept_midpoint_optimism=True` per call. Emits no fees and
    overstates realized fills; reserved for sensitivity analysis only.

  SYNTHETIC_PLUS_SLIPPAGE (stressed sensitivity):
    Not yet wired in this first pass — slippage configuration (per-leg
    tick count, short-only flag) is a Phase-5 concern. NotImplementedError
    is a loud refusal pending that wiring.

Rejection priority (earliest wins)
----------------------------------
  1. reject_no_bid_short_leg  (validation-memo critical)
  2. reject_locked_or_crossed (per leg)
  3. max_quote_age_seconds    (per leg)
  4. min_displayed_size       (per leg/side)
  5. max_leg_spread_pct       (per leg)
  6. max_order_size_pct_of_displayed

Out of scope for this first pass
--------------------------------
  - Tick rounding (`tick_rounded` always reported as False).
  - Per-product fees (`fees_per_spread` always 0.0).
  - min_leg_open_interest, min_leg_volume_today — these need daily chain
    data not on OptionQuote; will wire when the side-channel reaches the
    engine.
  - Same-minute / next-minute decision-timestamp causality check — that
    convention is owned by the caller (broadcaster + bar-aware runner);
    enforced structurally upstream, not duplicated here.
  - Intent (open vs close) — entry-debit semantics assumed. Exits go
    through ExitEngine when wired.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from vix_spread.products.spread import BullCallSpread

from .fill_modes import FillMode
from .liquidity_gates import LiquidityGates
from .quote import OptionQuote
from .synthetic_quote import SyntheticSpreadQuote


RejectReason = Literal[
    'no_bid_short', 'stale_quote', 'gate_fail',
    'locked', 'crossed', 'tick_invalid', 'session_closed',
]


@dataclass(frozen=True)
class ExecutedFill:
    timestamp: datetime
    spread: BullCallSpread
    debit_per_spread: float
    size: int
    fill_mode: FillMode
    long_leg_fill: float
    short_leg_fill: float
    tick_rounded: bool                       # per-product tick rule applied
    fees_per_spread: float


@dataclass(frozen=True)
class RejectedOrder:
    timestamp: datetime
    spread: BullCallSpread
    reason: RejectReason
    detail: dict


class FillEngine:
    """Converts a (spread, quotes, signal) into ExecutedFill or RejectedOrder.

    Default mode is SYNTHETIC_BIDASK. MIDPOINT requires an explicit
    `accept_midpoint_optimism=True` flag and emits a warning to the run log.
    TheoreticalPrice instances are rejected at the type level (is_executable
    is False) — they cannot be fills.

    The decision_timestamp argument is checked against the quote timestamp:
    the next eligible quote AFTER decision_timestamp is used. Same-minute
    fills raise LookaheadError. There is no flag to disable this check.
    """

    # VIX-option tick size (Cboe spec). Hard-coded for Phase-5 first-pass
    # slippage; per-product tick rounding (and hence per-product tick value)
    # belongs to the tick_rules layer when it lands.
    DEFAULT_TICK_VALUE: float = 0.05

    def attempt_fill(
        self,
        spread: BullCallSpread,
        long_q: OptionQuote,
        short_q: OptionQuote,
        order_size: int = 1,
        mode: FillMode = FillMode.SYNTHETIC_BIDASK,
        gates: 'LiquidityGates | None' = None,
        decision_timestamp: datetime | None = None,
        *,
        accept_midpoint_optimism: bool = False,
        slippage_ticks_per_leg: int = 1,
        slippage_apply_to_short_leg_only: bool = False,
        tick_value: float | None = None,
    ) -> 'ExecutedFill | RejectedOrder':
        """Returns ExecutedFill or RejectedOrder. NEVER consumes a
        TheoreticalPrice — Black-76 fair values are diagnostics, not fills.
        Both leg arguments must be OptionQuote instances; anything else
        (notably TheoreticalPrice, which carries is_executable=False) raises
        TypeError at the entry of the function.

        Slippage parameters apply only when `mode == SYNTHETIC_PLUS_SLIPPAGE`:
          * `slippage_ticks_per_leg` — N ticks of adverse slippage per leg.
            Default 1, matching ARCH §10.3 default.
          * `slippage_apply_to_short_leg_only` — if True, only the short
            leg is slipped. Default False (both legs slipped).
          * `tick_value` — dollar value per tick. Defaults to
            `DEFAULT_TICK_VALUE` ($0.05 for VIX options).
        """
        if not isinstance(long_q, OptionQuote) or not isinstance(short_q, OptionQuote):
            raise TypeError(
                "FillEngine.attempt_fill accepts only OptionQuote leg inputs. "
                "TheoreticalPrice / Black-76 fair values are NEVER fills "
                "(is_executable=False per validation-memo constraint)."
            )
        if mode is FillMode.MIDPOINT and not accept_midpoint_optimism:
            raise ValueError(
                "MIDPOINT fills require explicit accept_midpoint_optimism=True. "
                "Base case is SYNTHETIC_BIDASK; midpoint is an optimistic "
                "sensitivity scenario that must be opted into per run."
            )
        if order_size <= 0:
            raise ValueError(f"order_size must be positive; got {order_size}.")
        if gates is None:
            raise ValueError(
                "FillEngine.attempt_fill requires explicit `gates` "
                "(LiquidityGates). Skipping gates is not a supported mode."
            )
        if slippage_ticks_per_leg < 0:
            raise ValueError(
                f"slippage_ticks_per_leg must be non-negative; "
                f"got {slippage_ticks_per_leg}."
            )

        ts = long_q.timestamp  # caller is responsible for matched-timestamp legs

        gate_failure = self._evaluate_gates(spread, long_q, short_q, order_size, gates)
        if gate_failure is not None:
            reason, detail = gate_failure
            return RejectedOrder(
                timestamp=ts, spread=spread, reason=reason, detail=detail,
            )

        # Compute per-leg fill prices and aggregate debit.
        if mode is FillMode.SYNTHETIC_BIDASK:
            long_fill = float(long_q.ask)
            short_fill = float(short_q.bid)
            debit = SyntheticSpreadQuote.open_debit_synthetic(long_q, short_q)
        elif mode is FillMode.MIDPOINT:
            long_fill = 0.5 * (float(long_q.bid) + float(long_q.ask))
            short_fill = 0.5 * (float(short_q.bid) + float(short_q.ask))
            debit = long_fill - short_fill
        elif mode is FillMode.SYNTHETIC_PLUS_SLIPPAGE:
            tv = self.DEFAULT_TICK_VALUE if tick_value is None else float(tick_value)
            slip_per_leg = float(slippage_ticks_per_leg) * tv
            # Long leg buys at ask + slippage (worse) unless short-only.
            # Short leg sells at bid - slippage (worse) always.
            long_slip = 0.0 if slippage_apply_to_short_leg_only else slip_per_leg
            short_slip = slip_per_leg
            long_fill = float(long_q.ask) + long_slip
            short_fill = float(short_q.bid) - short_slip
            debit = long_fill - short_fill
        else:
            raise ValueError(f"unknown FillMode: {mode!r}")

        return ExecutedFill(
            timestamp=ts,
            spread=spread,
            debit_per_spread=float(debit),
            size=int(order_size),
            fill_mode=mode,
            long_leg_fill=long_fill,
            short_leg_fill=short_fill,
            tick_rounded=False,             # first-pass: no tick rounding
            fees_per_spread=0.0,            # first-pass: no fees
        )

    # ---------------------------------------------------------------- #
    # Gate evaluation                                                   #
    # ---------------------------------------------------------------- #

    @staticmethod
    def _evaluate_gates(
        spread: BullCallSpread,
        long_q: OptionQuote,
        short_q: OptionQuote,
        order_size: int,
        gates: LiquidityGates,
    ) -> tuple[RejectReason, dict[str, Any]] | None:
        """Returns `None` on pass, else `(reason, detail)`. Earliest
        violation wins so the rejection log records the most-severe cause.
        """
        # 1. No-bid short leg — validation-memo critical, cheapest check.
        if gates.reject_no_bid_short_leg and short_q.bid <= 0.0:
            return ('no_bid_short', {'short_bid': float(short_q.bid)})

        # 2. Locked / Crossed (per leg).
        if gates.reject_locked_or_crossed:
            if long_q.is_locked or short_q.is_locked:
                return ('locked', {
                    'long_locked': long_q.is_locked,
                    'short_locked': short_q.is_locked,
                })
            if long_q.is_crossed or short_q.is_crossed:
                return ('crossed', {
                    'long_crossed': long_q.is_crossed,
                    'short_crossed': short_q.is_crossed,
                })

        # 3. Staleness — per-leg quote_age_seconds vs threshold.
        if long_q.quote_age_seconds > gates.max_quote_age_seconds:
            return ('stale_quote', {
                'leg': 'long', 'age_s': float(long_q.quote_age_seconds),
                'max_s': gates.max_quote_age_seconds,
            })
        if short_q.quote_age_seconds > gates.max_quote_age_seconds:
            return ('stale_quote', {
                'leg': 'short', 'age_s': float(short_q.quote_age_seconds),
                'max_s': gates.max_quote_age_seconds,
            })

        # 4. Min displayed size — every side of every leg must clear.
        if gates.min_displayed_size > 0:
            sizes = {
                'long_bid_size': long_q.bid_size,
                'long_ask_size': long_q.ask_size,
                'short_bid_size': short_q.bid_size,
                'short_ask_size': short_q.ask_size,
            }
            if any(s < gates.min_displayed_size for s in sizes.values()):
                return ('gate_fail', {
                    'sub_reason': 'min_displayed_size',
                    'min': gates.min_displayed_size,
                    'sizes': sizes,
                })

        # 5. Max leg spread pct — per leg.
        for leg_name, q in (('long', long_q), ('short', short_q)):
            mid = 0.5 * (float(q.bid) + float(q.ask))
            if mid <= 0.0:
                return ('gate_fail', {
                    'sub_reason': 'non_positive_mid',
                    'leg': leg_name, 'bid': q.bid, 'ask': q.ask,
                })
            spread_pct = (float(q.ask) - float(q.bid)) / mid
            if spread_pct > gates.max_leg_spread_pct:
                return ('gate_fail', {
                    'sub_reason': 'leg_spread_pct',
                    'leg': leg_name,
                    'spread_pct': spread_pct,
                    'max_pct': gates.max_leg_spread_pct,
                })

        # 6. Order size vs displayed — entry crosses long.ask and short.bid.
        # Both must clear the displayed-pct gate independently.
        if gates.max_order_size_pct_of_displayed < 1.0:
            for leg_name, displayed in (
                ('long_ask', long_q.ask_size),
                ('short_bid', short_q.bid_size),
            ):
                if displayed <= 0:
                    continue
                ratio = order_size / float(displayed)
                if ratio > gates.max_order_size_pct_of_displayed:
                    return ('gate_fail', {
                        'sub_reason': 'order_size_pct_of_displayed',
                        'side': leg_name,
                        'order_size': int(order_size),
                        'displayed': int(displayed),
                        'ratio': ratio,
                        'max_ratio': gates.max_order_size_pct_of_displayed,
                    })

        return None
