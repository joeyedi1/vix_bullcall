"""ExitEngine — closes open spread positions per the declared `ExitPolicy`.

ARCHITECTURE §5.6. Two paths:

  * `FORCED_TUESDAY_LIQUIDATION` — cross the close-credit synthetic
    `(long.bid − short.ask)` against the per-minute NBBO. Same gate
    contract as `FillEngine` but with reversed leg priorities — on a
    close we SELL the long bid and BUY the short ask, so the no-bid
    guard moves to the long leg and the no-ask guard moves to the
    short leg. If gates trip → `FailedExit` (a first-class outcome,
    not a silent fill at theoretical).

  * `HOLD_TO_SETTLEMENT` — consume the actual VRO print from the
    `SettlementMarket` and compute the cash payoff via
    `Product.settlement_value(market)`. Validation memo §12.7: VRO is
    the actual SOQ print, NEVER spot VIX close, NEVER theoretical
    Black-76 value, NEVER Tuesday VX-future close.

Sign convention on the `ExecutedFill` returned by Path A:
  - `debit_per_spread = -close_credit` (negative for the typical credit
    close). This keeps the entry-flavoured field name on the dataclass
    while preserving signed-cash-flow accounting: `total_pnl_per_spread
    = -(entry_debit + exit_debit) × multiplier × size`. Documented
    loudly because the negative sign is non-obvious.

Out of scope for this first pass
--------------------------------
  - Tick rounding / fees on the close (mirrors entry: `tick_rounded=False`,
    `fees_per_spread=0.0`).
  - SYNTHETIC_PLUS_SLIPPAGE / MIDPOINT close modes — exits use only
    SYNTHETIC_BIDASK in the conservative-base case; sensitivity modes
    can be added when the report layer needs them.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Literal

from vix_spread.data.snapshot import SettlementMarket, VIXSnapshot
from vix_spread.products.base import Product
from vix_spread.products.spread import BullCallSpread

from .exit_policy import ExitPolicy
from .fill_engine import ExecutedFill
from .fill_modes import FillMode
from .liquidity_gates import LiquidityGates
from .quote import OptionQuote
from .synthetic_quote import SyntheticSpreadQuote


FailedExitReason = Literal[
    'no_bid_long', 'no_ask_short', 'no_quote',
    'stale_quote', 'locked', 'crossed', 'gate_fail',
]


@dataclass(frozen=True)
class OpenPosition:
    """A live spread position passed to the ExitEngine to close.

    `entry_fill` is audit-only — it is NOT used to compute the close P&L
    (the close is its own fill / settlement event). Carry it on the
    position so the trade log can pair entry and exit timestamps.
    """
    spread: BullCallSpread
    size: int
    entry_fill: ExecutedFill | None = None


@dataclass(frozen=True)
class SettlementOutcome:
    """Result of `HOLD_TO_SETTLEMENT` — actual VRO-driven cash payoff.

    Per-leg payoffs are the dollar amounts already-multiplier-scaled
    (Product.settlement_value returns intrinsic × multiplier). The net
    per spread is `long_leg_payoff − short_leg_payoff`; total realized
    P&L over the spread = `net_payoff_per_spread × size`.
    """
    timestamp: datetime
    spread: BullCallSpread
    size: int
    long_leg_payoff: float
    short_leg_payoff: float
    net_payoff_per_spread: float
    vro_print: float


@dataclass(frozen=True)
class FailedExit:
    """Result of `FORCED_TUESDAY_LIQUIDATION` when gates trip.

    First-class outcome per ARCH §5.6: the report binds these alongside
    P&L so a backtest with a high failed-exit rate is observably risky,
    not silently filled at theoretical.
    """
    timestamp: datetime
    spread: BullCallSpread
    size: int
    reason: FailedExitReason
    detail: dict


class ExitEngine:
    """Closes spread positions per ExitPolicy."""

    def __init__(
        self,
        gates: LiquidityGates,
        settlement_market: SettlementMarket | None = None,
        contract_id_for: Callable[[Product], str] | None = None,
    ) -> None:
        from vix_spread.data.vix_index_options import vix_option_active_contract_id
        self.gates = gates
        self.settlement_market = settlement_market
        self.contract_id_for = contract_id_for or vix_option_active_contract_id

    def execute_exit(
        self,
        position: OpenPosition,
        market: VIXSnapshot,
        policy: ExitPolicy,
    ) -> ExecutedFill | SettlementOutcome | FailedExit:
        """Close `position` per `policy`. `market` is consumed for quotes
        on FORCED_TUESDAY_LIQUIDATION; ignored on HOLD_TO_SETTLEMENT (the
        constructor-bound `settlement_market` carries the VRO prints)."""
        if policy is ExitPolicy.FORCED_TUESDAY_LIQUIDATION:
            return self._force_tuesday(position, market)
        if policy is ExitPolicy.HOLD_TO_SETTLEMENT:
            return self._hold_to_settlement(position)
        raise ValueError(f"unknown ExitPolicy: {policy!r}")

    # ---------------------------------------------------------------- #
    # Path A — forced Tuesday liquidation                              #
    # ---------------------------------------------------------------- #

    def _force_tuesday(
        self, position: OpenPosition, market: VIXSnapshot,
    ) -> 'ExecutedFill | FailedExit':
        long_id = self.contract_id_for(position.spread.long_leg)
        short_id = self.contract_id_for(position.spread.short_leg)
        long_q = market.options_quotes.get(long_id)
        short_q = market.options_quotes.get(short_id)

        if long_q is None or short_q is None:
            missing = [
                name for name, q in (("long", long_q), ("short", short_q))
                if q is None
            ]
            return FailedExit(
                timestamp=market.timestamp,
                spread=position.spread, size=position.size,
                reason='no_quote',
                detail={
                    'missing_legs': missing,
                    'long_id': long_id, 'short_id': short_id,
                },
            )

        gate_failure = self._evaluate_close_gates(
            position.spread, long_q, short_q, position.size, self.gates,
        )
        if gate_failure is not None:
            reason, detail = gate_failure
            return FailedExit(
                timestamp=market.timestamp,
                spread=position.spread, size=position.size,
                reason=reason, detail=detail,
            )

        # Close: SELL long at long.bid, BUY short at short.ask.
        # close_credit = long.bid - short.ask (positive in normal close).
        close_credit = SyntheticSpreadQuote.close_credit_synthetic(long_q, short_q)
        return ExecutedFill(
            timestamp=market.timestamp,
            spread=position.spread,
            debit_per_spread=-float(close_credit),    # NEGATIVE for credit close
            size=position.size,
            fill_mode=FillMode.SYNTHETIC_BIDASK,
            long_leg_fill=float(long_q.bid),
            short_leg_fill=float(short_q.ask),
            tick_rounded=False,
            fees_per_spread=0.0,
        )

    @staticmethod
    def _evaluate_close_gates(
        spread: BullCallSpread,
        long_q: OptionQuote,
        short_q: OptionQuote,
        size: int,
        gates: LiquidityGates,
    ) -> tuple[FailedExitReason, dict[str, Any]] | None:
        """Mirror of FillEngine._evaluate_gates with leg priorities flipped
        for the close side: we SELL the long bid (so long.bid > 0 critical)
        and BUY the short ask (so short.ask > 0 critical).
        """
        # 1. No bid on long — we can't sell our long leg.
        if long_q.bid <= 0.0:
            return ('no_bid_long', {'long_bid': float(long_q.bid)})

        # 2. No ask on short — we can't buy back the short leg.
        if short_q.ask <= 0.0:
            return ('no_ask_short', {'short_ask': float(short_q.ask)})

        # 3. Locked / Crossed (per leg).
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

        # 4. Staleness — per-leg quote_age vs threshold.
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

        # 5. Min displayed size — every side of every leg must clear.
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
                    'min': gates.min_displayed_size, 'sizes': sizes,
                })

        # 6. Max leg spread pct — per leg.
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

        # 7. Order size vs displayed — close crosses long.bid_size (sell
        # the long) and short.ask_size (buy back the short).
        if gates.max_order_size_pct_of_displayed < 1.0:
            for label, displayed in (
                ('long_bid', long_q.bid_size),
                ('short_ask', short_q.ask_size),
            ):
                if displayed <= 0:
                    continue
                ratio = size / float(displayed)
                if ratio > gates.max_order_size_pct_of_displayed:
                    return ('gate_fail', {
                        'sub_reason': 'order_size_pct_of_displayed',
                        'side': label,
                        'order_size': int(size),
                        'displayed': int(displayed),
                        'ratio': ratio,
                        'max_ratio': gates.max_order_size_pct_of_displayed,
                    })

        return None

    # ---------------------------------------------------------------- #
    # Path B — hold to SOQ settlement                                  #
    # ---------------------------------------------------------------- #

    def _hold_to_settlement(self, position: OpenPosition) -> SettlementOutcome:
        if self.settlement_market is None:
            raise ValueError(
                "ExitEngine policy=HOLD_TO_SETTLEMENT requires "
                "`settlement_market` in the constructor."
            )
        spread = position.spread
        long_payoff = float(
            spread.long_leg.settlement_value(self.settlement_market)
        )
        short_payoff = float(
            spread.short_leg.settlement_value(self.settlement_market)
        )
        net_per_spread = long_payoff - short_payoff
        vro_print = float(self.settlement_market.vro_for(spread.long_leg.expiry))
        return SettlementOutcome(
            timestamp=spread.long_leg.settlement_event,
            spread=spread,
            size=position.size,
            long_leg_payoff=long_payoff,
            short_leg_payoff=short_payoff,
            net_payoff_per_spread=net_per_spread,
            vro_print=vro_print,
        )
