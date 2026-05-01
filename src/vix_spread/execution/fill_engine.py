from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Literal

from vix_spread.products.spread import BullCallSpread

from .fill_modes import FillMode
from .quote import OptionQuote

if TYPE_CHECKING:
    from vix_spread.execution.liquidity_gates import LiquidityGates


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
    reason: Literal['no_bid_short', 'stale_quote', 'gate_fail',
                    'locked', 'crossed', 'tick_invalid', 'session_closed']
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
    ) -> 'ExecutedFill | RejectedOrder':
        """Returns ExecutedFill or RejectedOrder. NEVER consumes a
        TheoreticalPrice — Black-76 fair values are diagnostics, not fills.
        Both leg arguments must be OptionQuote instances; anything else
        (notably TheoreticalPrice, which carries is_executable=False) raises
        TypeError at the entry of the function."""
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
        ...
