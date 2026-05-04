from dataclasses import dataclass
from typing import TYPE_CHECKING

from .base import Product
from .vx_future import VXFuture

if TYPE_CHECKING:
    from vix_spread.data.snapshot import SettlementMarket


@dataclass
class VXFutureOption(Product):
    """CFE option on VX futures. European. Physically settled into ONE VX future.
    Underlying VX futures contract has $1000 multiplier.

    `deliverable_vx` is REQUIRED — the option's settlement payoff is
    the cash equivalent of the resulting position in this specific
    VX future, NOT a generic month or spot VIX. Making it a required
    field prevents the validation-memo conflation defect at construction
    time."""
    deliverable_vx: VXFuture = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if not isinstance(self.deliverable_vx, VXFuture):
            raise TypeError(
                f"VXFutureOption.deliverable_vx must be a VXFuture; "
                f"got {type(self.deliverable_vx).__name__}."
            )

    def option_multiplier(self) -> float:
        return 1000.0  # via underlying VX exposure

    def settlement_value(self, market: 'SettlementMarket') -> float:
        """Cash equivalent of the resulting VX futures position at exercise.

        Payoff per VX point = max(VX_settle − K, 0) for calls,
                              max(K − VX_settle, 0) for puts.
        Multiplied by the $1000 VX multiplier (carried via
        option_multiplier on this product).

        VX_settle is the deliverable VX future's final settlement price —
        do NOT cash-settle this product like a VIXIndexOption (validation
        memo §12.7)."""
        vx_settle = market.vx_settle_for(self.deliverable_vx.settlement_date)
        if self.right == 'call':
            intrinsic = max(vx_settle - self.strike, 0.0)
        elif self.right == 'put':
            intrinsic = max(self.strike - vx_settle, 0.0)
        else:
            raise ValueError(
                f"Unknown right: {self.right!r} (expected 'call' or 'put')."
            )
        return intrinsic * self.option_multiplier()

    def hedge_ratio_to_vx(self, delta_b76: float) -> float:
        """Already on VX-future-equivalent footing."""
        return delta_b76
