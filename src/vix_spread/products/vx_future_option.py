from dataclasses import dataclass
from typing import TYPE_CHECKING

from .base import Product

if TYPE_CHECKING:
    from vix_spread.data.snapshot import SettlementMarket
    from vix_spread.products.vx_future import VXFuture


@dataclass
class VXFutureOption(Product):
    """CFE option on VX futures. European. Physically settled into ONE VX future.
    Underlying VX futures contract has $1000 multiplier."""
    deliverable_vx: 'VXFuture' = None

    def option_multiplier(self) -> float:
        return 1000.0  # via underlying VX exposure

    def settlement_value(self, market: 'SettlementMarket') -> float:
        """Cash equivalent of the resulting VX futures position at exercise.
        Do NOT cash-settle this product like a VIXIndexOption."""
        ...

    def hedge_ratio_to_vx(self, delta_b76: float) -> float:
        """Already on VX-future-equivalent footing."""
        return delta_b76
