from dataclasses import dataclass
from typing import TYPE_CHECKING

from .base import Product

if TYPE_CHECKING:
    from vix_spread.data.snapshot import SettlementMarket


@dataclass
class VIXIndexOption(Product):
    """Cboe VIX index option. European. Cash-settled to SOQ/VRO. $100 multiplier.
    NEVER priced from spot VIX. Forward selected by ForwardSelector."""

    def option_multiplier(self) -> float:
        return 100.0

    def settlement_value(self, market: 'SettlementMarket') -> float:
        """Payoff = max(VRO − K, 0) × 100 for calls. VRO is the actual SOQ print
        for this expiry — NOT spot VIX close, NOT Tuesday VX future close,
        NOT theoretical Black-76 value."""
        ...

    def hedge_ratio_to_vx(self, delta_b76: float) -> float:
        """option dollar delta = 100 × Δ_B76; VX multiplier = 1000.
        Hedge = (100 × Δ_B76) / 1000 = 0.1 × Δ_B76 VX contracts per option."""
        return 0.1 * delta_b76
