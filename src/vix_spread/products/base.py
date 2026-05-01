from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from vix_spread.data.snapshot import SettlementMarket


@dataclass
class Product(ABC):
    """Abstract base. All pricing/settlement/P&L code dispatches on subclass.
    Mixing string ticker checks for product type is FORBIDDEN — every
    multiplier, settlement rule, and tick rule is a method on this class."""
    contract_root: str
    expiry: datetime
    settlement_event: datetime
    strike: float
    right: Literal['call', 'put']

    @abstractmethod
    def settlement_value(self, market: 'SettlementMarket') -> float:
        """Cash payoff (VIXIndexOption: max(VRO − K, 0) × 100) or
        post-exercise position value (VXFutureOption: into one VX future)."""

    @abstractmethod
    def option_multiplier(self) -> float:
        """Dollar multiplier per option point. Hard-coded per product subclass."""

    @abstractmethod
    def hedge_ratio_to_vx(self, delta_b76: float) -> float:
        """Convert Black-76 delta to VX futures contracts for hedging.
        Required to be a product method — prevents the validation-memo
        'factor of 10' hedging error."""
