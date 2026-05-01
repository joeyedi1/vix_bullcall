from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vix_spread.pricing.forward_selector import Forward


@dataclass(frozen=True)
class TheoreticalPrice:
    value: float
    delta: float
    gamma: float
    vega: float
    theta: float
    forward_used: 'Forward'
    iv_used: float
    T_minutes: float
    is_executable: bool = False  # SENTINEL — fill engine rejects on this
