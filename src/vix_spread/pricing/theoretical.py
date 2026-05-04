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
    rho: float = 0.0             # Black-76 ∂C/∂r; defaulted for back-compat


@dataclass(frozen=True)
class TheoreticalSpreadPrice:
    """Two-leg spread theoretical value + per-leg breakdown.

    `value` is the net debit (long − short) for a bull call spread, so a
    positive number means the spread costs to enter. Aggregated Greeks
    are leg differences. `is_executable=False` carries the same anti-leak
    sentinel as TheoreticalPrice — these objects are diagnostics and
    edge-bleed inputs, never P&L."""
    value: float
    long_leg: TheoreticalPrice
    short_leg: TheoreticalPrice
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    is_executable: bool = False
