from dataclasses import dataclass

from .base import Product


@dataclass(frozen=True)
class BullCallSpread:
    """Construction-time enforcement: both legs MUST be the same product type
    and MUST share the same expiry/settlement event. Mixing a VIXIndexOption
    long leg with a VXFutureOption short leg raises TypeError at __post_init__."""
    long_leg: Product   # lower strike call
    short_leg: Product  # higher strike call

    def __post_init__(self) -> None:
        if type(self.long_leg) is not type(self.short_leg):
            raise TypeError("Spread legs must be the same Product subclass.")
        if self.long_leg.settlement_event != self.short_leg.settlement_event:
            raise ValueError("Spread legs must share settlement_event.")
        if self.long_leg.strike >= self.short_leg.strike:
            raise ValueError("Bull call: long strike must be below short strike.")