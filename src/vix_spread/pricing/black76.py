from datetime import datetime
from typing import TYPE_CHECKING

from vix_spread.products.base import Product
from vix_spread.products.spread import BullCallSpread

if TYPE_CHECKING:
    from .forward_selector import Forward
    from .theoretical import TheoreticalPrice


class Black76Pricer:
    """Black-76 European call/put pricer for VIX-family options.

    NEVER prices from spot VIX. ALWAYS uses a Forward from ForwardSelector.
    NEVER applies a second convexity adjustment to an observed VX future —
    market convexity is already embedded in that price.

    Returns TheoreticalPrice objects with is_executable=False. The fill
    engine rejects these objects at the type level — they are diagnostics
    and edge-bleed inputs, never P&L.
    """

    def price(
        self,
        product: Product,
        forward: 'Forward',
        leg_iv: float,
        as_of: datetime,
        risk_free_rate: float,
    ) -> 'TheoreticalPrice':
        """Returns Black-76 fair value + Greeks tagged with selection_method
        and minute-level T."""
        ...

    def price_spread(
        self,
        spread: BullCallSpread,
        forward: 'Forward',
        iv_long: float,
        iv_short: float,
        as_of: datetime,
        risk_free_rate: float,
    ) -> 'TheoreticalSpreadPrice':
        """Strike-specific IV per leg is REQUIRED. Raises FlatVolError if
        iv_long == iv_short on a non-zero-width spread (validation-memo
        VVIX-as-leg-IV defect)."""
        ...
