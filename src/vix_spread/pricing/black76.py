"""Black76Pricer — Black-76 European pricer for VIX-family options.

ARCHITECTURE §4.4. Always consumes a `Forward` from `ForwardSelector` —
never spot VIX (validation memo §4 / §12.2). Returns `TheoreticalPrice`
with `is_executable=False` so the FillEngine type-rejects them as fills.

Greek conventions (math defaults; reporting layers convert at display):
  * delta / gamma / theta / rho = ∂value/∂(F, F², t, r)
  * vega = ∂value/∂σ, where σ is in absolute terms (1.0 = 100% vol)
  * theta is per YEAR of calendar time (divide by 365 for per-day)

Closed-form Black-76 European call/put on a forward `F`:

  d1 = [ln(F/K) + ½σ²T] / (σ√T)
  d2 = d1 - σ√T
  df = e^(-rT)

  call  = df · (F·N(d1) − K·N(d2))
  put   = df · (K·N(-d2) − F·N(-d1))

  delta_call = df · N(d1)
  delta_put  = -df · N(-d1)             ( = df · (N(d1) − 1) )
  gamma      = df · φ(d1) / (F · σ · √T)
  vega       = df · F · φ(d1) · √T
  theta      = r · price − df · F · φ(d1) · σ / (2 · √T)
  rho        = -T · price
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import TYPE_CHECKING

from scipy.stats import norm

from vix_spread.products.base import Product
from vix_spread.products.spread import BullCallSpread
from vix_spread.utils.errors import FlatVolError

from .theoretical import TheoreticalPrice, TheoreticalSpreadPrice
from .time_to_expiry import minutes_to_settlement

if TYPE_CHECKING:
    from .forward_selector import Forward


_MINUTES_PER_YEAR = 525_600.0


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
    ) -> TheoreticalPrice:
        """Returns Black-76 fair value + 5 Greeks for a single leg.

        Greeks are tagged with the `Forward` actually used (carrying its
        `selection_method` for audit) and with the minute-level T at
        which the price was struck.
        """
        T_year = minutes_to_settlement(as_of, product.settlement_event)
        F = float(forward.value)
        K = float(product.strike)
        sigma = float(leg_iv)
        r = float(risk_free_rate)

        if F <= 0 or K <= 0:
            raise ValueError(
                f"Forward and strike must be positive; got F={F}, K={K}."
            )
        if sigma <= 0:
            raise ValueError(f"Implied vol must be positive; got sigma={sigma}.")

        sqrt_T = math.sqrt(T_year)
        d1 = (math.log(F / K) + 0.5 * sigma * sigma * T_year) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        df = math.exp(-r * T_year)
        pdf_d1 = float(norm.pdf(d1))

        if product.right == 'call':
            price = df * (F * float(norm.cdf(d1)) - K * float(norm.cdf(d2)))
            delta = df * float(norm.cdf(d1))
        elif product.right == 'put':
            price = df * (K * float(norm.cdf(-d2)) - F * float(norm.cdf(-d1)))
            delta = -df * float(norm.cdf(-d1))
        else:
            raise ValueError(
                f"Unknown product.right: {product.right!r} "
                f"(expected 'call' or 'put')."
            )

        gamma = df * pdf_d1 / (F * sigma * sqrt_T)
        vega = df * F * pdf_d1 * sqrt_T
        theta = r * price - df * F * pdf_d1 * sigma / (2.0 * sqrt_T)
        rho = -T_year * price

        return TheoreticalPrice(
            value=price,
            delta=delta,
            gamma=gamma,
            vega=vega,
            theta=theta,
            forward_used=forward,
            iv_used=sigma,
            T_minutes=T_year * _MINUTES_PER_YEAR,
            is_executable=False,
            rho=rho,
        )

    def price_spread(
        self,
        spread: BullCallSpread,
        forward: 'Forward',
        iv_long: float,
        iv_short: float,
        as_of: datetime,
        risk_free_rate: float,
    ) -> TheoreticalSpreadPrice:
        """Strike-specific IV per leg is REQUIRED.

        Raises FlatVolError if `iv_long == iv_short` — the validation-memo
        defect signature of substituting VVIX (a flat-vol index) for the
        strike-specific IVs the spread depends on. Bull call spreads have
        non-zero width by `BullCallSpread.__post_init__` (long.strike <
        short.strike), so the flat-vol check applies unconditionally.

        Returns TheoreticalSpreadPrice with `value = long − short` (net
        debit), per-leg breakdown, and aggregated Greeks. is_executable
        is False — the FillEngine never accepts these as fills."""
        if iv_long == iv_short:
            raise FlatVolError(
                f"iv_long ({iv_long}) == iv_short ({iv_short}) on a "
                f"non-zero-width spread (strikes "
                f"{spread.long_leg.strike} → {spread.short_leg.strike}). "
                f"Pass strike-specific IVs from the chain — never a "
                f"flat-surface index like VVIX."
            )
        long_price = self.price(
            spread.long_leg, forward, iv_long, as_of, risk_free_rate,
        )
        short_price = self.price(
            spread.short_leg, forward, iv_short, as_of, risk_free_rate,
        )
        return TheoreticalSpreadPrice(
            value=long_price.value - short_price.value,
            long_leg=long_price,
            short_leg=short_price,
            delta=long_price.delta - short_price.delta,
            gamma=long_price.gamma - short_price.gamma,
            vega=long_price.vega - short_price.vega,
            theta=long_price.theta - short_price.theta,
            rho=long_price.rho - short_price.rho,
            is_executable=False,
        )
