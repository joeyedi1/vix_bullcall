"""LegIVProvider — strike-specific implied volatility for option-spread legs.

ARCHITECTURE §4.3. Two phase-1 implementations:

  * `ChainIVProvider` (this module): vendor `IVOL_LAST` from the option-chain
    row matching `(date, expiry, right, strike)`. When `IVOL_LAST` is NaN
    or 0 (verified ~17% of cells overall, 18-45% on active back-months,
    diagnostic 2026-05-07), falls back to Black-76 inversion of the
    chain-row midpoint price `(PX_BID + PX_ASK) / 2`. If BOTH paths fail,
    `LegIVResolutionError` is raised — never NaN propagation.

  * `SurfaceIVProvider` (deferred to Phase 7): calibrates SVI / SABR
    walk-forward and interpolates.

Substituting a flat-vol index (VVIX) is FORBIDDEN per the validation memo;
the spread pricer's `iv_long == iv_short` guard (`Black76Pricer.price_spread`)
is the type-system catch.

LegIV vs raw float
------------------
The ABC returns a `LegIV` object (not a float) so the audit log can record
HOW each IV was obtained. Backtest reporting (ARCHITECTURE §8.2) bins
fills by `LegIVSource` distribution — a backtest where 60% of legs were
B76-inverted from a thin midpoint is a different model-risk profile from
one where 95% used vendor IV. Same numerical value, different model risk.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

import pandas as pd
from scipy.optimize import brentq

from vix_spread.products.base import Product
from vix_spread.utils.errors import LegIVResolutionError

if TYPE_CHECKING:
    from .black76 import Black76Pricer
    from .forward_selector import Forward


# brentq bracket: 0.1% to 500% vol covers any reasonable VIX-option IV
# (real-world range typically 20-200%). Outside this bracket the inversion
# is refused — that signal means the input mid is outside Black-76's
# reachable range (e.g. mid below intrinsic, locked/crossed quote).
_IV_BRACKET = (1e-3, 5.0)
_IV_TOL = 1e-6


class LegIVSource(Enum):
    """How a `LegIV` was obtained — used for backtest model-risk audit."""
    VENDOR = "vendor"
    B76_INVERTED = "b76_inverted"


@dataclass(frozen=True)
class LegIV:
    """Resolved per-leg implied volatility, tagged with provenance.

    `value` is fractional (1.0 = 100% vol) — this is what `Black76Pricer.price`
    expects. Bloomberg `IVOL_LAST` arrives in percentage points and is
    converted by the provider; B76-inverted values are already fractional.

    `fallback_target_mid` is populated only when `source == B76_INVERTED`
    so the audit layer can record the chain-row midpoint that drove each
    inverted IV.
    """
    value: float
    source: LegIVSource
    fallback_target_mid: float | None = None


class LegIVProvider(ABC):
    """ABC for leg-IV providers. Implementations vary in how they obtain
    the IV (vendor lookup, Black-76 invert from quotes, calibrated surface).
    """

    @abstractmethod
    def get(
        self,
        *,
        product: Product,
        as_of: datetime,
        **kwargs,
    ) -> LegIV:
        """Return the implied vol for this exact (strike, expiry, right)
        at `as_of`. `ChainIVProvider` requires `forward: Forward` and
        `risk_free_rate: float` keyword args for its B76-invert fallback;
        `SurfaceIVProvider` may ignore them.
        """


class ChainIVProvider(LegIVProvider):
    """Phase-1 vendor-then-invert IV provider.

    Construction
    ------------
    `chain_data`: pivoted DataFrame indexed on
    `(date, expiry_date, right, strike)` with columns
    `[IVOL_LAST, PX_BID, PX_ASK]`.
      - `date`: a `datetime.date` matching `as_of.date()` at `get` time.
      - `expiry_date`: a `datetime.date` matching `product.expiry.date()`.
      - `right`: a single character `'C'` or `'P'`.
      - `strike`: float matching `product.strike`.
    Caller (DataProcessor downstream) is responsible for the pivot —
    this provider is a lookup engine, not a data transformer.

    `pricer`: a `Black76Pricer` instance used for the fallback inversion.
    The provider does NOT cache prices — every B76-invert triggers a
    fresh root-find at brentq tolerance.

    Resolution order
    ----------------
    1. Vendor `IVOL_LAST` (if not NaN and > 0): converted from percent to
       fractional and returned with `source=VENDOR`.
    2. Else compute `mid = (PX_BID + PX_ASK) / 2`; invert Black-76 to find
       `sigma` such that `pricer.price(...).value == mid`. Returns
       `source=B76_INVERTED, fallback_target_mid=mid`.
    3. Else raise `LegIVResolutionError`.

    Causal contract
    ---------------
    `get` looks up the chain row at `as_of.date()`. The provider does NOT
    validate that the chain DataFrame contains only data with timestamp
    `<= as_of` — that's the FeatureAvailability validator's job
    (ARCHITECTURE §6.4). Pass a vintage-causal slice in.
    """

    _REQUIRED_COLS = frozenset({"IVOL_LAST", "PX_BID", "PX_ASK"})
    _INDEX_NAMES = ("date", "expiry", "right", "strike")

    def __init__(self, chain_data: pd.DataFrame, pricer: 'Black76Pricer') -> None:
        missing = self._REQUIRED_COLS - set(chain_data.columns)
        if missing:
            raise ValueError(
                f"chain_data missing required columns: {sorted(missing)}. "
                f"Expected at minimum {sorted(self._REQUIRED_COLS)}."
            )
        if (
            not isinstance(chain_data.index, pd.MultiIndex)
            or chain_data.index.nlevels != 4
        ):
            raise ValueError(
                "chain_data must be MultiIndexed on "
                "(date, expiry, right, strike) — got "
                f"{type(chain_data.index).__name__} with "
                f"nlevels={getattr(chain_data.index, 'nlevels', 1)}."
            )
        self._chain = chain_data
        self._pricer = pricer

    def get(
        self,
        *,
        product: Product,
        as_of: datetime,
        forward: 'Forward | None' = None,
        risk_free_rate: float | None = None,
        **_,
    ) -> LegIV:
        if forward is None or risk_free_rate is None:
            raise ValueError(
                "ChainIVProvider.get requires `forward` and `risk_free_rate` "
                "keyword args (needed for the B76-invert fallback path)."
            )
        right_short = product.right[0].upper()
        if right_short not in ("C", "P"):
            raise ValueError(f"Unknown product.right: {product.right!r}")

        key = (
            as_of.date(),
            product.expiry.date(),
            right_short,
            float(product.strike),
        )
        try:
            row = self._chain.loc[key]
        except KeyError as exc:
            raise LegIVResolutionError(
                f"No chain row at key={key} for "
                f"product={product.contract_root} {product.right} "
                f"K={product.strike} exp={product.expiry.date()} "
                f"as_of={as_of.date()}."
            ) from exc

        # Path 1: vendor IVOL_LAST.
        vendor = row.get("IVOL_LAST", float("nan"))
        if pd.notna(vendor) and float(vendor) > 0.0:
            return LegIV(
                value=float(vendor) / 100.0,
                source=LegIVSource.VENDOR,
            )

        # Path 2: B76 invert from chain-row midpoint.
        bid = row.get("PX_BID", float("nan"))
        ask = row.get("PX_ASK", float("nan"))
        if pd.isna(bid) or pd.isna(ask):
            raise LegIVResolutionError(
                f"Both vendor IVOL_LAST and PX_BID/PX_ASK missing for "
                f"key={key}. Cannot resolve leg IV; refusing rather than "
                f"propagating NaN."
            )
        bid_f, ask_f = float(bid), float(ask)
        if bid_f <= 0.0 and ask_f <= 0.0:
            raise LegIVResolutionError(
                f"Both bid and ask non-positive for key={key} "
                f"(bid={bid_f}, ask={ask_f}); cannot compute a "
                f"meaningful midpoint for inversion."
            )
        mid = 0.5 * (bid_f + ask_f)
        try:
            iv = self._invert_b76(
                target_price=mid,
                product=product,
                forward=forward,
                as_of=as_of,
                risk_free_rate=risk_free_rate,
            )
        except (ValueError, RuntimeError) as exc:
            raise LegIVResolutionError(
                f"Black-76 midpoint inversion failed for key={key} "
                f"(mid={mid:.4f}): {exc}"
            ) from exc

        return LegIV(
            value=iv,
            source=LegIVSource.B76_INVERTED,
            fallback_target_mid=mid,
        )

    def _invert_b76(
        self,
        *,
        target_price: float,
        product: Product,
        forward: 'Forward',
        as_of: datetime,
        risk_free_rate: float,
    ) -> float:
        """Find sigma such that `pricer.price(...).value == target_price`.

        Raises `ValueError` (from brentq) if the bracket does not straddle
        zero — typically because `target_price` is below intrinsic or
        above the forward-discounted upper bound.
        """
        def f(iv: float) -> float:
            return self._pricer.price(
                product, forward, iv, as_of, risk_free_rate,
            ).value - target_price

        return float(brentq(f, _IV_BRACKET[0], _IV_BRACKET[1], xtol=_IV_TOL))
