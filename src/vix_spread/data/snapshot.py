"""Market-snapshot dataclasses — minimal Phase-2 surface.

ARCHITECTURE §6.2. Two consumers in this phase:

  * `SettlementMarket` — read-only registry of settlement-event prints
    (VRO/SOQ for VIX index options, final settle for VX futures).
    Consumed by `Product.settlement_value(market)`.

  * `OptionsMarketSnapshot` — minimal placeholder. Currently carries
    only the VX futures curve (settlement_date → price) needed by
    `ForwardSelector`'s `settlement_date_match` branch. Will grow to
    hold the option chain (NBBO quotes, IVs) when the chain ingestion
    layer lands; PCP and surface-interpolated forward branches will
    consume those additions.

Date keys throughout: VRO is keyed by SOQ event date (matches
`option.expiry.date()`), VX settlement keyed by the contract's
settlement date. Using `date` rather than `datetime` avoids exact-equality
foot-guns on tz-aware lookups.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vix_spread.execution.quote import OptionQuote


@dataclass(frozen=True)
class SettlementMarket:
    """Read-only registry of settlement-event prices."""
    vro_prints: dict[date, float]
    vx_settle_prints: dict[date, float]

    def vro_for(self, expiry: datetime) -> float:
        """The VRO/SOQ print for the SOQ event that settles options
        with this expiry. KeyError if not available."""
        key = expiry.date()
        if key not in self.vro_prints:
            raise KeyError(f"No VRO print for expiry={expiry} (key={key}).")
        return float(self.vro_prints[key])

    def vx_settle_for(self, settlement_date: date) -> float:
        """Final settlement price of the VX futures contract identified
        by `settlement_date`. KeyError if not available."""
        if settlement_date not in self.vx_settle_prints:
            raise KeyError(
                f"No VX settle print for settlement_date={settlement_date}."
            )
        return float(self.vx_settle_prints[settlement_date])


@dataclass(frozen=True)
class OptionsMarketSnapshot:
    """Minimal options-market snapshot for Phase-2 forward selection.

    Phase-2 surface is just `vx_curve` (settlement_date → VX future price)
    consumed by `ForwardSelector` for the `settlement_date_match` branch.
    Later phases add the option chain and IV surface.
    """
    timestamp: datetime
    vx_curve: dict[date, float]


@dataclass(frozen=True)
class VIXSnapshot:
    """Per-decision market context bundle (ARCHITECTURE §6.2).

    Bundles everything `SpreadEvaluator.evaluate` needs at one moment:

      - `timestamp`: decision time (tz-aware UTC).
      - `vx_curve`: settlement_date → VX future price; consumed by
        `ForwardSelector.settlement_date_match`. Duck-typed compatible
        with `OptionsMarketSnapshot` so the existing selector accepts a
        VIXSnapshot directly.
      - `options_quotes`: contract_id → `OptionQuote`; consumed by the
        `FillEngine`. The contract_id form is whatever the per-minute
        NBBO panel exposes (Tuesday last-trade form for active VIX
        index options — see `vix_option_active_contract_id`).
      - `risk_free_rate`: float, fed to Black-76 discounting.
      - `vix_spot`: DIAGNOSTIC ONLY. Validation memo §12.2 / §4.1: spot
        VIX is FORBIDDEN as a Black-76 forward input. The field is here
        for reporting / regime overlays only; never for pricing.
    """
    timestamp: datetime
    vx_curve: dict[date, float]
    options_quotes: dict[str, "OptionQuote"] = field(default_factory=dict)
    risk_free_rate: float = 0.0
    vix_spot: float | None = None
