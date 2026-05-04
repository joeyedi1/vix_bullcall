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

from dataclasses import dataclass
from datetime import date, datetime


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
