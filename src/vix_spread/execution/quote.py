from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class OptionQuote:
    """One minute-level NBBO snapshot for one option contract.
    Last-trade prices are EXPLICITLY excluded from fill logic — they are
    diagnostic only. The fill engine never reads `last_trade`."""
    timestamp: datetime
    contract_id: str
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    last_trade: float | None              # diagnostics ONLY, not fills
    last_trade_age_seconds: float | None
    is_locked: bool
    is_crossed: bool
    quote_age_seconds: float

    def is_stale(self, max_age_seconds: float) -> bool:
        return self.quote_age_seconds > max_age_seconds

    def is_no_bid(self) -> bool:
        return self.bid <= 0.0
