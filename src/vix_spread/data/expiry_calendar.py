"""VX futures expiry calendar.

Maps a contract month <-> Cboe final settlement date <-> Bloomberg ticker.

Cboe VX final settlement rule (CFE Rulebook, Chapter VX):
    The final settlement date for a contract month is the Wednesday that is
    thirty days prior to the third Friday of the calendar month immediately
    following the contract month. If CFE is closed on that Wednesday, the
    final settlement date is the business day immediately preceding it.

The settlement date is the join key into the option chain (ARCHITECTURE
§6.1) and the input to ForwardSelector's settlement-date branch (§4.1).
Generic month numbers ("M1", "M2") are NOT a valid join key in the pricing
path — the regime layer (§3) is the only consumer of generic-month series.

Bloomberg ticker format
-----------------------
`UX{month_code}{year_last_two_digits} Index`, e.g. `UXM25 Index` = June 2025.

The 2-digit-year form is empirically the only stable form: Bloomberg's
single-digit aliases (e.g. `UXM5`) work for some active/recent contracts
but return `BAD_SEC: Unknown/Invalid security` for expired contracts
(e.g. `UXF5` for Jan 2025 once it has expired). 2-digit is unambiguous
across decades and resolves correctly for both active and historical
contracts. This is verified empirically against the Bloomberg
ReferenceDataRequest.

Holiday calendar
----------------
Default proxy is `pandas.tseries.holiday.USFederalHolidayCalendar`. Its
mid-month Wednesday observances (Juneteenth on Jun 19) match CFE for the
historical window of interest. Columbus Day and Veterans Day are included
in the federal calendar but never coincide with a VX settlement Wednesday
(settlement Wed always falls in days ~15-22 of the contract month). For
runs requiring exact CFE alignment, pass `holidays=` explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Final

from pandas.tseries.holiday import USFederalHolidayCalendar


MONTH_CODES: Final[dict[int, str]] = {
    1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M",
    7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z",
}
CODE_TO_MONTH: Final[dict[str, int]] = {v: k for k, v in MONTH_CODES.items()}


def _third_friday(year: int, month: int) -> date:
    first = date(year, month, 1)
    offset = (4 - first.weekday()) % 7  # Friday = weekday 4
    return first + timedelta(days=offset + 14)


def _roll_back_if_closed(d: date, holidays: set[date]) -> date:
    while d in holidays or d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


_HOLIDAY_CACHE: dict[int, set[date]] = {}


def _us_federal_holidays(year: int) -> set[date]:
    if year not in _HOLIDAY_CACHE:
        cal = USFederalHolidayCalendar()
        idx = cal.holidays(start=f"{year - 1}-12-01", end=f"{year + 1}-01-31")
        _HOLIDAY_CACHE[year] = {ts.date() for ts in idx}
    return _HOLIDAY_CACHE[year]


def vx_settlement_date(
    year: int,
    month: int,
    holidays: set[date] | None = None,
) -> date:
    """Cboe final settlement date for the VX contract month `(year, month)`.

    `holidays` overrides the default federal-holiday proxy. Pass an explicit
    set when running near edge cases (e.g. CFE-only closures, partial-day
    holidays) where federal != exchange calendar.
    """
    if month not in MONTH_CODES:
        raise ValueError(f"month must be 1..12, got {month}")

    next_year = year + (1 if month == 12 else 0)
    next_month = 1 if month == 12 else month + 1
    third_fri_next = _third_friday(next_year, next_month)
    candidate = third_fri_next - timedelta(days=30)

    if holidays is None:
        holidays = _us_federal_holidays(candidate.year)
    return _roll_back_if_closed(candidate, holidays)


def vx_bloomberg_ticker(year: int, month: int) -> str:
    """Bloomberg security string for the VX contract month `(year, month)`.

    Format: `UX{month_code}{year_last_two_digits} Index`. See module
    docstring — 2-digit year is the only form Bloomberg consistently
    resolves for both active and expired contracts.
    """
    if month not in MONTH_CODES:
        raise ValueError(f"month must be 1..12, got {month}")
    return f"UX{MONTH_CODES[month]}{year % 100:02d} Index"


@dataclass(frozen=True)
class VXContract:
    """Identified VX futures contract: contract month, settlement, ticker."""
    year: int
    month: int
    settlement_date: date
    bloomberg_ticker: str

    @classmethod
    def from_month(
        cls,
        year: int,
        month: int,
        holidays: set[date] | None = None,
    ) -> "VXContract":
        return cls(
            year=year,
            month=month,
            settlement_date=vx_settlement_date(year, month, holidays),
            bloomberg_ticker=vx_bloomberg_ticker(year, month),
        )


def vx_contract_for_settlement(
    settlement_date: date,
    holidays: set[date] | None = None,
) -> VXContract:
    """Inverse lookup: given a settlement date, return the VX contract.

    Used as the join key into the option chain for ForwardSelector
    (ARCHITECTURE §4.1, §6.1). Searches the contract month containing the
    settlement date and its immediate neighbours to handle holiday-rolled
    edge cases where settlement crosses a month boundary.
    """
    y, m = settlement_date.year, settlement_date.month
    candidates: list[tuple[int, int]] = [(y, m)]
    candidates.append((y - 1, 12) if m == 1 else (y, m - 1))
    candidates.append((y + 1, 1) if m == 12 else (y, m + 1))

    for cy, cm in candidates:
        contract = VXContract.from_month(cy, cm, holidays)
        if contract.settlement_date == settlement_date:
            return contract

    raise ValueError(
        f"No standard VX contract month produces settlement on {settlement_date}. "
        f"This may be a non-standard / weekly contract."
    )
