"""Tests for the VX expiry calendar.

Each row in KNOWN_CONTRACTS is a Cboe-published settlement date paired with
its Bloomberg ticker. The Juneteenth-2024 entry is the holiday-adjusted
case: Wed Jun 19 2024 was the federal Juneteenth holiday, so VX June 2024
(`UXM4 Index`) settled on Tue Jun 18 2024.
"""
from datetime import date

import pytest

from vix_spread.data.expiry_calendar import (
    VXContract,
    vx_bloomberg_ticker,
    vx_contract_for_settlement,
    vx_settlement_date,
)


KNOWN_CONTRACTS = [
    # (year, month, settlement_date, bloomberg_ticker, comment)
    (2025, 6,  date(2025, 6, 18),  "UXM25 Index", "standard Wed"),
    (2025, 5,  date(2025, 5, 21),  "UXK25 Index", "standard Wed"),
    (2024, 6,  date(2024, 6, 18),  "UXM24 Index", "Juneteenth Wed -> Tue rollback"),
    (2024, 3,  date(2024, 3, 20),  "UXH24 Index", "standard Wed"),
    (2024, 1,  date(2024, 1, 17),  "UXF24 Index", "standard Wed"),
    (2023, 12, date(2023, 12, 20), "UXZ23 Index", "standard Wed, year crosses"),
    (2023, 1,  date(2023, 1, 18),  "UXF23 Index", "standard Wed"),
]


@pytest.mark.parametrize(
    "year,month,settle,ticker,comment",
    KNOWN_CONTRACTS,
    ids=[f"{c[0]}-{c[1]:02d} {c[4]}" for c in KNOWN_CONTRACTS],
)
def test_settlement_date_matches_known(year, month, settle, ticker, comment):
    assert vx_settlement_date(year, month) == settle


@pytest.mark.parametrize(
    "year,month,settle,ticker,comment",
    KNOWN_CONTRACTS,
    ids=[f"{c[0]}-{c[1]:02d}" for c in KNOWN_CONTRACTS],
)
def test_bloomberg_ticker_matches_known(year, month, settle, ticker, comment):
    assert vx_bloomberg_ticker(year, month) == ticker


@pytest.mark.parametrize(
    "year,month,settle,ticker,comment",
    KNOWN_CONTRACTS,
    ids=[f"{c[0]}-{c[1]:02d}" for c in KNOWN_CONTRACTS],
)
def test_inverse_settlement_lookup_round_trips(year, month, settle, ticker, comment):
    contract = vx_contract_for_settlement(settle)
    assert contract == VXContract(
        year=year,
        month=month,
        settlement_date=settle,
        bloomberg_ticker=ticker,
    )


def test_juneteenth_2024_rolls_back_to_tuesday():
    """Wed Jun 19 2024 was the federal Juneteenth holiday — settlement
    must roll back to Tue Jun 18 2024 per the CFE rule."""
    settle = vx_settlement_date(2024, 6)
    assert settle == date(2024, 6, 18)
    assert settle.weekday() == 1  # Tuesday


def test_invalid_month_raises():
    with pytest.raises(ValueError):
        vx_bloomberg_ticker(2025, 13)
    with pytest.raises(ValueError):
        vx_settlement_date(2025, 0)


def test_inverse_lookup_for_non_settlement_date_raises():
    """A random non-settlement date does not correspond to any VX contract."""
    with pytest.raises(ValueError):
        vx_contract_for_settlement(date(2025, 4, 1))


def test_explicit_holiday_override():
    """Passing an explicit holidays set bypasses the federal-calendar proxy.
    Empty set => no rollback even on Juneteenth."""
    settle = vx_settlement_date(2024, 6, holidays=set())
    assert settle == date(2024, 6, 19)  # the un-adjusted Wednesday
