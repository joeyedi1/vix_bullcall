"""SpreadEvaluator — integration tests across Phase-2/3/4 layers.

Verifies the composer correctly orchestrates ForwardSelector +
ChainIVProvider + Black76Pricer + FillEngine for a single decision.
Also exercises the no-quote refusal branch (a gate_fail rejection rather
than a raise / NaN propagation).
"""
from datetime import date, datetime, timezone

import pandas as pd
import pytest

from vix_spread.data.snapshot import VIXSnapshot
from vix_spread.data.vix_index_options import vix_option_active_contract_id
from vix_spread.execution.fill_engine import (
    ExecutedFill,
    FillEngine,
    RejectedOrder,
)
from vix_spread.execution.fill_modes import FillMode
from vix_spread.execution.liquidity_gates import LiquidityGates
from vix_spread.execution.quote import OptionQuote
from vix_spread.pricing.black76 import Black76Pricer
from vix_spread.pricing.evaluator import SpreadEvaluation, SpreadEvaluator
from vix_spread.pricing.forward_selector import ForwardSelector
from vix_spread.pricing.leg_iv import ChainIVProvider, LegIVSource
from vix_spread.products.spread import BullCallSpread
from vix_spread.products.vix_index_option import VIXIndexOption
from vix_spread.utils.errors import ForwardSelectionError


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


SOQ_WED = datetime(2026, 6, 17, 14, 30, tzinfo=timezone.utc)
SOQ_DATE = SOQ_WED.date()
LAST_TRADE = date(2026, 6, 16)
AS_OF = datetime(2026, 5, 1, 14, 0, tzinfo=timezone.utc)
RFR = 0.04


def _spread() -> BullCallSpread:
    long_leg = VIXIndexOption(
        contract_root="VIX", expiry=SOQ_WED, settlement_event=SOQ_WED,
        strike=20.0, right="call",
    )
    short_leg = VIXIndexOption(
        contract_root="VIX", expiry=SOQ_WED, settlement_event=SOQ_WED,
        strike=22.0, right="call",
    )
    return BullCallSpread(long_leg=long_leg, short_leg=short_leg)


def _quote(
    contract_id: str, bid: float, ask: float,
    *, bid_size: int = 100, ask_size: int = 100,
    is_locked: bool = False, is_crossed: bool = False,
    quote_age_seconds: float = 2.0,
) -> OptionQuote:
    return OptionQuote(
        timestamp=AS_OF, contract_id=contract_id,
        bid=bid, ask=ask, bid_size=bid_size, ask_size=ask_size,
        last_trade=None, last_trade_age_seconds=None,
        is_locked=is_locked, is_crossed=is_crossed,
        quote_age_seconds=quote_age_seconds,
    )


def _chain_panel() -> pd.DataFrame:
    """Vendor IV chain for the long+short legs."""
    rows = [
        {"date": AS_OF.date(), "expiry": SOQ_DATE, "right": "C",
         "strike": 20.0, "IVOL_LAST": 70.0, "PX_BID": 2.10, "PX_ASK": 2.20},
        {"date": AS_OF.date(), "expiry": SOQ_DATE, "right": "C",
         "strike": 22.0, "IVOL_LAST": 65.0, "PX_BID": 0.95, "PX_ASK": 1.05},
    ]
    df = pd.DataFrame(rows)
    return df.set_index(["date", "expiry", "right", "strike"])


def _snapshot(quotes: dict[str, OptionQuote] | None = None) -> VIXSnapshot:
    """Snapshot with vx_curve at the spread's settlement date and
    optional `quotes` keyed on the active-form contract_id."""
    return VIXSnapshot(
        timestamp=AS_OF,
        vx_curve={SOQ_DATE: 22.0},          # forward = 22.0
        options_quotes=quotes or {},
        risk_free_rate=RFR,
        vix_spot=18.0,                       # diagnostic only
    )


@pytest.fixture
def evaluator() -> SpreadEvaluator:
    return SpreadEvaluator(
        chain_iv_provider=ChainIVProvider(_chain_panel(), Black76Pricer()),
        forward_selector=ForwardSelector(),
        pricer=Black76Pricer(),
        fill_engine=FillEngine(),
        gates=LiquidityGates(
            max_leg_spread_pct=0.5, min_displayed_size=1,
            max_quote_age_seconds=30.0,
        ),
    )


# --------------------------------------------------------------------------- #
# Active contract_id helper — pre-flight                                      #
# --------------------------------------------------------------------------- #


def test_active_contract_id_is_tuesday_last_trade_form():
    """A Product whose `expiry.date()` is Wed SOQ produces a Bloomberg
    contract_id with the Tuesday last-trade date."""
    spread = _spread()
    long_id = vix_option_active_contract_id(spread.long_leg)
    short_id = vix_option_active_contract_id(spread.short_leg)
    # SOQ Wed = Jun 17 → last trade Tue = Jun 16
    assert long_id == "VIX US 06/16/26 C20 Index"
    assert short_id == "VIX US 06/16/26 C22 Index"


# --------------------------------------------------------------------------- #
# Happy path: theoretical + executed fill                                     #
# --------------------------------------------------------------------------- #


def test_evaluate_happy_path_returns_filled_evaluation(evaluator):
    long_id = "VIX US 06/16/26 C20 Index"
    short_id = "VIX US 06/16/26 C22 Index"
    quotes = {
        long_id: _quote(long_id, bid=2.10, ask=2.20),
        short_id: _quote(short_id, bid=0.95, ask=1.05),
    }
    result = evaluator.evaluate(spread=_spread(), snapshot=_snapshot(quotes))

    assert isinstance(result, SpreadEvaluation)
    assert isinstance(result.fill, ExecutedFill)
    assert result.is_filled

    # Forward came from settlement-date match.
    assert result.forward.value == pytest.approx(22.0)
    assert result.forward.selection_method == "settlement_date_match"

    # IVs came from vendor (panel had populated IVOL_LAST for both legs).
    assert result.iv_long.source is LegIVSource.VENDOR
    assert result.iv_short.source is LegIVSource.VENDOR
    assert result.iv_long.value == pytest.approx(0.70)
    assert result.iv_short.value == pytest.approx(0.65)

    # Theoretical is the FlatVolError-guarded spread value.
    assert result.theoretical.long_leg.iv_used == pytest.approx(0.70)
    assert result.theoretical.short_leg.iv_used == pytest.approx(0.65)

    # SYNTHETIC_BIDASK debit = 2.20 - 0.95 = 1.25
    assert result.fill.debit_per_spread == pytest.approx(1.25)
    assert result.fill.fill_mode is FillMode.SYNTHETIC_BIDASK


def test_edge_bleed_is_theoretical_minus_executed(evaluator):
    long_id = "VIX US 06/16/26 C20 Index"
    short_id = "VIX US 06/16/26 C22 Index"
    quotes = {
        long_id: _quote(long_id, bid=2.10, ask=2.20),
        short_id: _quote(short_id, bid=0.95, ask=1.05),
    }
    result = evaluator.evaluate(spread=_spread(), snapshot=_snapshot(quotes))
    assert result.is_filled
    expected = result.theoretical.value - result.fill.debit_per_spread
    assert result.edge_bleed == pytest.approx(expected)


def test_edge_bleed_is_none_when_rejected(evaluator):
    long_id = "VIX US 06/16/26 C20 Index"
    short_id = "VIX US 06/16/26 C22 Index"
    quotes = {
        long_id: _quote(long_id, bid=2.10, ask=2.20),
        short_id: _quote(short_id, bid=0.0, ask=1.05),       # no-bid short
    }
    result = evaluator.evaluate(spread=_spread(), snapshot=_snapshot(quotes))
    assert isinstance(result.fill, RejectedOrder)
    assert result.edge_bleed is None


# --------------------------------------------------------------------------- #
# Refusal branches                                                             #
# --------------------------------------------------------------------------- #


def test_evaluate_returns_rejected_when_quote_stale(evaluator):
    long_id = "VIX US 06/16/26 C20 Index"
    short_id = "VIX US 06/16/26 C22 Index"
    quotes = {
        long_id: _quote(long_id, bid=2.10, ask=2.20, quote_age_seconds=120.0),
        short_id: _quote(short_id, bid=0.95, ask=1.05),
    }
    result = evaluator.evaluate(spread=_spread(), snapshot=_snapshot(quotes))
    assert isinstance(result.fill, RejectedOrder)
    assert result.fill.reason == "stale_quote"
    # Theoretical + IVs still computed even when fill is rejected — the
    # report can still measure model edge for compare-only signals.
    assert result.theoretical is not None
    assert result.iv_long.source is LegIVSource.VENDOR


def test_evaluate_no_quote_long_leg_returns_no_quote_rejection(evaluator):
    short_id = "VIX US 06/16/26 C22 Index"
    quotes = {short_id: _quote(short_id, bid=0.95, ask=1.05)}  # long missing
    result = evaluator.evaluate(spread=_spread(), snapshot=_snapshot(quotes))
    assert isinstance(result.fill, RejectedOrder)
    assert result.fill.reason == "gate_fail"
    assert result.fill.detail["sub_reason"] == "no_quote"
    assert "long" in result.fill.detail["missing_legs"]
    assert result.fill.detail["long_id"] == "VIX US 06/16/26 C20 Index"


def test_evaluate_no_quote_short_leg_returns_no_quote_rejection(evaluator):
    long_id = "VIX US 06/16/26 C20 Index"
    quotes = {long_id: _quote(long_id, bid=2.10, ask=2.20)}    # short missing
    result = evaluator.evaluate(spread=_spread(), snapshot=_snapshot(quotes))
    assert isinstance(result.fill, RejectedOrder)
    assert result.fill.reason == "gate_fail"
    assert result.fill.detail["sub_reason"] == "no_quote"
    assert "short" in result.fill.detail["missing_legs"]


def test_evaluate_raises_on_forward_selection_failure(evaluator):
    """If `vx_curve` lacks the spread's settlement_date, ForwardSelector
    raises ForwardSelectionError — composer propagates rather than
    silently falling back."""
    snap = VIXSnapshot(
        timestamp=AS_OF, vx_curve={},  # empty
        options_quotes={}, risk_free_rate=RFR,
    )
    with pytest.raises(ForwardSelectionError):
        evaluator.evaluate(spread=_spread(), snapshot=snap)


# --------------------------------------------------------------------------- #
# Fill mode propagation                                                        #
# --------------------------------------------------------------------------- #


def test_evaluate_propagates_synthetic_plus_slippage(evaluator):
    long_id = "VIX US 06/16/26 C20 Index"
    short_id = "VIX US 06/16/26 C22 Index"
    quotes = {
        long_id: _quote(long_id, bid=2.10, ask=2.20),
        short_id: _quote(short_id, bid=0.95, ask=1.05),
    }
    result = evaluator.evaluate(
        spread=_spread(), snapshot=_snapshot(quotes),
        fill_mode=FillMode.SYNTHETIC_PLUS_SLIPPAGE,
        slippage_ticks_per_leg=1,
    )
    assert isinstance(result.fill, ExecutedFill)
    assert result.fill.fill_mode is FillMode.SYNTHETIC_PLUS_SLIPPAGE
    # SYNTHETIC_BIDASK debit = 1.25; +0.05 each leg -> 1.35
    assert result.fill.debit_per_spread == pytest.approx(1.35)


def test_evaluate_midpoint_requires_explicit_optimism_flag(evaluator):
    long_id = "VIX US 06/16/26 C20 Index"
    short_id = "VIX US 06/16/26 C22 Index"
    quotes = {
        long_id: _quote(long_id, bid=2.10, ask=2.20),
        short_id: _quote(short_id, bid=0.95, ask=1.05),
    }
    with pytest.raises(ValueError, match="accept_midpoint_optimism"):
        evaluator.evaluate(
            spread=_spread(), snapshot=_snapshot(quotes),
            fill_mode=FillMode.MIDPOINT,
        )
    # With the explicit flag, the call succeeds.
    result = evaluator.evaluate(
        spread=_spread(), snapshot=_snapshot(quotes),
        fill_mode=FillMode.MIDPOINT, accept_midpoint_optimism=True,
    )
    assert isinstance(result.fill, ExecutedFill)
    # mid(long) - mid(short) = 2.15 - 1.00 = 1.15
    assert result.fill.debit_per_spread == pytest.approx(1.15)


# --------------------------------------------------------------------------- #
# B76-invert IV path through the composer                                     #
# --------------------------------------------------------------------------- #


def test_evaluate_uses_b76_inverted_iv_when_vendor_missing():
    """Composer should fall back to B76-inverted IV transparently when
    vendor IV is missing — and tag the LegIV.source accordingly."""
    pricer = Black76Pricer()
    long_leg = VIXIndexOption(
        contract_root="VIX", expiry=SOQ_WED, settlement_event=SOQ_WED,
        strike=20.0, right="call",
    )
    short_leg = VIXIndexOption(
        contract_root="VIX", expiry=SOQ_WED, settlement_event=SOQ_WED,
        strike=22.0, right="call",
    )
    spread = BullCallSpread(long_leg=long_leg, short_leg=short_leg)

    # Forward is 22.0; create chain panel with vendor IV MISSING for long
    # leg, present (different from short) for the short leg.
    from vix_spread.pricing.forward_selector import Forward
    forward = Forward(
        value=22.0, selection_method="settlement_date_match",
        model_risk_flag=False, settlement_date=SOQ_WED,
    )
    # Synthetic mid for long leg @ true_iv=0.85 → use that as the chain mid.
    target_mid = pricer.price(long_leg, forward, 0.85, AS_OF, RFR).value
    panel = pd.DataFrame([
        {"date": AS_OF.date(), "expiry": SOQ_DATE, "right": "C",
         "strike": 20.0, "IVOL_LAST": float("nan"),
         "PX_BID": target_mid - 1e-6, "PX_ASK": target_mid + 1e-6},
        {"date": AS_OF.date(), "expiry": SOQ_DATE, "right": "C",
         "strike": 22.0, "IVOL_LAST": 65.0, "PX_BID": 0.95, "PX_ASK": 1.05},
    ]).set_index(["date", "expiry", "right", "strike"])

    long_id = "VIX US 06/16/26 C20 Index"
    short_id = "VIX US 06/16/26 C22 Index"
    quotes = {
        long_id: _quote(long_id, bid=2.10, ask=2.20),
        short_id: _quote(short_id, bid=0.95, ask=1.05),
    }
    snap = _snapshot(quotes)

    evaluator = SpreadEvaluator(
        chain_iv_provider=ChainIVProvider(panel, pricer),
        forward_selector=ForwardSelector(),
        pricer=pricer,
        fill_engine=FillEngine(),
        gates=LiquidityGates(
            max_leg_spread_pct=0.5, min_displayed_size=1,
            max_quote_age_seconds=30.0,
        ),
    )
    result = evaluator.evaluate(spread=spread, snapshot=snap)

    # Long leg: B76-inverted, recovers true_iv=0.85 within tol
    assert result.iv_long.source is LegIVSource.B76_INVERTED
    assert result.iv_long.value == pytest.approx(0.85, abs=1e-5)
    # Short leg: vendor unchanged at 0.65
    assert result.iv_short.source is LegIVSource.VENDOR
    assert result.iv_short.value == pytest.approx(0.65)
    # Fill still works
    assert isinstance(result.fill, ExecutedFill)
