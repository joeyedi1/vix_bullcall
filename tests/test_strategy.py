"""VIXBullCallSpreadStrategy — per-minute decision composer.

Verifies the decision flow:
  hypothesis filters -> SpreadSelector -> SpreadEvaluator -> FixedRiskSizer
each producing a typed `StrategyDecision` skip/enter outcome with the
correct `reason` tag for audit.
"""
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from vix_spread.data.snapshot import VIXSnapshot
from vix_spread.execution.exit_policy import ExitPolicy
from vix_spread.execution.fill_engine import FillEngine
from vix_spread.execution.fill_modes import FillMode
from vix_spread.execution.liquidity_gates import LiquidityGates
from vix_spread.execution.quote import OptionQuote
from vix_spread.pricing.black76 import Black76Pricer
from vix_spread.pricing.evaluator import SpreadEvaluator
from vix_spread.pricing.forward_selector import ForwardSelector
from vix_spread.pricing.leg_iv import ChainIVProvider
from vix_spread.regime.base import RegimeSignal
from vix_spread.strategy.hypothesis import make_contrarian_tail_hypothesis
from vix_spread.strategy.sizing import FixedRiskSizer
from vix_spread.strategy.spread_selector import SpreadSelector
from vix_spread.strategy.strategy import (
    StrategyDecision,
    VIXBullCallSpreadStrategy,
)
from vix_spread.utils.errors import LookaheadError


# --------------------------------------------------------------------------- #
# Fixtures — synthetic minimal market state                                   #
# --------------------------------------------------------------------------- #


SOQ_DATE = date(2026, 5, 20)            # Wed SOQ for May 2026 monthly
AS_OF = datetime(2026, 4, 15, 14, 0, tzinfo=timezone.utc)
FORWARD = 22.0
RFR = 0.04
EQUITY = 100_000.0


def _ticker(strike: float) -> str:
    return f"VIX US 05/19/26 C{strike:g} Index"


def _q(strike: float, bid: float, ask: float) -> OptionQuote:
    return OptionQuote(
        timestamp=AS_OF, contract_id=_ticker(strike),
        bid=bid, ask=ask, bid_size=100, ask_size=100,
        last_trade=None, last_trade_age_seconds=None,
        is_locked=False, is_crossed=False, quote_age_seconds=2.0,
    )


def _snapshot(quotes: dict[str, OptionQuote] | None = None) -> VIXSnapshot:
    return VIXSnapshot(
        timestamp=AS_OF,
        vx_curve={SOQ_DATE: FORWARD},
        options_quotes=quotes or {},
        risk_free_rate=RFR, vix_spot=18.0,
    )


def _signal(
    *,
    state_label: int = 0,
    probs: tuple[float, float] = (0.85, 0.15),
    contango: float = 0.05,
    as_of: datetime = AS_OF,
) -> RegimeSignal:
    return RegimeSignal(
        as_of=as_of,
        filtered_probs=np.array(probs),
        state_label=state_label,
        curve_features={"slope_30_182": contango},
        hypothesis_tag="contrarian_tail",
        as_of_inputs={"log_vix": as_of},
    )


def _populated_quotes() -> dict[str, OptionQuote]:
    """Strikes 22, 24, 26, 28, 30, 32 with normal NBBO."""
    return {
        _ticker(s): _q(s, bid=2.0, ask=2.1) for s in (22, 24, 26, 28, 30, 32)
    }


def _chain_panel() -> pd.DataFrame:
    """Vendor IV panel — distinct IV per strike to dodge the FlatVolError
    guard (which fires when iv_long == iv_short on any non-zero-width
    spread). Skew increases monotonically by strike to mimic realistic
    vol surface."""
    rows = [
        {"date": AS_OF.date(), "expiry": SOQ_DATE, "right": "C",
         "strike": float(s), "IVOL_LAST": 70.0 + s * 0.5,   # 81, 82, 83, ... by strike
         "PX_BID": 2.0, "PX_ASK": 2.1}
        for s in (22, 24, 26, 28, 30, 32)
    ]
    return pd.DataFrame(rows).set_index(["date", "expiry", "right", "strike"])


@pytest.fixture
def strategy() -> VIXBullCallSpreadStrategy:
    pricer = Black76Pricer()
    return VIXBullCallSpreadStrategy(
        hypothesis=make_contrarian_tail_hypothesis(),
        spread_selector=SpreadSelector(
            long_offset=2.0, short_offset=4.0, dte_min=7, dte_max=60,
        ),
        evaluator=SpreadEvaluator(
            chain_iv_provider=ChainIVProvider(_chain_panel(), pricer),
            forward_selector=ForwardSelector(),
            pricer=pricer,
            fill_engine=FillEngine(),
            gates=LiquidityGates(
                max_leg_spread_pct=0.5, min_displayed_size=1,
                max_quote_age_seconds=30.0,
            ),
        ),
        sizer=FixedRiskSizer(risk_per_trade_pct=0.005),
        exit_policy=ExitPolicy.FORCED_TUESDAY_LIQUIDATION,
    )


# --------------------------------------------------------------------------- #
# Happy path                                                                  #
# --------------------------------------------------------------------------- #


def test_evaluate_enters_when_all_filters_pass(strategy):
    decision = strategy.evaluate(
        market=_snapshot(_populated_quotes()),
        signal=_signal(),
        as_of=AS_OF,
        equity=EQUITY,
    )
    assert isinstance(decision, StrategyDecision)
    assert decision.action == "enter"
    assert decision.reason == "entered"
    assert decision.hypothesis_name == "contrarian_tail"
    assert decision.spread is not None
    assert decision.evaluation is not None
    assert decision.size is not None and decision.size > 0
    # As-of stamp matches input; strategy doesn't shift time.
    assert decision.as_of == AS_OF


def test_evaluate_carries_full_provenance_on_enter(strategy):
    decision = strategy.evaluate(
        market=_snapshot(_populated_quotes()),
        signal=_signal(),
        as_of=AS_OF, equity=EQUITY,
    )
    assert decision.evaluation.is_filled
    assert decision.evaluation.theoretical is not None
    assert decision.evaluation.iv_long is not None
    assert decision.evaluation.iv_short is not None
    assert decision.evaluation.forward.value == pytest.approx(FORWARD)


# --------------------------------------------------------------------------- #
# Skip branches — each filter / stage                                          #
# --------------------------------------------------------------------------- #


def test_skips_with_regime_filter_reason(strategy):
    """High-vol state -> hypothesis regime filter False -> skip."""
    decision = strategy.evaluate(
        market=_snapshot(_populated_quotes()),
        signal=_signal(state_label=1, probs=(0.15, 0.85)),
        as_of=AS_OF, equity=EQUITY,
    )
    assert decision.action == "skip"
    assert decision.reason == "regime_filter"
    assert decision.spread is None
    assert decision.evaluation is None
    assert decision.size is None


def test_skips_with_curve_filter_reason(strategy):
    """Backwardation -> hypothesis curve filter False -> skip."""
    decision = strategy.evaluate(
        market=_snapshot(_populated_quotes()),
        signal=_signal(contango=-0.04),
        as_of=AS_OF, equity=EQUITY,
    )
    assert decision.action == "skip"
    assert decision.reason == "curve_filter"
    assert decision.spread is None


def test_skips_with_no_spread_reason(strategy):
    """Filters pass but no quotes available -> SpreadSelector returns None."""
    decision = strategy.evaluate(
        market=_snapshot(quotes={}),
        signal=_signal(),
        as_of=AS_OF, equity=EQUITY,
    )
    assert decision.action == "skip"
    assert decision.reason == "no_spread"
    assert decision.spread is None


def test_skips_with_fill_rejected_reason(strategy):
    """Filters and selection pass but the FillEngine rejects (e.g.,
    no-bid short on the picked strike — though our selector pre-filters,
    a stale quote would still trip the gates)."""
    quotes = {
        # Strike 24 (long pick), 28 (short pick): make 28 stale to fail gates.
        _ticker(22): _q(22, bid=2.0, ask=2.1),
        _ticker(24): _q(24, bid=1.5, ask=1.6),
        _ticker(28): OptionQuote(
            timestamp=AS_OF, contract_id=_ticker(28),
            bid=1.0, ask=1.1, bid_size=100, ask_size=100,
            last_trade=None, last_trade_age_seconds=None,
            is_locked=False, is_crossed=False,
            quote_age_seconds=120.0,                # > 30s gate threshold
        ),
        _ticker(30): _q(30, bid=0.5, ask=0.6),
    }
    decision = strategy.evaluate(
        market=_snapshot(quotes), signal=_signal(),
        as_of=AS_OF, equity=EQUITY,
    )
    assert decision.action == "skip"
    assert decision.reason == "fill_rejected"
    # Evaluation IS populated on this skip — it carries the RejectedOrder
    # for audit logging.
    assert decision.evaluation is not None
    assert decision.spread is not None  # selector did pick a spread
    assert decision.size is None


def test_skips_with_size_zero_reason(strategy):
    """Filters + selection + fill all succeed, but equity is too small
    to size even one contract (debit × multiplier > equity × pct)."""
    decision = strategy.evaluate(
        market=_snapshot(_populated_quotes()),
        signal=_signal(),
        as_of=AS_OF,
        equity=100.0,                                # tiny budget
    )
    assert decision.action == "skip"
    assert decision.reason == "size_zero"
    assert decision.evaluation is not None
    assert decision.spread is not None


# --------------------------------------------------------------------------- #
# Lookahead guard                                                              #
# --------------------------------------------------------------------------- #


def test_raises_lookahead_when_signal_after_as_of(strategy):
    """Strategy must refuse a signal computed from data later than
    `as_of` — structural guard against silent leakage."""
    future_signal = _signal(as_of=AS_OF + timedelta(hours=1))
    with pytest.raises(LookaheadError):
        strategy.evaluate(
            market=_snapshot(_populated_quotes()),
            signal=future_signal,
            as_of=AS_OF, equity=EQUITY,
        )


def test_signal_at_exactly_as_of_is_allowed(strategy):
    """signal.as_of == as_of is permitted (the signal was computed AT the
    decision time using only data with timestamp <= as_of)."""
    decision = strategy.evaluate(
        market=_snapshot(_populated_quotes()),
        signal=_signal(as_of=AS_OF),
        as_of=AS_OF, equity=EQUITY,
    )
    assert decision.action == "enter"


# --------------------------------------------------------------------------- #
# Composition / audit                                                          #
# --------------------------------------------------------------------------- #


def test_strategy_carries_exit_policy_for_audit(strategy):
    """ARCH §7.1: exit_policy is constructor-bound; audit-only on the
    entry strategy (the ExitEngine consumes during the loop)."""
    assert strategy.exit_policy is ExitPolicy.FORCED_TUESDAY_LIQUIDATION


def test_decision_hypothesis_name_propagates(strategy):
    """The hypothesis name is on every decision for audit binning."""
    enter = strategy.evaluate(
        market=_snapshot(_populated_quotes()),
        signal=_signal(),
        as_of=AS_OF, equity=EQUITY,
    )
    skip = strategy.evaluate(
        market=_snapshot(_populated_quotes()),
        signal=_signal(state_label=1, probs=(0.15, 0.85)),
        as_of=AS_OF, equity=EQUITY,
    )
    assert enter.hypothesis_name == "contrarian_tail"
    assert skip.hypothesis_name == "contrarian_tail"


def test_evaluate_propagates_fill_mode_to_evaluator(strategy):
    """fill_mode on evaluate -> evaluator -> FillEngine; SYNTHETIC_PLUS_SLIPPAGE
    should produce a higher debit than SYNTHETIC_BIDASK."""
    quotes = _populated_quotes()
    base = strategy.evaluate(
        market=_snapshot(quotes), signal=_signal(),
        as_of=AS_OF, equity=EQUITY,
        fill_mode=FillMode.SYNTHETIC_BIDASK,
    )
    slipped = strategy.evaluate(
        market=_snapshot(quotes), signal=_signal(),
        as_of=AS_OF, equity=EQUITY,
        fill_mode=FillMode.SYNTHETIC_PLUS_SLIPPAGE,
        slippage_ticks_per_leg=1,
    )
    assert base.action == "enter"
    assert slipped.action == "enter"
    assert (
        slipped.evaluation.fill.debit_per_spread
        > base.evaluation.fill.debit_per_spread
    )
