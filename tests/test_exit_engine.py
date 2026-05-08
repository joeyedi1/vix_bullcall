"""ExitEngine — closes spread positions per ExitPolicy.

Two paths:
  - FORCED_TUESDAY_LIQUIDATION: cross close-credit synthetic, gate-checked.
    Failure modes return FailedExit (NOT silent fills at theoretical).
  - HOLD_TO_SETTLEMENT: actual VRO print → SettlementOutcome.

The headline guard test is `test_force_tuesday_returns_failed_exit_when_short_no_market`:
the user-explicit "we aren't hallucinating liquidity on exit" check.
"""
from datetime import date, datetime, timezone

import pytest

from vix_spread.data.snapshot import SettlementMarket, VIXSnapshot
from vix_spread.execution.exit_engine import (
    ExitEngine,
    FailedExit,
    OpenPosition,
    SettlementOutcome,
)
from vix_spread.execution.exit_policy import ExitPolicy
from vix_spread.execution.fill_engine import ExecutedFill
from vix_spread.execution.fill_modes import FillMode
from vix_spread.execution.liquidity_gates import LiquidityGates
from vix_spread.execution.quote import OptionQuote
from vix_spread.products.spread import BullCallSpread
from vix_spread.products.vix_index_option import VIXIndexOption


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


SOQ_DATE = date(2026, 5, 20)
SOQ_DT = datetime(2026, 5, 20, 14, 30, tzinfo=timezone.utc)
EXIT_TS = datetime(2026, 5, 19, 19, 0, tzinfo=timezone.utc)   # Tue afternoon


def _spread(long_strike: float = 20.0, short_strike: float = 22.0) -> BullCallSpread:
    return BullCallSpread(
        long_leg=VIXIndexOption(
            contract_root="VIX", expiry=SOQ_DT, settlement_event=SOQ_DT,
            strike=long_strike, right="call",
        ),
        short_leg=VIXIndexOption(
            contract_root="VIX", expiry=SOQ_DT, settlement_event=SOQ_DT,
            strike=short_strike, right="call",
        ),
    )


def _ticker(strike: float) -> str:
    return f"VIX US 05/19/26 C{strike:g} Index"


def _q(
    contract_id: str, bid: float, ask: float,
    *, bid_size: int = 100, ask_size: int = 100,
    is_locked: bool = False, is_crossed: bool = False,
    quote_age_seconds: float = 2.0,
) -> OptionQuote:
    return OptionQuote(
        timestamp=EXIT_TS, contract_id=contract_id,
        bid=bid, ask=ask, bid_size=bid_size, ask_size=ask_size,
        last_trade=None, last_trade_age_seconds=None,
        is_locked=is_locked, is_crossed=is_crossed,
        quote_age_seconds=quote_age_seconds,
    )


def _market(quotes: dict[str, OptionQuote]) -> VIXSnapshot:
    return VIXSnapshot(
        timestamp=EXIT_TS, vx_curve={SOQ_DATE: 22.0},
        options_quotes=quotes, risk_free_rate=0.04, vix_spot=21.5,
    )


@pytest.fixture
def gates() -> LiquidityGates:
    return LiquidityGates(
        max_leg_spread_pct=0.5, min_displayed_size=1,
        max_quote_age_seconds=30.0,
    )


@pytest.fixture
def position() -> OpenPosition:
    return OpenPosition(spread=_spread(), size=2)


# --------------------------------------------------------------------------- #
# Path A: FORCED_TUESDAY_LIQUIDATION — happy path                              #
# --------------------------------------------------------------------------- #


def test_force_tuesday_close_credit_math(gates, position):
    """Close: SELL long at long.bid, BUY short at short.ask.
    close_credit = long.bid - short.ask.
    debit_per_spread = -close_credit (NEGATIVE for credit close)."""
    engine = ExitEngine(gates=gates)
    quotes = {
        _ticker(20): _q(_ticker(20), bid=2.50, ask=2.60),
        _ticker(22): _q(_ticker(22), bid=0.95, ask=1.05),
    }
    out = engine.execute_exit(
        position=position, market=_market(quotes),
        policy=ExitPolicy.FORCED_TUESDAY_LIQUIDATION,
    )
    assert isinstance(out, ExecutedFill)
    # close_credit = 2.50 - 1.05 = 1.45 → debit = -1.45
    assert out.debit_per_spread == pytest.approx(-1.45)
    assert out.long_leg_fill == pytest.approx(2.50)
    assert out.short_leg_fill == pytest.approx(1.05)
    assert out.fill_mode is FillMode.SYNTHETIC_BIDASK
    assert out.size == 2
    assert out.tick_rounded is False
    assert out.fees_per_spread == 0.0


def test_force_tuesday_propagates_position_size(gates, position):
    """Position size carries through to the ExecutedFill on close."""
    engine = ExitEngine(gates=gates)
    quotes = {
        _ticker(20): _q(_ticker(20), bid=2.50, ask=2.60),
        _ticker(22): _q(_ticker(22), bid=0.95, ask=1.05),
    }
    out = engine.execute_exit(
        position=OpenPosition(spread=position.spread, size=10),
        market=_market(quotes),
        policy=ExitPolicy.FORCED_TUESDAY_LIQUIDATION,
    )
    assert isinstance(out, ExecutedFill)
    assert out.size == 10


# --------------------------------------------------------------------------- #
# Path A: FailedExit branches                                                 #
# --------------------------------------------------------------------------- #


def test_force_tuesday_returns_failed_exit_when_long_no_bid(gates, position):
    """Long leg can't be sold (bid=0): FailedExit reason='no_bid_long'.
    Validation memo: this is the close-side parallel of the entry-side
    no_bid_short guard."""
    engine = ExitEngine(gates=gates)
    quotes = {
        _ticker(20): _q(_ticker(20), bid=0.0, ask=2.60),     # long no-bid
        _ticker(22): _q(_ticker(22), bid=0.95, ask=1.05),
    }
    out = engine.execute_exit(
        position=position, market=_market(quotes),
        policy=ExitPolicy.FORCED_TUESDAY_LIQUIDATION,
    )
    assert isinstance(out, FailedExit)
    assert out.reason == "no_bid_long"
    assert out.detail == {"long_bid": 0.0}
    assert out.size == position.size
    assert out.timestamp == EXIT_TS


def test_force_tuesday_returns_failed_exit_when_short_no_market(gates, position):
    """The headline 'aren't hallucinating liquidity on exit' guard.
    Far-OTM short leg with no quote at all (bid=0 AND ask=0): we
    cannot buy it back to close → FailedExit reason='no_ask_short'."""
    engine = ExitEngine(gates=gates)
    quotes = {
        _ticker(20): _q(_ticker(20), bid=2.50, ask=2.60),
        _ticker(22): _q(_ticker(22), bid=0.0, ask=0.0),       # no market at all
    }
    out = engine.execute_exit(
        position=position, market=_market(quotes),
        policy=ExitPolicy.FORCED_TUESDAY_LIQUIDATION,
    )
    assert isinstance(out, FailedExit)
    assert out.reason == "no_ask_short"
    assert out.detail == {"short_ask": 0.0}


def test_force_tuesday_returns_failed_exit_when_quote_missing(gates, position):
    """Long leg's contract_id not in market.options_quotes."""
    engine = ExitEngine(gates=gates)
    quotes = {
        # _ticker(20) MISSING
        _ticker(22): _q(_ticker(22), bid=0.95, ask=1.05),
    }
    out = engine.execute_exit(
        position=position, market=_market(quotes),
        policy=ExitPolicy.FORCED_TUESDAY_LIQUIDATION,
    )
    assert isinstance(out, FailedExit)
    assert out.reason == "no_quote"
    assert "long" in out.detail["missing_legs"]


def test_force_tuesday_returns_failed_exit_when_stale(gates, position):
    """quote_age_seconds beyond gate threshold → stale_quote rejection."""
    engine = ExitEngine(gates=gates)
    quotes = {
        _ticker(20): _q(_ticker(20), bid=2.50, ask=2.60, quote_age_seconds=120.0),
        _ticker(22): _q(_ticker(22), bid=0.95, ask=1.05),
    }
    out = engine.execute_exit(
        position=position, market=_market(quotes),
        policy=ExitPolicy.FORCED_TUESDAY_LIQUIDATION,
    )
    assert isinstance(out, FailedExit)
    assert out.reason == "stale_quote"
    assert out.detail["leg"] == "long"


def test_force_tuesday_returns_failed_exit_when_locked(gates, position):
    engine = ExitEngine(gates=gates)
    quotes = {
        _ticker(20): _q(_ticker(20), bid=2.50, ask=2.50, is_locked=True),
        _ticker(22): _q(_ticker(22), bid=0.95, ask=1.05),
    }
    out = engine.execute_exit(
        position=position, market=_market(quotes),
        policy=ExitPolicy.FORCED_TUESDAY_LIQUIDATION,
    )
    assert isinstance(out, FailedExit)
    assert out.reason == "locked"


def test_force_tuesday_returns_failed_exit_when_min_size_violated(position):
    """Size-vs-displayed gate enforced on close-side priorities:
    long.bid_size (the side we sell) and short.ask_size (the side we buy)."""
    gates_thin = LiquidityGates(
        max_leg_spread_pct=0.5, min_displayed_size=20,    # thin floor
        max_quote_age_seconds=30.0,
    )
    engine = ExitEngine(gates=gates_thin)
    quotes = {
        _ticker(20): _q(_ticker(20), bid=2.50, ask=2.60, bid_size=5, ask_size=5),
        _ticker(22): _q(_ticker(22), bid=0.95, ask=1.05, bid_size=50, ask_size=50),
    }
    out = engine.execute_exit(
        position=position, market=_market(quotes),
        policy=ExitPolicy.FORCED_TUESDAY_LIQUIDATION,
    )
    assert isinstance(out, FailedExit)
    assert out.reason == "gate_fail"
    assert out.detail["sub_reason"] == "min_displayed_size"


# --------------------------------------------------------------------------- #
# Path B: HOLD_TO_SETTLEMENT                                                  #
# --------------------------------------------------------------------------- #


def _settlement_market(vro: float) -> SettlementMarket:
    return SettlementMarket(
        vro_prints={SOQ_DT.date(): vro},
        vx_settle_prints={},
    )


def test_hold_to_settlement_payoff_below_long_strike(gates):
    """VRO=18, spread C20/C22: both legs OTM, expire worthless.
    Net payoff = 0 per spread."""
    engine = ExitEngine(gates=gates, settlement_market=_settlement_market(vro=18.0))
    out = engine.execute_exit(
        position=OpenPosition(spread=_spread(), size=2),
        market=_market({}),
        policy=ExitPolicy.HOLD_TO_SETTLEMENT,
    )
    assert isinstance(out, SettlementOutcome)
    assert out.long_leg_payoff == 0.0
    assert out.short_leg_payoff == 0.0
    assert out.net_payoff_per_spread == 0.0
    assert out.vro_print == 18.0


def test_hold_to_settlement_payoff_between_strikes(gates):
    """VRO=23, spread C20/C22: long ITM, short ITM but smaller payoff
    because at higher strike. Long pays max(23-20,0)*100=300;
    Short pays max(23-22,0)*100=100. Net = 200 per spread."""
    engine = ExitEngine(gates=gates, settlement_market=_settlement_market(vro=23.0))
    out = engine.execute_exit(
        position=OpenPosition(spread=_spread(), size=2),
        market=_market({}),
        policy=ExitPolicy.HOLD_TO_SETTLEMENT,
    )
    assert isinstance(out, SettlementOutcome)
    assert out.long_leg_payoff == pytest.approx(300.0)
    assert out.short_leg_payoff == pytest.approx(100.0)
    assert out.net_payoff_per_spread == pytest.approx(200.0)
    assert out.vro_print == 23.0


def test_hold_to_settlement_max_payoff_above_short_strike(gates):
    """VRO=25, spread C20/C22: deep ITM both legs.
    Long: max(25-20,0)*100=500; Short: max(25-22,0)*100=300.
    Net = 200 (the spread width × multiplier — max possible profit)."""
    engine = ExitEngine(gates=gates, settlement_market=_settlement_market(vro=25.0))
    out = engine.execute_exit(
        position=OpenPosition(spread=_spread(), size=1),
        market=_market({}),
        policy=ExitPolicy.HOLD_TO_SETTLEMENT,
    )
    assert isinstance(out, SettlementOutcome)
    assert out.long_leg_payoff == pytest.approx(500.0)
    assert out.short_leg_payoff == pytest.approx(300.0)
    assert out.net_payoff_per_spread == pytest.approx(200.0)


def test_hold_to_settlement_timestamp_is_soq_event(gates):
    """SettlementOutcome.timestamp should be the option's settlement_event
    (the SOQ Wed morning when VRO is published), not the call's `as_of`."""
    engine = ExitEngine(gates=gates, settlement_market=_settlement_market(vro=21.0))
    out = engine.execute_exit(
        position=OpenPosition(spread=_spread(), size=1),
        market=_market({}),
        policy=ExitPolicy.HOLD_TO_SETTLEMENT,
    )
    assert isinstance(out, SettlementOutcome)
    assert out.timestamp == SOQ_DT


def test_hold_to_settlement_requires_settlement_market_in_constructor(gates):
    """policy=HOLD_TO_SETTLEMENT without a constructor settlement_market
    is a configuration error — refused at execute_exit."""
    engine = ExitEngine(gates=gates, settlement_market=None)
    with pytest.raises(ValueError, match="settlement_market"):
        engine.execute_exit(
            position=OpenPosition(spread=_spread(), size=1),
            market=_market({}),
            policy=ExitPolicy.HOLD_TO_SETTLEMENT,
        )


def test_hold_to_settlement_propagates_keyerror_when_vro_missing(gates):
    """If the SettlementMarket lacks the VRO for this expiry, the lookup
    raises KeyError — the caller / report layer surfaces it as a data
    gap rather than fabricating a payoff."""
    sm = SettlementMarket(vro_prints={}, vx_settle_prints={})  # empty!
    engine = ExitEngine(gates=gates, settlement_market=sm)
    with pytest.raises(KeyError):
        engine.execute_exit(
            position=OpenPosition(spread=_spread(), size=1),
            market=_market({}),
            policy=ExitPolicy.HOLD_TO_SETTLEMENT,
        )


# --------------------------------------------------------------------------- #
# Dispatch                                                                     #
# --------------------------------------------------------------------------- #


def test_unknown_policy_raises(gates, position):
    engine = ExitEngine(gates=gates)
    with pytest.raises(ValueError, match="ExitPolicy"):
        engine.execute_exit(
            position=position, market=_market({}),
            policy="garbage",  # type: ignore[arg-type]
        )
