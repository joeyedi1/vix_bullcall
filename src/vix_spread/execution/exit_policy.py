"""ExitPolicy — enum for `ExitEngine` policy selection.

ARCHITECTURE §5.6. The two production exit modes; the engine itself
(ARCH §10) is a Phase-6 deliverable — this enum lives here so Phase-5
strategy composition (`VIXBullCallSpreadStrategy.exit_policy`) can
declare a policy now without an ExitEngine yet.

  - `FORCED_TUESDAY_LIQUIDATION`: liquidate before the Wednesday SOQ
    window. Reduces SOQ risk but is NOT costless — liquidity can
    deteriorate near final trading hours; far-OTM short legs can go
    no-bid. The validation memo treats this as the headline policy.

  - `HOLD_TO_SETTLEMENT`: accept the SOQ/VRO outcome via
    `Product.settlement_value(market)`. Uses the ACTUAL historical VRO
    print — never spot VIX close, never theoretical Black-76 value.

A backtest run reports under one declared policy; the secondary policy
is a sensitivity scenario, not a default.
"""
from enum import Enum


class ExitPolicy(Enum):
    FORCED_TUESDAY_LIQUIDATION = "forced_tuesday"
    HOLD_TO_SETTLEMENT = "hold_to_settlement"
