from enum import Enum


class FillMode(Enum):
    SYNTHETIC_BIDASK = 'synthetic_bidask'           # base case — headline
    MIDPOINT = 'midpoint'                            # optimistic sensitivity
    SYNTHETIC_PLUS_SLIPPAGE = 'synthetic_slip'       # stressed sensitivity
