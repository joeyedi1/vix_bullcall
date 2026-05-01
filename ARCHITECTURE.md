# VIX Bull Call Spread Backtester — Implementation Plan

**Purpose.** Build a production-grade backtester for a VIX bull call spread strategy. Tests performance across regimes — specifically the 2025 low-volatility / hyper-contango environment flagged in the validation memo. The architecture must enforce the strict model-risk controls from the validation memo *by construction*: regime classification cannot peek at the future, options cannot be priced from spot VIX or generic month numbers, the two VIX option products cannot be conflated, and executed P&L cannot be computed from mid-price last-trade bars.

**Hard constraints (validation-memo enforced).**
- `RegimeClassifier` is strictly causal. Only filtered probabilities `P(S_t | y_{1:t})` are exposed. Smoothed probabilities, full-sample Viterbi paths, and any percentile / scaler / threshold computed from data with timestamps `> t` are blocked at the type system, not by convention.
- `Product` is an abstract base with two concrete subclasses: `VIXIndexOption` (cash-settled to SOQ/VRO, `$100` multiplier, European) and `VXFutureOption` (physically settled into one VX futures contract, `$1000` underlying VX multiplier, European). All pricing, settlement, P&L, and Greek-scaling code branches on product type — never on string tickers or column flags.
- Pricing uses Black-76 with exact minute-level calendar time-to-expiry (`T = minutes_to_settlement / 525_600`). The forward input is selected by exact settlement-date match or same-expiry put-call-parity implied forward — never spot VIX, never a generic front month.
- Execution simulation is built on synthetic bid/ask spreads constructed from 1-minute quote data. Midpoint fills are an explicit optional sensitivity scenario, never the base case. Theoretical Black-76 prices are diagnostics, never fills.

---

## 1. Design principles

1. **Strict causal data flow, enforced by type.** Every feature carries an `as_of` timestamp. A `FeatureAvailability` validator inspects the timestamps of all inputs to the regime engine, the pricer, and the fill engine, and raises `LookaheadError` if any input is stamped after the decision time. This is a runtime invariant, not a code-review check.

2. **Product type is a first-class object.** Pricing, settlement, exit, multiplier, and Greek-scaling code dispatch on `Product` subclass. There is no `if ticker.startswith("VIX")` logic anywhere in the pricing or P&L pipeline. New products (e.g., VIXW weeklies as a sibling of `VIXIndexOption`) plug in by subclassing.

3. **Theoretical and executable prices are separated.** `Black76Pricer` returns `TheoreticalPrice` objects with `is_executable=False` as a sentinel; `FillEngine` returns `ExecutedFill` objects. The accounting layer accepts only `ExecutedFill`. Theoretical prices appear in diagnostics and edge-bleed reports, never in P&L.

4. **`RegimeSignal` is a first-class object, not a boolean.** A signal carries its decision timestamp, the filtered probability vector, the curve features, the strategy-hypothesis tag, and the per-input as-of state. Downstream code consumes signals; signals do not consume themselves.

5. **Walk-forward is the only acceptable backtest mode.** There is no "fit on full history, evaluate on full history" path in the codebase. The HMM, percentile thresholds, scalers, and IV surface calibrators are all wrapped in walk-forward refitters that take `(t,)` and return parameters fit on `(:t-1)` only.

6. **Execution realism has primacy over signal cleverness.** The fill engine can reject an arbitrarily good signal due to no-bid legs, stale quotes, locked/crossed markets, or liquidity-gate failures. Reports surface rejected signals as a first-class number alongside P&L. A signal that cannot be filled is not P&L.

7. **Three execution scenarios are reported, always.** Every backtest produces results under (a) **base case**: synthetic bid/ask conservative fills, (b) **optimistic sensitivity**: midpoint or calibrated price-improvement fills, (c) **stressed sensitivity**: synthetic ask + slippage. The base case is the headline. The other two bracket model risk.

---

## 2. Product universe

Two products, kept rigorously separate. Mixing them is the single largest source of pricing error flagged in the validation memo.

### 2.1 Product specifications

| Dimension | `VIXIndexOption` | `VXFutureOption` |
|---|---|---|
| Venue | Cboe Options Exchange | Cboe Futures Exchange (CFE) |
| Underlying for Black-76 | VIX forward for the option's settlement date | The specific VX futures contract this option delivers into |
| Exercise style | European | European |
| Settlement style | Cash-settled to VRO / SOQ | Physically settled into one VX futures contract |
| Option multiplier | `$100` per VIX point | Determined by underlying VX exposure — not a flat `$100` |
| Underlying multiplier (for delta hedging) | n/a (cash settlement) | `$1000` per VX futures contract |
| Settlement event timestamp | Wednesday-morning VRO print | VX future delivery into the next session |
| Expiration payoff | `max(VRO − K, 0) × 100` | Conversion into VX futures position at strike |
| Forward selection | Same-settlement-date VX future, or PCP-implied forward | The exact deliverable VX futures contract |
| Greek dollar scaling | `100 × Δ_B76` | Convert to VX-futures equivalents: `(100 × Δ_B76) / 1000 = 0.1 × Δ_B76` |

### 2.2 Product class hierarchy

```python
class Product(ABC):
    """Abstract base. All pricing/settlement/P&L code dispatches on subclass.
    Mixing string ticker checks for product type is FORBIDDEN — every
    multiplier, settlement rule, and tick rule is a method on this class."""
    contract_root: str
    expiry: datetime
    settlement_event: datetime
    strike: float
    right: Literal['call', 'put']

    @abstractmethod
    def settlement_value(self, market: 'SettlementMarket') -> float:
        """Cash payoff (VIXIndexOption: max(VRO − K, 0) × 100) or
        post-exercise position value (VXFutureOption: into one VX future)."""

    @abstractmethod
    def option_multiplier(self) -> float:
        """Dollar multiplier per option point. Hard-coded per product subclass."""

    @abstractmethod
    def hedge_ratio_to_vx(self, delta_b76: float) -> float:
        """Convert Black-76 delta to VX futures contracts for hedging.
        Required to be a product method — prevents the validation-memo
        'factor of 10' hedging error."""


class VIXIndexOption(Product):
    """Cboe VIX index option. European. Cash-settled to SOQ/VRO. $100 multiplier.
    NEVER priced from spot VIX. Forward selected by ForwardSelector."""

    def option_multiplier(self) -> float:
        return 100.0

    def settlement_value(self, market: 'SettlementMarket') -> float:
        """Payoff = max(VRO − K, 0) × 100 for calls. VRO is the actual SOQ print
        for this expiry — NOT spot VIX close, NOT Tuesday VX future close,
        NOT theoretical Black-76 value."""
        ...

    def hedge_ratio_to_vx(self, delta_b76: float) -> float:
        """option dollar delta = 100 × Δ_B76; VX multiplier = 1000.
        Hedge = (100 × Δ_B76) / 1000 = 0.1 × Δ_B76 VX contracts per option."""
        return 0.1 * delta_b76


class VXFutureOption(Product):
    """CFE option on VX futures. European. Physically settled into ONE VX future.
    Underlying VX futures contract has $1000 multiplier."""
    deliverable_vx: 'VXFuture'

    def option_multiplier(self) -> float:
        return 1000.0  # via underlying VX exposure

    def settlement_value(self, market: 'SettlementMarket') -> float:
        """Cash equivalent of the resulting VX futures position at exercise.
        Do NOT cash-settle this product like a VIXIndexOption."""
        ...

    def hedge_ratio_to_vx(self, delta_b76: float) -> float:
        """Already on VX-future-equivalent footing."""
        return delta_b76
```

The product type is set at contract-mapping time from the chain source (Cboe options vs CFE options-on-futures) and is immutable thereafter. The contract-mapping layer raises if a row's source and product class don't agree.

### 2.3 Bull call spread as a typed pair

```python
@dataclass(frozen=True)
class BullCallSpread:
    """Construction-time enforcement: both legs MUST be the same product type
    and MUST share the same expiry/settlement event. Mixing a VIXIndexOption
    long leg with a VXFutureOption short leg raises TypeError at __post_init__."""
    long_leg: Product   # lower strike call
    short_leg: Product  # higher strike call

    def __post_init__(self) -> None:
        if type(self.long_leg) is not type(self.short_leg):
            raise TypeError("Spread legs must be the same Product subclass.")
        if self.long_leg.settlement_event != self.short_leg.settlement_event:
            raise ValueError("Spread legs must share settlement_event.")
        if self.long_leg.strike >= self.short_leg.strike:
            raise ValueError("Bull call: long strike must be below short strike.")
```

---

## 3. Regime classification framework

The validation memo flagged three independent failure modes here: HMM transition-matrix inconsistency, full-sample 2025 thresholds, and smoothed-vs-filtered probability conflation. Each is fixed at the architecture level, not by documentation.

### 3.1 Strict causal contract

```python
@dataclass(frozen=True)
class RegimeSignal:
    """Output of the regime engine. ALL fields are functions of data with
    timestamp <= as_of. The as_of_inputs map is audited by the
    FeatureAvailability validator on every signal generation."""
    as_of: datetime                        # decision timestamp; action is AFTER this
    filtered_probs: np.ndarray             # P(S_t | y_{1:t}), shape (n_states,)
    state_label: int                       # argmax of filtered_probs, label-stable
    curve_features: dict[str, float]       # signed-normalized contango features
    hypothesis_tag: Literal['contrarian_tail',
                            'breakout_momentum',
                            'curve_normalization']
    as_of_inputs: dict[str, datetime]      # per-input as-of timestamps; audited


class RegimeClassifier(ABC):
    """ABC for regime classifiers. Implementations MUST be strictly causal.

    Hard rule: predict_filtered(t) may only consume data with timestamp <= t.
    Smoothed probabilities P(S_t | y_{1:T}) and full-sample Viterbi paths
    are FORBIDDEN as production signals — they are not exposed through any
    public method on this class. Implementations may compute smoothed paths
    internally for diagnostics but must not return them.
    """

    @abstractmethod
    def fit_walk_forward(
        self,
        history: 'FeaturePanel',
        as_of: datetime,
    ) -> 'FittedRegime':
        """Fit using only data with timestamp <= as_of. The validator inspects
        the panel slice taken and raises LookaheadError on any violation."""

    @abstractmethod
    def predict_filtered(
        self,
        fitted: 'FittedRegime',
        as_of: datetime,
    ) -> RegimeSignal:
        """Returns filtered probability P(S_t | y_{1:t}) ONLY. Must not
        invoke any internal smoothing routine. Unit-tested by injecting a
        future-stamped spike into the panel and asserting the filtered
        probability for as_of is unchanged."""
```

### 3.2 HMM specification (corrected)

The validation memo identified that the source backtester's published transition matrix and stated stationary distribution are mutually inconsistent (`p_low ≈ 0.745`, not `≈ 0.50`). The `HMMSpec` requires the user to declare exactly one of these and computes the other:

```python
@dataclass
class HMMSpec:
    """Either transition_matrix OR stationary_distribution must be supplied;
    the other is computed from it. If BOTH are supplied, consistency is checked
    and HMMSpecificationError is raised on mismatch. The validation-memo
    failure mode (declaring both, inconsistent) is impossible by construction."""
    n_states: int
    transition_matrix: np.ndarray | None       # row-stochastic; P[i,j] = P(S_{t+1}=j | S_t=i)
    stationary_distribution: np.ndarray | None
    state_label_rule: Literal['by_emission_mean', 'by_emission_variance']
    emission_family: Literal['gaussian', 'student_t']

    def validate(self) -> None:
        """Asserts each row of transition_matrix sums to 1.0; computes implied
        stationary distribution; checks consistency with declared stationary
        distribution if supplied. Asserts state_label_rule produces a
        deterministic ordering across walk-forward refits."""
        ...
```

State labeling stability across refits is enforced by `state_label_rule` — without it, the "low-vol" state can swap indices between adjacent walk-forward fits and corrupt the signal series silently. Default rule: `by_emission_variance` (low-vol state always has the lower-variance emission).

### 3.3 Walk-forward calibration

```python
class WalkForwardRegimeFitter:
    """Refits the HMM at a configurable cadence using only data through t-1.
    Persists per-(as_of) HMM parameters for audit and replay.

    The 'full-sample 2025 thresholds' failure mode is impossible because
    every percentile, scaler, and threshold the strategy depends on is
    derived from a fit object that itself was constrained to (:as_of)."""
    cadence: Literal['daily', 'weekly', 'monthly']
    lookback: timedelta                        # rolling window length
    refit_log: 'WalkForwardRefitLog'           # one row per refit, archived

    def refit_at(self, as_of: datetime) -> 'FittedRegime':
        """Slices history to (as_of − lookback, as_of), refits HMM, applies
        state_label_rule, persists. LookaheadError if any data with
        timestamp >= as_of leaks into the slice."""
        ...
```

### 3.4 Term-structure sign convention (memo amber issue)

A single canonical convention is enforced at feature-construction time:

```python
def curve_slope(F_near: float, F_far: float) -> float:
    """Returns F_far / F_near − 1. POSITIVE means contango; NEGATIVE means
    backwardation. Enforced for M1/M2, 30D/182D, and any interpolated
    futures-curve feature. A unit test validates against a known contango
    snapshot AND a known backwardation snapshot; both must produce the
    expected sign."""
    return F_far / F_near - 1.0
```

### 3.5 Strategy hypothesis declaration

The memo flagged that "low-vol filter" and "high-vol entry" describe different strategies. The architecture forces the user to declare one before signal generation:

```python
@dataclass(frozen=True)
class StrategyHypothesis:
    """Declared once at strategy construction. Drives signal logic, expected
    holding period, exit rules, and the regime/curve filter functions.
    Comparing hypotheses requires SEPARATE backtest runs, never a flag
    flipped mid-run."""
    name: Literal['contrarian_tail', 'breakout_momentum', 'curve_normalization']
    entry_regime_filter: Callable[[RegimeSignal], bool]
    entry_curve_filter: Callable[[dict[str, float]], bool]
    expected_holding_period: timedelta
```

The hypothesis tag is logged with every backtest run.

---

## 4. Pricing & Greeks framework

Black-76 with strict input validation. The pricer never sees a string ticker — it sees a `Product`, a `Forward`, a `LegIV`, and a minute-level timestamp.

### 4.1 Forward selection

```python
class ForwardSelector:
    """Selects the Black-76 forward input. Hierarchy:
       1. Exact same-settlement-date VX future (preferred).
       2. Put-call parity implied forward from same-expiry options.
       3. Term-structure interpolation (FALLBACK ONLY, sets model_risk_flag).
    Selecting from spot VIX is FORBIDDEN and raises ForwardSelectionError."""

    def select(
        self,
        product: Product,
        market: 'OptionsMarketSnapshot',
        as_of: datetime,
    ) -> 'Forward':
        """Returns Forward with selection_method tag for audit.
        For VXFutureOption, only branch (1) is valid — the deliverable VX
        is the forward; PCP and interpolation are unreachable."""
        ...


@dataclass(frozen=True)
class Forward:
    value: float
    selection_method: Literal['settlement_date_match',
                              'put_call_parity',
                              'interpolated']
    model_risk_flag: bool                  # True iff selection_method == 'interpolated'
    settlement_date: datetime
```

### 4.2 Time-to-expiry

```python
def minutes_to_settlement(as_of: datetime, settlement_event: datetime) -> float:
    """Calendar-minute time-to-expiry. Both timestamps MUST be tz-aware and
    in exchange time. Returns T = minutes / 525_600 for direct Black-76 use.
    No 252/365 ambiguity — calendar minutes match the IV-vendor convention
    used in this codebase."""
    if as_of.tzinfo is None or settlement_event.tzinfo is None:
        raise TimezoneError("Both timestamps must be tz-aware exchange time.")
    if as_of >= settlement_event:
        raise ExpiryError("Cannot price an option after its settlement event.")
    return (settlement_event - as_of).total_seconds() / 60.0 / 525_600.0
```

### 4.3 Leg-specific implied volatility

```python
class LegIVProvider(ABC):
    """Returns an implied volatility specific to (strike, expiry).
    Substituting VVIX as the leg IV is FORBIDDEN — the pricer raises
    FlatVolError if both legs of a non-zero-width spread are passed
    the same IV."""
    @abstractmethod
    def get(self, strike: float, expiry: datetime, as_of: datetime) -> float:
        """Implied vol for this exact strike and expiry. Sourced from the
        vendor option chain (ChainIVProvider) or a calibrated surface
        (SurfaceIVProvider)."""


class ChainIVProvider(LegIVProvider):
    """Reads vendor IV from the option chain row matching (strike, expiry).
    Phase-1 default."""


class SurfaceIVProvider(LegIVProvider):
    """Calibrates a surface (SVI / SABR) walk-forward and interpolates.
    Phase-7 extension."""
```

### 4.4 Pricer

```python
class Black76Pricer:
    """Black-76 European call/put pricer for VIX-family options.

    NEVER prices from spot VIX. ALWAYS uses a Forward from ForwardSelector.
    NEVER applies a second convexity adjustment to an observed VX future —
    market convexity is already embedded in that price.

    Returns TheoreticalPrice objects with is_executable=False. The fill
    engine rejects these objects at the type level — they are diagnostics
    and edge-bleed inputs, never P&L.
    """
    def price(
        self,
        product: Product,
        forward: Forward,
        leg_iv: float,
        as_of: datetime,
        risk_free_rate: float,
    ) -> 'TheoreticalPrice':
        """Returns Black-76 fair value + Greeks tagged with selection_method
        and minute-level T."""
        ...

    def price_spread(
        self,
        spread: BullCallSpread,
        forward: Forward,
        iv_long: float,
        iv_short: float,
        as_of: datetime,
        risk_free_rate: float,
    ) -> 'TheoreticalSpreadPrice':
        """Strike-specific IV per leg is REQUIRED. Raises FlatVolError if
        iv_long == iv_short on a non-zero-width spread (validation-memo
        VVIX-as-leg-IV defect)."""
        ...


@dataclass(frozen=True)
class TheoreticalPrice:
    value: float
    delta: float
    gamma: float
    vega: float
    theta: float
    forward_used: Forward
    iv_used: float
    T_minutes: float
    is_executable: bool = False             # SENTINEL — fill engine rejects on this
```

### 4.5 Greek dollar scaling

Dollar Greeks are computed by the *product*, not the pricer:

```python
class VIXIndexOption(Product):
    def dollar_delta(self, theoretical: TheoreticalPrice) -> float:
        """100 × Δ_B76 dollars per VIX point per option contract."""
        return 100.0 * theoretical.delta

    def vx_hedge_contracts(self, theoretical: TheoreticalPrice) -> float:
        """VX has $1000 multiplier; result is 0.1 × Δ_B76 VX contracts
        per option contract. The validation-memo 'factor of 10' hedging
        defect is impossible because this conversion is a product method,
        not a free-floating multiplier in the hedge code."""
        return (100.0 * theoretical.delta) / 1000.0
```

---

## 5. Execution simulation framework

The largest source of performance inflation in the source backtester. The architecture treats fills as the deliverable and theoretical prices as decoration.

### 5.1 Quote-level data contract

```python
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
    last_trade: float | None                # diagnostics ONLY, not fills
    last_trade_age_seconds: float | None
    is_locked: bool
    is_crossed: bool
    quote_age_seconds: float

    def is_stale(self, max_age_seconds: float) -> bool: ...
    def is_no_bid(self) -> bool: ...
```

### 5.2 Synthetic bid/ask for spreads

```python
class SyntheticSpreadQuote:
    """Constructs a debit-spread synthetic NBBO from two leg quotes.
    For a bull call spread (buy K1 @ ask, sell K2 @ bid; K1 < K2):
        synthetic_ask  (open debit)   = ask(K1) − bid(K2)
        synthetic_bid  (close credit) = bid(K1) − ask(K2)
    These are the conservative-base-case fill levels.
    Midpoint is NEVER used in the base case."""

    @staticmethod
    def open_debit_synthetic(long_q: OptionQuote, short_q: OptionQuote) -> float:
        return long_q.ask - short_q.bid

    @staticmethod
    def close_credit_synthetic(long_q: OptionQuote, short_q: OptionQuote) -> float:
        return long_q.bid - short_q.ask
```

### 5.3 Liquidity gates

```python
@dataclass(frozen=True)
class LiquidityGates:
    """Per-leg and per-spread acceptance criteria. A spread that fails any
    gate is REJECTED — it is not silently filled at a worse theoretical
    price. Rejections are reported as a first-class category in backtest
    output."""
    max_leg_spread_pct: float                # (ask − bid) / mid <= this
    min_leg_open_interest: int
    min_leg_volume_today: int
    min_displayed_size: int
    max_quote_age_seconds: float
    reject_locked_or_crossed: bool = True
    reject_no_bid_short_leg: bool = True     # short call with bid <= 0
    max_order_size_pct_of_displayed: float = 0.5
    max_order_size_pct_of_oi: float = 0.05


class LiquidityGateEngine:
    def evaluate(
        self,
        spread: BullCallSpread,
        long_q: OptionQuote,
        short_q: OptionQuote,
        order_size: int,
        gates: LiquidityGates,
    ) -> 'LiquidityGateResult':
        """Returns pass/fail with per-gate breakdown. The breakdown becomes
        a row in BacktestResults.rejected_orders for postmortem."""
        ...
```

### 5.4 Fill modes

Three fill modes, three reports — always:

```python
class FillMode(Enum):
    SYNTHETIC_BIDASK = 'synthetic_bidask'           # base case — headline
    MIDPOINT = 'midpoint'                            # optimistic sensitivity
    SYNTHETIC_PLUS_SLIPPAGE = 'synthetic_slip'       # stressed sensitivity


class FillEngine:
    """Converts a (spread, quotes, signal) into ExecutedFill or RejectedOrder.

    Default mode is SYNTHETIC_BIDASK. MIDPOINT requires an explicit
    `accept_midpoint_optimism=True` flag and emits a warning to the run log.
    TheoreticalPrice instances are rejected at the type level (is_executable
    is False) — they cannot be fills.

    The decision_timestamp argument is checked against the quote timestamp:
    the next eligible quote AFTER decision_timestamp is used. Same-minute
    fills raise LookaheadError. There is no flag to disable this check.
    """
    def attempt_fill(
        self,
        spread: BullCallSpread,
        long_q: OptionQuote,
        short_q: OptionQuote,
        order_size: int,
        mode: FillMode,
        gates: LiquidityGates,
        decision_timestamp: datetime,
    ) -> 'ExecutedFill | RejectedOrder':
        ...


@dataclass(frozen=True)
class ExecutedFill:
    timestamp: datetime
    spread: BullCallSpread
    debit_per_spread: float
    size: int
    fill_mode: FillMode
    long_leg_fill: float
    short_leg_fill: float
    tick_rounded: bool                       # per-product tick rule applied
    fees_per_spread: float


@dataclass(frozen=True)
class RejectedOrder:
    timestamp: datetime
    spread: BullCallSpread
    reason: Literal['no_bid_short', 'stale_quote', 'gate_fail',
                    'locked', 'crossed', 'tick_invalid', 'session_closed']
    detail: dict
```

### 5.5 Tick rounding and fees

Per-product tick rules and fee schedules are class attributes on the product, not free-floating constants:

```python
class VIXIndexOption(Product):
    TICK_RULES: ClassVar['TickRule']         # versioned, audited per Cboe spec
    FEE_PER_CONTRACT: ClassVar[float]

class VXFutureOption(Product):
    TICK_RULES: ClassVar['TickRule']         # versioned, audited per CFE spec
    FEE_PER_CONTRACT: ClassVar[float]
```

Every simulated fill is validated against the product's tick rule before being accepted as `ExecutedFill`. A theoretical price that doesn't round to a valid tick produces `RejectedOrder(reason='tick_invalid')`.

### 5.6 Forced Tuesday exit vs SOQ settlement

```python
class ExitPolicy(Enum):
    FORCED_TUESDAY_LIQUIDATION = 'forced_tuesday'    # exit before SOQ window
    HOLD_TO_SETTLEMENT = 'hold_to_settlement'        # accept SOQ/VRO outcome


class ExitEngine:
    """Forced-Tuesday exit reduces SOQ risk but is NOT costless. Liquidity
    can deteriorate near final trading hours; far-OTM short legs can go
    no-bid. HOLD_TO_SETTLEMENT uses ACTUAL historical VRO prints — never
    spot VIX close, never Tuesday VX future close, never theoretical
    Black-76 value. FailedExit is a first-class outcome."""
    def execute_exit(
        self,
        position: 'OpenPosition',
        market: 'OptionsMarketSnapshot',
        policy: ExitPolicy,
    ) -> 'ExecutedFill | SettlementOutcome | FailedExit':
        ...
```

`FailedExit` is reported separately. A forced-Tuesday liquidation can fail to fill — silent fills at theoretical are not allowed.

---

## 6. Data architecture

### 6.1 Source responsibilities

| Data | Source | Frequency | Notes |
|---|---|---|---|
| Spot VIX | Cboe historical / vendor | Daily / 1-min | Used for diagnostics & state features only — NEVER as Black-76 input |
| VX futures (all listed maturities) | Cboe / CFE historical | 1-min OHLC + quotes | Settlement-date table is the join key, not generic month numbers |
| VIX index options chain | Cboe DataShop / OPRA | 1-min NBBO quotes | NBBO required; trade prints not used as fills |
| VX options-on-futures chain | Cboe DataShop / CFE | 1-min NBBO quotes | Separate ingestion path from VIX index options |
| VRO / SOQ history | Cboe official | Per-expiry | Required for `HOLD_TO_SETTLEMENT` P&L |
| Risk-free rate | FRED `DGS3MO` / OIS | Daily | Black-76 discounting |
| VX expiry calendar | Cboe official | Static + maintenance | Drives settlement-date matching |

### 6.2 Internal data objects

```python
@dataclass(frozen=True)
class VIXSnapshot:
    timestamp: datetime
    vix_spot: float                           # diagnostic only — not a Black-76 input
    vx_curve: dict[str, float]                 # {settlement_date: future_price}
    options_chain: 'OptionsMarketSnapshot'
    risk_free_rate: float


@dataclass(frozen=True)
class OptionsMarketSnapshot:
    timestamp: datetime
    product_type: Literal['vix_index_option', 'vx_future_option']
    quotes: dict[str, OptionQuote]             # contract_id -> quote
    iv_surface: 'IVSurface | None'


@dataclass(frozen=True)
class FeaturePanel:
    """Daily-frequency panel for regime classification. Aligned to NY close.
    Every column carries a per-row as_of timestamp for the FeatureAvailability
    validator — no column is exempt."""
    dates: pd.DatetimeIndex
    features: pd.DataFrame
    as_of_map: dict[str, pd.Series]            # column -> per-row as_of timestamp
```

### 6.3 Vintage tagging and audit

Every raw pull writes to `data/raw/{source}/{product}/{vintage}.parquet` with the pull timestamp embedded. The `FeatureAvailability` validator reads from this layer and computes the as-of map automatically. Replays use the vintage tag to reconstruct exactly what was knowable at any historical decision time.

### 6.4 The release-boundary rule

For backtest replay: a feature with vintage timestamp `v` may be used in a decision with `as_of >= v`. Anything else is `LookaheadError`. The validator runs on every backtest start and on every walk-forward refit — it does not depend on developer discipline.

---

## 7. Strategy & signal layer

The strategy is the thinnest layer in the system. It composes regime + product selection + spread construction + risk sizing + exit policy.

### 7.1 Strategy interface

```python
class VIXBullCallSpreadStrategy:
    """Composes regime classifier, spread selector, sizing, and exit policy.

    The strategy hypothesis is declared at construction and is immutable
    for the life of the strategy object. Comparing hypotheses requires
    constructing separate strategies and running separate backtests."""
    hypothesis: StrategyHypothesis
    regime: RegimeClassifier
    spread_selector: 'SpreadSelector'
    sizer: 'PositionSizer'
    exit_policy: ExitPolicy
    gates: LiquidityGates

    def evaluate(
        self,
        market: VIXSnapshot,
        as_of: datetime,
    ) -> 'StrategyDecision':
        """Returns enter / hold / exit / skip with full provenance.
        The decision is for action AFTER as_of — never AT as_of.
        Same-bar fills are rejected by FillEngine."""
        ...


class SpreadSelector:
    """Selects (long_strike, short_strike, expiry) given regime and curve.
    Strikes are constrained to executable rows in the chain — selecting a
    strike whose short leg is no-bid is rejected at the selector stage,
    not at the gate stage."""
    def select(
        self,
        market: VIXSnapshot,
        signal: RegimeSignal,
        as_of: datetime,
    ) -> BullCallSpread | None:
        ...
```

### 7.2 Signal-to-execution timing

The decision-to-fill timing rule is where look-ahead most often sneaks in:

1. Daily features build at NY close `t-1`.
2. `RegimeClassifier.predict_filtered(as_of=close_{t-1})` returns a `RegimeSignal`.
3. Strategy evaluates — produces `StrategyDecision` tagged `as_of=close_{t-1}`.
4. `FillEngine.attempt_fill` consumes the **next eligible quote minute after** `close_{t-1}`. Same-minute is `LookaheadError`.
5. P&L accrues from the executed fill timestamp forward.

A unit test injects a future-stamped quote into step 4 and asserts `LookaheadError`.

---

## 8. Backtest engine & evaluation

### 8.1 Walk-forward backtest

```python
class WalkForwardBacktest:
    """The only backtest mode in the codebase. Refits regime parameters at
    the configured cadence, replays signals walk-forward, accepts only
    ExecutedFills into P&L. Always runs all three fill modes."""
    strategy: VIXBullCallSpreadStrategy
    fill_engine: FillEngine
    fill_modes: list[FillMode]                 # always all three for reporting
    start: datetime
    end: datetime
    refit_cadence: Literal['daily', 'weekly', 'monthly']
    embargo: timedelta = timedelta(days=0)

    def run(self) -> 'BacktestResults':
        """Returns results bundle with three execution scenarios, full
        rejection log, regime audit, and forward-selection audit."""
        ...
```

### 8.2 Reporting structure

```python
@dataclass
class BacktestResults:
    base_case: 'ExecutionScenarioResult'        # synthetic bid/ask — headline
    optimistic: 'ExecutionScenarioResult'        # midpoint
    stressed: 'ExecutionScenarioResult'          # synthetic + slippage
    rejected_orders: pd.DataFrame                # every RejectedOrder, with reason
    regime_audit: 'RegimeAuditTrail'             # per-refit HMM diagnostics
    forward_selection_audit: pd.DataFrame        # per-trade selection method
    held_to_settlement: 'SettlementSensitivity | None'


@dataclass
class ExecutionScenarioResult:
    pnl_series: pd.Series
    equity_curve: pd.Series
    trade_log: pd.DataFrame                      # one row per ExecutedFill
    metrics: dict[str, float]                    # Sharpe, Sortino, max DD, hit rate
    realized_vs_theoretical: pd.DataFrame        # edge-bleed analysis
```

The rejected-orders count is reported in the headline summary alongside P&L. An attractive equity curve with a 60% rejection rate is not a strategy.

### 8.3 Diagnostic suite

- HMM stability per refit: state-label consistency, transition-matrix Frobenius distance between adjacent fits, stationary-distribution drift over the sample.
- Forward-selection breakdown: % of trades using settlement-date match vs PCP vs interpolated. Interpolated trades are flagged model-risk-tagged in the trade log.
- Edge bleed: theoretical Black-76 spread value vs executed debit, per trade. The distribution of bleed quantifies execution drag.
- Held-to-settlement sensitivity: re-runs with `HOLD_TO_SETTLEMENT` exit policy using actual historical VRO. Compared against forced-Tuesday exit.

---

## 9. Reporting & visualization

```
viz.equity_curves(results)                # 3 lines: base, optimistic, stressed
viz.regime_overlay(results)                # equity overlaid with filtered HMM probs
viz.rejection_heatmap(results)             # rejection reason × time × strike bucket
viz.edge_bleed(results)                    # (theoretical − executed) distribution
viz.forward_selection(results)             # method breakdown over time
viz.held_to_settlement_compare(results)    # forced-Tuesday vs SOQ-settled
viz.curve_state_calendar(results)          # contango/backwardation regime tape
```

The base-case chart is always reported first. Optimistic and stressed cases are not allowed to be the headline.

---

## 10. Codebase structure

```
vix_spread_backtester/
├── pyproject.toml
├── README.md
├── .env.example
│
├── config/
│   ├── strategy.yaml                     # hypothesis, gates, sizing
│   ├── regime.yaml                       # HMM spec, walk-forward cadence
│   ├── data_sources.yaml                 # vendor endpoints, paths
│   ├── execution.yaml                    # fill modes, tick rules, fees
│   └── backtest.yaml                     # period, refit cadence, embargo
│
├── data/                                 # gitignored
│   ├── raw/                              # immutable vintage-tagged pulls
│   │   ├── vix/{vintage}.parquet
│   │   ├── vx_futures/{vintage}.parquet
│   │   ├── vix_index_options/{vintage}/
│   │   ├── vx_future_options/{vintage}/
│   │   ├── vro_settlement/{vintage}.parquet
│   │   ├── rates/{vintage}.parquet
│   │   └── _manifest.jsonl
│   ├── processed/
│   └── vintages/
│
├── src/vix_spread/
│   ├── __init__.py
│   │
│   ├── products/
│   │   ├── __init__.py
│   │   ├── base.py                       # Product ABC
│   │   ├── vix_index_option.py           # VIXIndexOption: SOQ, $100
│   │   ├── vx_future_option.py           # VXFutureOption: physical, $1000
│   │   ├── vx_future.py                  # VXFuture (the underlying)
│   │   ├── spread.py                     # BullCallSpread typed pair
│   │   └── tick_rules.py                 # per-product tick + fee tables
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── base.py                       # BaseDataFetcher + caching
│   │   ├── vix_history.py                # spot VIX (diagnostic only)
│   │   ├── vx_futures.py                 # VX futures with settlement calendar
│   │   ├── vix_index_options.py          # Cboe options chain ingestion
│   │   ├── vx_future_options.py          # CFE options-on-futures chain
│   │   ├── vro_settlement.py             # historical SOQ/VRO prints
│   │   ├── rates.py                      # FRED / OIS
│   │   ├── expiry_calendar.py            # VX settlement-date table
│   │   ├── snapshot.py                   # VIXSnapshot / OptionsMarketSnapshot
│   │   ├── feature_panel.py              # FeaturePanel with as_of_map
│   │   ├── availability.py               # FeatureAvailability validator
│   │   └── cache.py                      # parquet vintage cache
│   │
│   ├── regime/
│   │   ├── __init__.py
│   │   ├── base.py                       # RegimeClassifier ABC, RegimeSignal
│   │   ├── hmm_spec.py                   # HMMSpec with consistency validator
│   │   ├── hmm_filter.py                 # filtered-prob HMM, no smoothing API
│   │   ├── walk_forward.py               # WalkForwardRegimeFitter + log
│   │   ├── curve_features.py             # signed contango features
│   │   └── state_labeling.py             # stable label rule across refits
│   │
│   ├── pricing/
│   │   ├── __init__.py
│   │   ├── black76.py                    # Black76Pricer
│   │   ├── forward_selector.py           # settlement-date / PCP / interp
│   │   ├── leg_iv.py                     # ChainIVProvider, SurfaceIVProvider
│   │   ├── time_to_expiry.py             # minute-level T calculation
│   │   ├── theoretical.py                # TheoreticalPrice (is_executable=False)
│   │   └── pcp.py                        # put-call parity implied forward
│   │
│   ├── execution/
│   │   ├── __init__.py
│   │   ├── quote.py                      # OptionQuote, staleness, no-bid
│   │   ├── synthetic_quote.py            # SyntheticSpreadQuote
│   │   ├── liquidity_gates.py            # LiquidityGates + engine
│   │   ├── fill_engine.py                # FillEngine, ExecutedFill, RejectedOrder
│   │   ├── fill_modes.py                 # FillMode enum + handlers
│   │   ├── tick_rounding.py              # per-product tick application
│   │   ├── exit_engine.py                # ExitPolicy: forced-Tue vs SOQ
│   │   └── settlement.py                 # SettlementOutcome with VRO
│   │
│   ├── strategy/
│   │   ├── __init__.py
│   │   ├── hypothesis.py                 # StrategyHypothesis
│   │   ├── strategy.py                   # VIXBullCallSpreadStrategy
│   │   ├── spread_selector.py            # SpreadSelector
│   │   ├── sizing.py                     # PositionSizer
│   │   └── timing.py                     # signal-to-execution timing rules
│   │
│   ├── backtest/
│   │   ├── __init__.py
│   │   ├── walk_forward.py               # WalkForwardBacktest
│   │   ├── results.py                    # BacktestResults bundle
│   │   ├── metrics.py                    # Sharpe, Sortino, DD, hit rate
│   │   ├── audit.py                      # RegimeAuditTrail, forward audit
│   │   └── edge_bleed.py                 # theoretical-vs-executed distribution
│   │
│   ├── viz/
│   │   ├── __init__.py
│   │   ├── equity_curves.py              # 3-scenario equity overlay
│   │   ├── regime_overlay.py             # equity + filtered probs
│   │   ├── rejection_heatmap.py
│   │   ├── edge_bleed.py
│   │   ├── forward_selection.py
│   │   └── settlement_compare.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── time.py                       # tz-aware exchange time helpers
│       ├── logging.py                    # structured run logs
│       └── errors.py                     # LookaheadError, ForwardSelectionError, ...
│
├── notebooks/
│   ├── 00_setup.ipynb
│   ├── 01_data_audit.ipynb
│   ├── 02_regime_calibration_walkforward.ipynb
│   ├── 03_pricing_validation.ipynb       # Black-76 vs synthetic, PCP recovery
│   ├── 04_execution_realism.ipynb        # synthetic vs midpoint vs stressed
│   ├── 05_backtest_2025_regime.ipynb     # the headline run
│   └── 06_held_to_settlement_sensitivity.ipynb
│
├── scripts/
│   ├── pull_data.py
│   ├── build_panels.py
│   ├── refit_regime_walkforward.py
│   ├── run_backtest.py
│   └── generate_report.py
│
└── tests/
    ├── conftest.py
    ├── test_products_separation.py        # mixing product types raises
    ├── test_regime_causality.py           # future-spike injection, no leak
    ├── test_hmm_consistency.py            # transition <-> stationary check
    ├── test_pricing_no_spot_vix.py        # asserts ForwardSelectionError
    ├── test_pricing_minute_T.py
    ├── test_pcp_implied_forward.py
    ├── test_fill_engine_no_midpoint_default.py
    ├── test_fill_engine_rejects_theoretical.py
    ├── test_liquidity_gates.py
    ├── test_tick_rounding.py
    ├── test_walk_forward_no_lookahead.py
    └── fixtures/
        ├── synthetic_chains/
        ├── synthetic_vx_curve/
        └── soq_calendar/
```

### 10.1 Config example: `config/strategy.yaml`

```yaml
hypothesis:
  name: contrarian_tail
  entry_regime_filter: low_vol_with_min_duration
  entry_curve_filter: hyper_contango_30d182d_below_minus_3pct
  expected_holding_period_days: 21

spread:
  product_type: vix_index_option         # VIXIndexOption (SOQ, $100) — not VX-on-futures
  long_strike_rule: atm_plus_n_points
  short_strike_rule: long_strike_plus_n_points
  long_n: 2
  short_n: 8
  expiry_rule: dte_between_30_and_60

sizing:
  method: fixed_risk
  risk_per_trade_pct: 0.005
  max_concurrent_spreads: 1

exit:
  policy: forced_tuesday_liquidation
  also_report_held_to_settlement: true   # sensitivity run, requires VRO history

gates:
  max_leg_spread_pct: 0.15
  min_leg_open_interest: 500
  min_leg_volume_today: 50
  min_displayed_size: 5
  max_quote_age_seconds: 30
  reject_locked_or_crossed: true
  reject_no_bid_short_leg: true
  max_order_size_pct_of_displayed: 0.5
  max_order_size_pct_of_oi: 0.05
```

### 10.2 Config example: `config/regime.yaml`

```yaml
hmm:
  n_states: 2
  emission_family: gaussian
  state_label_rule: by_emission_variance     # low-vol = low-variance emission, refit-stable
  # Specify ONE of these. The other is computed and validated.
  transition_matrix:
    - [0.960, 0.040]
    - [0.117, 0.883]
  stationary_distribution: null              # derived; expect ~[0.745, 0.255]

walk_forward:
  cadence: weekly
  lookback_days: 1260                        # ~5 years rolling
  embargo_days: 0
  persist_refit_log: true

features:
  curve_slope_convention: positive_means_contango
  use_filtered_only: true
  forbid_smoothed: true                      # raises LookaheadError if used
```

### 10.3 Config example: `config/execution.yaml`

```yaml
fill_modes:
  base_case: synthetic_bidask
  also_report:
    - midpoint                               # optimistic sensitivity
    - synthetic_plus_slippage                # stressed sensitivity

slippage:
  per_leg_ticks: 1
  apply_to_short_leg_only: false

tick_rules:
  vix_index_option: cboe_vix_options_v1      # versioned, audited
  vx_future_option: cfe_vx_options_v1

fees:
  vix_index_option_per_contract: 0.65        # illustrative; load from venue
  vx_future_option_per_contract: 1.50

next_minute_rule:
  decision_to_fill_lag_minutes: 1            # fill at minute t+1 after signal at t
  reject_same_minute_fills: true
```

---

## 11. Implementation phases

Ordered to deliver the validation-memo promotion gate first.

### Phase 1 — Foundations & data audit (week 1)
- Repo, `pyproject.toml`, CI with the test suite from §10 wired in from day one.
- Vintage-tagged ingestion for VIX spot, VX futures, VRO/SOQ, expiry calendar, rates.
- Option chain ingestion for both `VIXIndexOption` and `VXFutureOption` in **separate** ingestion paths with separate schemas.
- `FeaturePanel` with as-of map; `FeatureAvailability` validator.
- Notebook 01: data audit confirming row counts, settlement-date join integrity, no missing VRO prints.

**Deliverable:** all data on disk with vintage tags; validator runs clean on a synthetic feature pull.

### Phase 2 — Product layer & pricing (week 2)
- `Product` ABC, `VIXIndexOption`, `VXFutureOption`, `BullCallSpread` with construction-time type checks.
- `Black76Pricer` with strict-no-spot-VIX contract.
- `ForwardSelector` with three-tier hierarchy (settlement-date / PCP / interpolated).
- `LegIVProvider` interface with `ChainIVProvider` first.
- Tests: mixing product types raises; pricing from spot VIX raises; minute-level T arithmetic; PCP forward recovers same value as observed VX future on liquid expiries.

**Deliverable:** `Black76Pricer.price_spread(...)` returns sane Greeks for both product types on a sample chain.

### Phase 3 — Regime layer with strict causality (week 3)
- `HMMSpec` with consistency validator (transition ↔ stationary).
- `WalkForwardRegimeFitter` with refit log.
- `RegimeClassifier` exposing only filtered probs.
- Curve-feature module with signed-contango convention.
- Tests: future-spike injection asserts no leak; the validation-memo's published transition matrix raises until the inconsistency is corrected.

**Deliverable:** `predict_filtered(as_of=t)` returns `RegimeSignal` with full as-of audit trail.

### Phase 4 — Execution layer (week 4)
- `OptionQuote`, `SyntheticSpreadQuote`, `LiquidityGateEngine`.
- `FillEngine` with `SYNTHETIC_BIDASK` as base; `MIDPOINT` and `SYNTHETIC_PLUS_SLIPPAGE` opt-in.
- Tick rounding per product; fee schedules per product.
- `ExitEngine` with `FORCED_TUESDAY_LIQUIDATION` and `HOLD_TO_SETTLEMENT`.
- Tests: theoretical price rejected as fill; same-minute fills rejected; midpoint requires explicit flag.

**Deliverable:** `FillEngine.attempt_fill(...)` returns `ExecutedFill | RejectedOrder` end-to-end.

### Phase 5 — Strategy & backtest (week 5)
- `StrategyHypothesis`, `VIXBullCallSpreadStrategy`, `SpreadSelector`, `PositionSizer`.
- `WalkForwardBacktest` running all three fill modes per run.
- `BacktestResults` bundle with rejection log, regime audit, forward-selection audit.

**Deliverable:** end-to-end run on a 6-month sample window producing the three-scenario report.

### Phase 6 — Reporting, validation runs, 2025 headline (week 6)
- All visualization modules from §9.
- Headline run: 2010–2026 walk-forward backtest, with the 2025 low-vol/hyper-contango regime in-sample at signal time only when the walk-forward window includes it.
- Held-to-settlement sensitivity run.
- Edge-bleed report.
- Promotion-gate checklist: every validation-memo RED item has a matching green test.

**Deliverable:** signed-off backtest report meeting the validation-memo promotion gate.

### Phase 7+ — Extensions (deferred)
- `SurfaceIVProvider` (SVI / SABR) replacing chain-row IV.
- Complex-order-book fill model (where exchange data permits) replacing synthetic bid/ask as the base case.
- Adaptive position sizing keyed to regime confidence.
- Same architecture extensible to other VIX spread structures (calendars, butterflies, ratios) by subclassing `Spread`.

---

## 12. Critical design decisions & trade-offs

### 12.1 Why product type is dispatched, not branched

Branching pricing/settlement/multiplier on string flags is exactly how the source backtester drifted into conflating VIX index options and VX options-on-futures. Making `Product` a sealed ABC with two subclasses forces every code path that depends on settlement style or multiplier to be a method on the product. New code that reaches for an `if ticker.startswith(...)` cannot pass review because the field doesn't exist.

### 12.2 Why theoretical prices have `is_executable: bool = False`

The validation memo flagged that fair value should never be a fill. The cheapest enforcement is at the type system: `TheoreticalPrice` carries the sentinel, and `FillEngine` rejects on it. A developer who tries to short-circuit the synthetic-quote pipeline and feed in a Black-76 value gets a runtime error in the unit tests, not a silently optimistic backtest in production.

### 12.3 Why filtered-only is structural, not documented

"Use filtered probabilities, not smoothed" as a comment is a known anti-pattern. The architecture exposes only `predict_filtered(as_of)` on `RegimeClassifier`. Internal HMM implementations may compute smoothed probabilities for diagnostics, but the public interface does not return them. The unit test that injects a future spike and asserts no change in the filtered output is the contract.

### 12.4 Why three fill modes always run

Reporting only the base case underweights model risk; reporting only the midpoint inflates performance. Running all three on every backtest, with the base case as the headline and the others as bracketing sensitivities, is the cheapest defense against narrative cherry-picking. The cost is ~3× backtest compute, negligible at this scale.

### 12.5 Why the next-minute fill rule is enforced, not assumed

Same-minute fills sneak in any time the developer iterates on a signal-generation function in a notebook. The fill engine's `decision_timestamp` parameter is checked against the quote timestamp at fill time. Same-minute quote → `LookaheadError`. There is no flag to disable it.

### 12.6 Why HMM state-labeling stability is enforced

Across walk-forward refits, an HMM can swap state indices — what was state 0 at fit `k` becomes state 1 at fit `k+1`. The signal series then flips silently and the strategy looks like noise. `state_label_rule` (e.g., `by_emission_variance` → low-vol state always has the lower-variance emission) is applied after every refit and the assignment is logged for audit.

### 12.7 Why VRO is required, not optional

`HOLD_TO_SETTLEMENT` exit P&L cannot be approximated by spot VIX or Tuesday VX close. The data layer treats VRO history as a required dependency for any backtest including that exit policy, and the backtest fails at startup if VRO is missing for an in-sample expiry. No silent fallbacks.

### 12.8 Lucas-critique reminder

The 2025 low-vol/hyper-contango regime was structurally specific (post-April-shock snap-back, a particular term-premium environment, a particular dealer positioning regime). Even with walk-forward calibration, strategy *hyperparameters* (strike spacing, holding period, gate thresholds) might still be chosen with knowledge of how 2025 unfolded. Mitigation: report performance per regime epoch separately, and keep strategy hyperparameter changes in a versioned log so post-2025 tweaks don't pollute pre-2025 evaluations.

### 12.9 Simplicity discipline (per `CLAUDE.md`)

The architecture refuses speculative abstraction. There is no "pluggable settlement engine" for products that don't exist; there are exactly two concrete `Product` subclasses. There is no generic "spread" container for spreads we aren't trading; there is `BullCallSpread`, with construction-time invariants. New products and new spread structures are added when they are needed, by subclassing — not pre-built into v1.

---

## 13. Dependencies

```
numpy pandas scipy matplotlib seaborn       # core
hmmlearn                                    # HMM with filtered-only wrapper
scikit-learn                                # walk-forward CV scaffolding
statsmodels                                 # diagnostics, HAC SE
arch                                        # bootstrap, MCS
pyarrow                                     # parquet vintages
pydantic                                    # config validation
typer                                       # CLI
pyyaml
pytz tzdata                                 # tz-aware exchange time
fredapi                                     # rates
sodapy / requests                           # vendor APIs
pytest pytest-cov hypothesis                # testing
```

Notable exclusions: no library that exposes a smoothed-probability API as the default (we wrap `hmmlearn` to expose only filtering through `RegimeClassifier`). No vendor SDK that returns last-trade prices as fills without quote context.

---

## 14. What to build first

A two-day spike to prove the four hard constraints are enforceable end-to-end before the full build:

1. **Day 1 morning.** `Product` ABC + two subclasses. `BullCallSpread` with construction-time type check. Test: mixing `VIXIndexOption` and `VXFutureOption` legs raises.
2. **Day 1 afternoon.** `Black76Pricer` + `ForwardSelector` (settlement-date branch only). Test: pricing a `VIXIndexOption` from spot VIX raises `ForwardSelectionError`.
3. **Day 2 morning.** `RegimeClassifier` ABC + minimal HMM impl exposing only filtered probs. Test: future-spike injection produces no change in filtered output.
4. **Day 2 afternoon.** `FillEngine` with `SYNTHETIC_BIDASK` only. Test: passing a `TheoreticalPrice` where an `OptionQuote` is expected raises at the type level.

If the four tests are green, the architecture supports the validation-memo promotion gate. Everything after is filling in.

---

## 15. Open questions to confirm before coding

1. **Product scope.** Is the headline strategy on `VIXIndexOption` only, or do we need `VXFutureOption` in phase 1 too? Default: `VIXIndexOption` headline; `VXFutureOption` scaffold-only in phase 1.
2. **Quote data vendor.** OPRA full feed vs Cboe DataShop NBBO snapshots vs vendor 1-minute bars. Affects whether synthetic bid/ask is exact or approximate. Confirm before phase 4.
3. **IV source.** Chain-row IV from vendor vs in-house surface calibration. Phase 1 = `ChainIVProvider`; phase 7 = `SurfaceIVProvider`.
4. **VRO history coverage.** Confirm we have VRO prints for every expiry in the backtest window. Required for `HOLD_TO_SETTLEMENT`.
5. **Strategy hypothesis at kickoff.** Three options from the validation memo. Default: `contrarian_tail`, entered during low-vol/hyper-contango, exiting Tuesday before SOQ.
6. **Refit cadence.** Daily / weekly / monthly walk-forward refit. Default: weekly — balances HMM parameter stability against responsiveness.
7. **Position sizing.** Fixed risk per spread vs Kelly-ish vs filtered-prob-scaled. Default: fixed risk per spread (simplest, most auditable).

---

## Summary

The architecture is organized around four hard constraints from the validation memo, each enforced *structurally* rather than by convention. Regime classification exposes only filtered probabilities — smoothed paths are not in the public interface. `VIXIndexOption` and `VXFutureOption` are sealed product types that dispatch all settlement and multiplier logic on subclass — there is no string-ticker branching. Black-76 takes a `Forward` selected by settlement-date match or PCP, with minute-level calendar `T` — spot VIX cannot be a Black-76 input. The fill engine accepts only `OptionQuote`-derived synthetic bid/ask fills — `TheoreticalPrice` carries `is_executable=False` and is rejected at the type level. Walk-forward is the only backtest mode, and three execution scenarios (base / optimistic / stressed) are reported on every run. The 2-day spike in §14 proves the constraints are enforceable before the full 6-week build kicks off.
