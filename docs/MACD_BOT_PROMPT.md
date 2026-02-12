# MACD Swing Trading Bot - Implementation Prompt

Build a fully automated swing trading bot that trades US stocks via Alpaca Markets API using a proven MACD strategy with a 71% win rate.

---

## Project Overview

**Goal:** Create a Python-based trading bot that implements a 3-system MACD trading strategy for swing trading US equities.

**Key Requirements:**
- Fully automated execution (no human approval needed)
- Alpaca Markets integration for trading
- Swing trading style (Daily + 4H + 1H timeframes)
- Personal use (single user, not multi-tenant)

---

## Technology Stack

```
Language: Python 3.11+
Broker API: alpaca-trade-api
Data Processing: pandas, numpy
Technical Analysis: pandas-ta (or TA-Lib)
Database: SQLite (for trade history and state)
Scheduling: APScheduler
Notifications: Slack webhook (optional)
```

---

## Project Structure

Create this directory structure:

```
macd_bot/
├── config/
│   ├── __init__.py
│   └── settings.py          # Configuration parameters
├── core/
│   ├── __init__.py
│   ├── indicators.py        # MACD calculations
│   ├── signals.py           # Signal generation (3 systems)
│   ├── risk_manager.py      # Position sizing, stop-loss
│   └── trade_executor.py    # Order execution via Alpaca
├── data/
│   ├── __init__.py
│   └── market_data.py       # Data fetching from Alpaca
├── models/
│   ├── __init__.py
│   └── database.py          # SQLite models for trades
├── utils/
│   ├── __init__.py
│   ├── logger.py            # Logging setup
│   └── notifications.py     # Slack/Discord alerts
├── tests/
│   ├── __init__.py
│   ├── test_indicators.py
│   ├── test_signals.py
│   └── test_risk_manager.py
├── main.py                  # Entry point
├── requirements.txt
├── .env.example
└── README.md
```

---

## Configuration (config/settings.py)

```python
# MACD Parameters
MACD_FAST_PERIOD = 12
MACD_SLOW_PERIOD = 26
MACD_SIGNAL_PERIOD = 9
ZERO_LINE_BUFFER = 0.5  # Ignore crossovers within -0.5 to 0.5

# Timeframes for multi-timeframe analysis
TIMEFRAMES = {
    "higher": "1Day",    # Daily - determines bias
    "middle": "4Hour",   # 4H - confirms direction
    "lower": "1Hour"     # 1H - entry timing
}

# Risk Management
RISK_PER_TRADE = 0.02        # 2% risk per trade
MAX_POSITION_SIZE = 0.10     # 10% max single position
MAX_POSITIONS = 5            # Maximum concurrent positions
MAX_PORTFOLIO_EXPOSURE = 0.50  # 50% max total exposure
MIN_REWARD_RATIO = 1.5       # Minimum risk/reward ratio
ATR_PERIOD = 14              # ATR calculation period
ATR_MULTIPLIER = 2.0         # Stop-loss = 2x ATR

# Trading Limits
MAX_DRAWDOWN = 0.15          # 15% max drawdown - halt trading
MIN_PRICE = 10.0             # Minimum stock price
MIN_VOLUME = 1_000_000       # Minimum avg daily volume

# Scheduling (Eastern Time)
SCAN_INTERVAL_MINUTES = 5    # Check for signals every 5 min
POSITION_CHECK_SECONDS = 60  # Monitor positions every 1 min
PREMARKET_SCAN_TIME = "08:30"
EOD_PROCESSING_TIME = "16:00"

# Alpaca
PAPER_TRADING = True  # Start with paper trading!
```

---

## Core Trading Strategy Implementation

### System 1: Trend Trading Strategy

**File: core/signals.py**

```python
def check_system1_trend_signal(macd_data: dict) -> dict:
    """
    System 1: Trend Trading with Zero Line Filter

    RULES:
    1. Zero Line Filter (MANDATORY):
       - MACD > 0 → ONLY LONG trades allowed
       - MACD < 0 → ONLY SHORT trades allowed

    2. Crossover Entry:
       - LONG: MACD line crosses ABOVE signal line (while MACD > 0)
       - SHORT: MACD line crosses BELOW signal line (while MACD < 0)

    3. Momentum Filter:
       - IGNORE crossovers if MACD is between -0.5 and 0.5
       - This filters out weak, choppy signals

    Returns:
        {
            "signal": "LONG" | "SHORT" | None,
            "system": 1,
            "reason": str,
            "macd_value": float,
            "signal_value": float
        }
    """
    pass
```

### System 2: Reversal Strategy

```python
def check_system2_reversal_signal(price_data: pd.DataFrame, macd_data: dict) -> dict:
    """
    System 2: Reversal Strategy with Divergence Detection

    RULES:
    1. Divergence Detection:
       - BEARISH DIVERGENCE: Price makes Higher Highs, MACD makes Lower Highs
       - BULLISH DIVERGENCE: Price makes Lower Lows, MACD makes Higher Lows

    2. Histogram Confirmation Patterns:
       a) THE FLIP: First opposite-color bar after a series
          - Green bar after 5+ red bars = bullish
          - Red bar after 5+ green bars = bearish

       b) SHRINKING TOWER: Progressively smaller bars
          - Shrinking red bars = selling exhaustion
          - Shrinking green bars = buying exhaustion

    3. Entry Trigger:
       - Divergence PLUS histogram pattern = trade signal

    Returns:
        {
            "signal": "LONG" | "SHORT" | None,
            "system": 2,
            "divergence_type": "bullish" | "bearish" | None,
            "histogram_pattern": "flip" | "shrinking" | None,
            "reason": str
        }
    """
    pass
```

### System 3: Confirmation Strategy

```python
def check_system3_confirmation(
    daily_macd: dict,
    h4_macd: dict,
    h1_macd: dict,
    key_levels: list
) -> dict:
    """
    System 3: Multi-Timeframe Confirmation + Key Levels

    RULES:
    1. Triple Timeframe Stack:
       - Daily MACD determines overall BIAS (> 0 = bullish, < 0 = bearish)
       - 4H MACD must CONFIRM direction (crossover in same direction)
       - 1H MACD provides entry TIMING (crossover trigger)

       FOR LONG:
       - Daily MACD > 0 (bullish bias)
       - 4H shows bullish crossover or MACD > signal
       - 1H shows bullish crossover (entry trigger)

       FOR SHORT:
       - Daily MACD < 0 (bearish bias)
       - 4H shows bearish crossover or MACD < signal
       - 1H shows bearish crossover (entry trigger)

    2. Key Level Confluence (26% higher win rate!):
       - Prioritize signals at support/resistance levels
       - Look for crossovers near:
         * Previous swing highs/lows
         * Round numbers (50, 100, etc.)
         * Recent consolidation zones

    Returns:
        {
            "signal": "LONG" | "SHORT" | None,
            "system": 3,
            "timeframe_alignment": bool,
            "at_key_level": bool,
            "confidence": "high" | "medium" | "low",
            "reason": str
        }
    """
    pass
```

---

## Signal Generation Logic

```python
def generate_trade_signal(symbol: str) -> dict:
    """
    Main signal generation function that combines all 3 systems.

    PRIORITY ORDER:
    1. First check System 3 (multi-timeframe) - highest confidence
    2. Then check System 1 (trend) - core strategy
    3. Finally check System 2 (reversal) - counter-trend

    VALIDATION CHECKLIST:
    □ Market is open
    □ Symbol meets liquidity requirements (volume, spread)
    □ Not already in position for this symbol
    □ Position limits not exceeded
    □ Drawdown limit not exceeded
    □ Signal passes at least one system's criteria
    □ All three timeframes are aligned (for System 3)
    □ Zero line filter is satisfied (for System 1)

    Returns combined signal with confidence score.
    """
    pass
```

---

## Risk Management (core/risk_manager.py)

```python
class RiskManager:
    """
    Handles all risk-related calculations and validations.
    """

    def calculate_position_size(
        self,
        account_equity: float,
        entry_price: float,
        stop_loss_price: float
    ) -> int:
        """
        Calculate position size based on fixed risk per trade.

        Formula:
        risk_amount = account_equity * RISK_PER_TRADE
        risk_per_share = abs(entry_price - stop_loss_price)
        shares = risk_amount / risk_per_share

        Constraints:
        - Never exceed MAX_POSITION_SIZE of equity
        - Round down to whole shares
        - Minimum 1 share
        """
        pass

    def calculate_stop_loss(
        self,
        entry_price: float,
        atr: float,
        direction: str,
        swing_point: float = None
    ) -> float:
        """
        Calculate stop-loss price.

        Method 1: ATR-based (default)
        - LONG: stop = entry - (ATR * ATR_MULTIPLIER)
        - SHORT: stop = entry + (ATR * ATR_MULTIPLIER)

        Method 2: Swing point (if provided)
        - LONG: stop = recent swing low - small buffer
        - SHORT: stop = recent swing high + small buffer

        Use the TIGHTER of the two methods.
        Maximum stop distance: 5% from entry.
        """
        pass

    def calculate_take_profit(
        self,
        entry_price: float,
        stop_loss_price: float,
        direction: str
    ) -> float:
        """
        Calculate take-profit price.

        Formula:
        risk = abs(entry - stop_loss)
        reward = risk * MIN_REWARD_RATIO

        LONG: take_profit = entry + reward
        SHORT: take_profit = entry - reward
        """
        pass

    def validate_trade(self, trade_params: dict) -> tuple[bool, str]:
        """
        Validate trade against all risk rules.

        Checks:
        □ Risk/reward ratio >= MIN_REWARD_RATIO (1.5)
        □ Position size <= MAX_POSITION_SIZE
        □ Total positions < MAX_POSITIONS
        □ Portfolio exposure < MAX_PORTFOLIO_EXPOSURE
        □ Current drawdown < MAX_DRAWDOWN
        □ Stop distance <= 5%

        Returns: (is_valid, rejection_reason)
        """
        pass
```

---

## Order Execution (core/trade_executor.py)

```python
class TradeExecutor:
    """
    Handles order submission and management via Alpaca API.
    """

    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        """Initialize Alpaca API connection."""
        pass

    def submit_bracket_order(
        self,
        symbol: str,
        side: str,  # "buy" or "sell"
        qty: int,
        take_profit_price: float,
        stop_loss_price: float,
        limit_price: float = None  # Optional limit for entry
    ) -> dict:
        """
        Submit a bracket order (entry + stop-loss + take-profit).

        This creates an atomic order where:
        - Entry order executes first
        - Stop-loss and take-profit are OCO (one-cancels-other)

        Use market order for entry unless limit_price specified.
        """
        pass

    def update_stop_loss(
        self,
        position_id: str,
        new_stop_price: float
    ) -> bool:
        """
        Update stop-loss for trailing stop functionality.
        Call this when position is profitable to lock in gains.
        """
        pass

    def close_position(self, symbol: str, reason: str) -> dict:
        """
        Close entire position for a symbol.

        Reasons might be:
        - "opposite_signal": Exit crossover occurred
        - "stop_loss_hit": Price hit stop
        - "take_profit_hit": Target reached
        - "zero_line_cross": MACD crossed zero against position
        - "manual": User requested close
        """
        pass
```

---

## Exit Rules Implementation

```python
def check_exit_signals(position: dict, current_data: dict) -> dict:
    """
    Check if any exit condition is met for an open position.

    EXIT CONDITIONS (check in order):

    1. STOP-LOSS HIT
       - Price <= stop_loss (for longs)
       - Price >= stop_loss (for shorts)
       → Immediate exit, log as loss

    2. TAKE-PROFIT HIT
       - Price >= take_profit (for longs)
       - Price <= take_profit (for shorts)
       → Immediate exit, log as win

    3. OPPOSITE CROSSOVER
       - For LONG: MACD crosses BELOW signal on entry timeframe
       - For SHORT: MACD crosses ABOVE signal on entry timeframe
       → Exit at market

    4. ZERO LINE VIOLATION
       - For LONG: MACD crosses below zero
       - For SHORT: MACD crosses above zero
       → Exit at market (trend reversal)

    5. TRAILING STOP (after 1R profit achieved)
       - Once position is profitable by 1R (risk amount):
         * Move stop to breakeven
       - For each additional 0.5R of profit:
         * Trail stop up by 0.25R

    Returns:
        {
            "should_exit": bool,
            "exit_reason": str,
            "exit_type": "stop_loss" | "take_profit" | "signal" | "trailing",
        }
    """
    pass
```

---

## Scheduling & Main Loop (main.py)

```python
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

def main():
    """
    Main entry point for the trading bot.

    SCHEDULE:
    - 8:30 AM ET: Pre-market scan (load watchlist, calc daily levels)
    - Every 5 min during market hours: Signal scan
    - Every 1 min during market hours: Position monitor
    - 4:00 PM ET: End-of-day processing
    - 5:00 PM ET: Daily report generation
    """

    scheduler = BlockingScheduler(timezone="America/New_York")

    # Pre-market scan
    scheduler.add_job(
        premarket_scan,
        CronTrigger(hour=8, minute=30, day_of_week='mon-fri')
    )

    # Signal scan every 5 minutes during market hours (9:30 AM - 4:00 PM)
    scheduler.add_job(
        scan_for_signals,
        'cron',
        hour='9-15',
        minute='*/5',
        day_of_week='mon-fri'
    )

    # Position monitor every minute
    scheduler.add_job(
        monitor_positions,
        'cron',
        hour='9-15',
        minute='*',
        day_of_week='mon-fri'
    )

    # End of day processing
    scheduler.add_job(
        end_of_day_processing,
        CronTrigger(hour=16, minute=0, day_of_week='mon-fri')
    )

    scheduler.start()


def scan_for_signals():
    """
    Main signal scanning function - runs every 5 minutes.

    FLOW:
    1. Check if market is open (skip if closed)
    2. Check current drawdown (halt if exceeded)
    3. For each symbol in watchlist:
       a. Fetch latest data for all timeframes
       b. Calculate MACD indicators
       c. Run through 3 signal systems
       d. If signal found:
          - Calculate position size
          - Calculate stop-loss and take-profit
          - Validate against risk rules
          - Submit bracket order
          - Log and notify
    """
    pass


def monitor_positions():
    """
    Position monitoring function - runs every minute.

    FLOW:
    1. Get all open positions from Alpaca
    2. For each position:
       a. Fetch current price
       b. Check all exit conditions
       c. Update trailing stop if applicable
       d. Close position if exit signal triggered
       e. Log status
    """
    pass
```

---

## Database Schema (models/database.py)

```python
"""
SQLite database for trade history and state persistence.

Tables:
1. trades - Record of all trades
2. signals - Log of all generated signals
3. daily_stats - Daily performance metrics
"""

# Trade record
CREATE_TRADES_TABLE = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,  -- 'long' or 'short'
    entry_time TIMESTAMP NOT NULL,
    entry_price REAL NOT NULL,
    quantity INTEGER NOT NULL,
    stop_loss REAL NOT NULL,
    take_profit REAL NOT NULL,
    exit_time TIMESTAMP,
    exit_price REAL,
    exit_reason TEXT,
    pnl REAL,
    pnl_percent REAL,
    system_used INTEGER,  -- 1, 2, or 3
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

# Signal log
CREATE_SIGNALS_TABLE = """
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP NOT NULL,
    symbol TEXT NOT NULL,
    signal_type TEXT NOT NULL,  -- 'LONG', 'SHORT', or 'NONE'
    system INTEGER NOT NULL,
    confidence TEXT,
    macd_value REAL,
    signal_value REAL,
    histogram_value REAL,
    was_executed BOOLEAN DEFAULT FALSE,
    rejection_reason TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""
```

---

## Notification Messages

```python
# Trade executed notification
TRADE_EXECUTED_MSG = """
🚀 *TRADE EXECUTED*
━━━━━━━━━━━━━━━━━━
Symbol: {symbol}
Direction: {side}
Quantity: {qty} shares
Entry: ${entry_price:.2f}
Stop Loss: ${stop_loss:.2f}
Take Profit: ${take_profit:.2f}
Risk: ${risk_amount:.2f} ({risk_percent:.1f}%)
R:R Ratio: {rr_ratio:.1f}:1
System: {system}
"""

# Position closed notification
POSITION_CLOSED_MSG = """
{emoji} *POSITION CLOSED*
━━━━━━━━━━━━━━━━━━
Symbol: {symbol}
Direction: {side}
P&L: ${pnl:.2f} ({pnl_percent:.1f}%)
Exit Reason: {reason}
Hold Time: {hold_time}
"""

# Daily summary notification
DAILY_SUMMARY_MSG = """
📊 *DAILY SUMMARY* - {date}
━━━━━━━━━━━━━━━━━━
Trades Today: {num_trades}
Winners: {winners} | Losers: {losers}
Win Rate: {win_rate:.1f}%
Daily P&L: ${daily_pnl:.2f}
Account Value: ${account_value:.2f}
Drawdown: {drawdown:.1f}%
"""
```

---

## Testing Requirements

Create unit tests for:

1. **test_indicators.py**
   - Test MACD calculation accuracy against known values
   - Test crossover detection (bullish and bearish)
   - Test divergence detection algorithm
   - Test histogram pattern recognition

2. **test_signals.py**
   - Test System 1 signal generation
   - Test System 2 signal generation
   - Test System 3 multi-timeframe alignment
   - Test signal priority logic

3. **test_risk_manager.py**
   - Test position sizing calculations
   - Test stop-loss calculations (ATR and swing-based)
   - Test take-profit calculations
   - Test trade validation rules

---

## Environment Variables (.env.example)

```bash
# Alpaca API Credentials
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets  # Paper trading

# Optional: Notifications
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/xxx

# Trading Mode
PAPER_TRADING=true
```

---

## Watchlist

Start with these liquid, high-volume stocks:

```python
DEFAULT_WATCHLIST = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "TSLA", "AMD", "NFLX", "SPY",
    "QQQ", "JPM", "BAC", "DIS", "PYPL"
]
```

---

## Implementation Order

Build in this order:

1. **Phase 1: Foundation**
   - [ ] Project structure and config
   - [ ] Alpaca API connection
   - [ ] Market data fetching
   - [ ] MACD indicator calculations
   - [ ] Basic logging

2. **Phase 2: Signal Generation**
   - [ ] System 1 (trend) implementation
   - [ ] System 2 (reversal) implementation
   - [ ] System 3 (confirmation) implementation
   - [ ] Signal combination logic

3. **Phase 3: Risk Management**
   - [ ] Position sizing
   - [ ] Stop-loss calculation
   - [ ] Take-profit calculation
   - [ ] Trade validation

4. **Phase 4: Execution**
   - [ ] Order submission (bracket orders)
   - [ ] Position monitoring
   - [ ] Exit signal handling
   - [ ] Trailing stop implementation

5. **Phase 5: Operations**
   - [ ] Scheduler setup
   - [ ] Database persistence
   - [ ] Notifications
   - [ ] Daily reporting

6. **Phase 6: Testing**
   - [ ] Unit tests
   - [ ] Paper trading validation (30 days minimum)
   - [ ] Performance analysis

---

## Critical Reminders

1. **ALWAYS start with paper trading** - Set PAPER_TRADING=true
2. **Never trade against the zero line** - This is the most important rule
3. **Ignore weak signals** - Skip crossovers within the -0.5 to 0.5 channel
4. **Respect position limits** - Never exceed MAX_POSITIONS or MAX_PORTFOLIO_EXPOSURE
5. **Log everything** - Every signal, every trade, every decision
6. **Handle errors gracefully** - API failures should not crash the bot
7. **Test thoroughly** - Run paper trading for 30+ days before live

---

Now implement this trading bot following all the specifications above. Start with Phase 1 and work through each phase sequentially. Ask clarifying questions if any requirements are unclear.
