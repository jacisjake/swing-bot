import os
import pandas as pd
import logging
import time
import schedule
from datetime import datetime, timedelta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# API credentials
API_KEY = os.getenv("ALPACA_LIVE_API_KEY")
SECRET_KEY = os.getenv("ALPACA_LIVE_API_SECRET")
SYMBOL = os.getenv("TRADING_SYMBOL", "AAPL")  # Default to AAPL if not set
QTY = int(os.getenv("TRADING_QUANTITY", "1"))  # Default to 1 if not set

# Initialize clients (LIVE trading)
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=False)

# Strategy
SHORT_SMA = 10
LONG_SMA = 20

def fetch_data():
    try:
        start = datetime.now() - timedelta(days=60)
        bars = data_client.get_stock_bars(
            StockBarsRequest(symbol_or_symbols=SYMBOL, timeframe=TimeFrame.Day, start=start)
        ).df
        logger.info(f"Fetched {len(bars)} bars for {SYMBOL}")
        return bars[bars["symbol"] == SYMBOL]
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return None

def analyze(df):
    try:
        if df is None or len(df) < LONG_SMA:
            logger.warning("Insufficient data for analysis")
            return 0
            
        df["SMA_Short"] = df["close"].rolling(SHORT_SMA).mean()
        df["SMA_Long"] = df["close"].rolling(LONG_SMA).mean()
        
        short_sma = df["SMA_Short"].iloc[-1]
        long_sma = df["SMA_Long"].iloc[-1]
        current_price = df["close"].iloc[-1]
        
        logger.info(f"Current price: ${current_price:.2f}, Short SMA: ${short_sma:.2f}, Long SMA: ${long_sma:.2f}")
        
        if short_sma > long_sma:
            return 1
        elif short_sma < long_sma:
            return -1
        return 0
    except Exception as e:
        logger.error(f"Error in analysis: {e}")
        return 0

def place_order(signal):
    try:
        if signal == 0:
            logger.info("✅ No signal.")
            return
            
        # Check current position first
        try:
            position = trading_client.get_position(SYMBOL)
            current_qty = float(position.qty) if position else 0
            logger.info(f"Current position in {SYMBOL}: {current_qty} shares")
        except Exception:
            current_qty = 0
            logger.info(f"No current position in {SYMBOL}")
            
        # Determine if we should place order
        if signal == 1 and current_qty <= 0:  # Buy signal and no long position
            side = OrderSide.BUY
        elif signal == -1 and current_qty >= 0:  # Sell signal and no short position
            side = OrderSide.SELL
        else:
            logger.info(f"Signal {signal} ignored due to existing position")
            return
            
        order = MarketOrderRequest(symbol=SYMBOL, qty=QTY, side=side, time_in_force=TimeInForce.GTC)
        result = trading_client.submit_order(order)
        logger.info(f"📈 Placed {side.value.upper()} order for {SYMBOL} - Order ID: {result.id}")
        
    except Exception as e:
        logger.error(f"Error placing order: {e}")

def run():
    logger.info(f"🚦 Running bot for {SYMBOL}")
    try:
        df = fetch_data()
        signal = analyze(df)
        place_order(signal)
        logger.info("Trading cycle completed successfully")
    except Exception as e:
        logger.error(f"Error in trading cycle: {e}")

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs('/app/logs', exist_ok=True)
    
    logger.info("🚀 Starting Alpaca Swing Trading Bot")
    logger.info(f"Trading symbol: {SYMBOL}")
    logger.info("⚠️  LIVE TRADING MODE - Real money at risk!")
    
    # Schedule the bot to run every 10 minutes
    schedule.every(10).minutes.do(run)
    
    # Run immediately on startup
    run()
    
    # Keep the script running
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute
