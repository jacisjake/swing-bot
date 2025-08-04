import os
import pandas as pd
import logging
import time
import schedule
from datetime import datetime, timedelta
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
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

# Trading symbols
STOCK_SYMBOL = os.getenv("STOCK_SYMBOL", "NVDA")  # Stock for market hours
CRYPTO_SYMBOL = os.getenv("CRYPTO_SYMBOL", "LTC/USD")  # Crypto for after hours
STOCK_QTY = int(os.getenv("STOCK_QUANTITY", "1"))  # Stock quantity
CRYPTO_QTY = float(os.getenv("CRYPTO_QUANTITY", "0.1"))  # Crypto quantity (smaller amounts)

# Initialize clients (LIVE trading)
stock_data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
crypto_data_client = CryptoHistoricalDataClient(API_KEY, SECRET_KEY)
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=False)

# Strategy
SHORT_SMA = 10
LONG_SMA = 20

def is_market_hours():
    """Check if it's during US stock market hours (9:30 AM - 4:00 PM ET, Mon-Fri)"""
    from datetime import datetime
    import pytz
    
    et = pytz.timezone('US/Eastern')
    now_et = datetime.now(et)
    
    # Check if it's a weekday (0=Monday, 4=Friday)
    if now_et.weekday() > 4:  # Saturday (5) or Sunday (6)
        return False
    
    # Check if it's between 9:30 AM and 4:00 PM ET
    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    
    return market_open <= now_et <= market_close

def get_current_symbol_and_qty():
    """Get the appropriate symbol and quantity based on market hours"""
    if is_market_hours():
        return STOCK_SYMBOL, STOCK_QTY, "stock"
    else:
        return CRYPTO_SYMBOL, CRYPTO_QTY, "crypto"

def fetch_data():
    symbol, qty, asset_type = get_current_symbol_and_qty()
    
    try:
        start = datetime.now() - timedelta(days=60)
        
        if asset_type == "stock":
            request = StockBarsRequest(
                symbol_or_symbols=[symbol], 
                timeframe=TimeFrame.Day, 
                start=start
            )
            bars_response = stock_data_client.get_stock_bars(request)
        else:  # crypto
            request = CryptoBarsRequest(
                symbol_or_symbols=[symbol], 
                timeframe=TimeFrame.Hour,  # Use hourly data for crypto (more active)
                start=start
            )
            bars_response = crypto_data_client.get_crypto_bars(request)
        
        # Convert to DataFrame and handle the multi-index
        df = bars_response.df
        
        # If there's a symbol level in the index, get data for our symbol
        if symbol in df.index.get_level_values(0):
            df = df.loc[symbol].copy()
        elif hasattr(df.index, 'get_level_values') and len(df.index.get_level_values(0)) > 0:
            # If multi-index but different structure, try to extract our symbol
            df = df.xs(symbol, level=0) if symbol in df.index.get_level_values(0) else df
        
        logger.info(f"Fetched {len(df)} bars for {symbol} ({asset_type})")
        logger.info(f"Data columns: {list(df.columns)}")
        logger.info(f"Latest close price: ${df['close'].iloc[-1]:.2f}")
        
        return df, symbol, qty, asset_type
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None, symbol, qty, asset_type

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

def place_order(signal, symbol, qty, asset_type):
    try:
        if signal == 0:
            logger.info("✅ No signal.")
            return
            
        # Check current position first
        try:
            position = trading_client.get_position(symbol)
            current_qty = float(position.qty) if position else 0
            logger.info(f"Current position in {symbol}: {current_qty} shares/units")
        except Exception:
            current_qty = 0
            logger.info(f"No current position in {symbol}")
            
        # Determine if we should place order
        if signal == 1 and current_qty <= 0:  # Buy signal and no long position
            side = OrderSide.BUY
        elif signal == -1 and current_qty >= 0:  # Sell signal and no short position
            side = OrderSide.SELL
        else:
            logger.info(f"Signal {signal} ignored due to existing position")
            return
            
        order = MarketOrderRequest(symbol=symbol, qty=qty, side=side, time_in_force=TimeInForce.GTC)
        result = trading_client.submit_order(order)
        logger.info(f"📈 Placed {side.value.upper()} order for {symbol} - Order ID: {result.id}")
        
    except Exception as e:
        logger.error(f"Error placing order: {e}")

def run():
    symbol, qty, asset_type = get_current_symbol_and_qty()
    market_status = "MARKET HOURS" if asset_type == "stock" else "AFTER HOURS"
    
    logger.info(f"🚦 Running bot for {symbol} ({asset_type.upper()}) - {market_status}")
    try:
        df, symbol, qty, asset_type = fetch_data()
        signal = analyze(df)
        place_order(signal, symbol, qty, asset_type)
        logger.info("Trading cycle completed successfully")
    except Exception as e:
        logger.error(f"Error in trading cycle: {e}")

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs('/app/logs', exist_ok=True)
    
    logger.info("🚀 Starting Alpaca Multi-Asset Swing Trading Bot")
    logger.info(f"Stock symbol (market hours): {STOCK_SYMBOL}")
    logger.info(f"Crypto symbol (after hours): {CRYPTO_SYMBOL}")
    logger.info("⚠️  LIVE TRADING MODE - Real money at risk!")
    
    # Schedule the bot to run every 10 minutes
    schedule.every(10).minutes.do(run)
    
    # Run immediately on startup
    run()
    
    # Keep the script running
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute
