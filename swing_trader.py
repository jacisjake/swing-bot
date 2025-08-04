import os
import pandas as pd
import logging
import time
import schedule
import pytz
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
CRYPTO_QTY = float(os.getenv("CRYPTO_QUANTITY", "0.1"))  # Crypto quantity

# Risk Management Settings
STOP_LOSS_PERCENT = float(os.getenv("STOP_LOSS_PERCENT", "0.03"))  # 3% stop loss
TAKE_PROFIT_PERCENT = float(os.getenv("TAKE_PROFIT_PERCENT", "0.06"))  # 6% take profit

# Initialize clients (LIVE trading)
stock_data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
crypto_data_client = CryptoHistoricalDataClient(API_KEY, SECRET_KEY)
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=False)

# Strategy parameters
SHORT_SMA = 10
LONG_SMA = 20

def is_market_hours():
    """Check if US stock market is currently open"""
    try:
        et_tz = pytz.timezone('US/Eastern')
        now_et = datetime.now(et_tz)
        
        # Check if weekend
        if now_et.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
            
        # Check if within trading hours (9:30 AM - 4:00 PM ET)
        market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= now_et <= market_close
    except Exception as e:
        logger.error(f"Error checking market hours: {e}")
        return False

def get_current_symbol_and_qty():
    """Return current trading symbol and quantity based on market hours"""
    if is_market_hours():
        return STOCK_SYMBOL, STOCK_QTY, "STOCK", "MARKET HOURS"
    else:
        return CRYPTO_SYMBOL, CRYPTO_QTY, "CRYPTO", "AFTER HOURS"

def fetch_data(symbol, is_crypto=False):
    """Fetch historical data for analysis"""
    try:
        start = datetime.now() - timedelta(days=60)
        
        if is_crypto:
            # Use hourly data for crypto (24/7 trading)
            request = CryptoBarsRequest(
                symbol_or_symbols=[symbol], 
                timeframe=TimeFrame.Hour, 
                start=start
            )
            bars_response = crypto_data_client.get_crypto_bars(request)
        else:
            # Use daily data for stocks
            request = StockBarsRequest(
                symbol_or_symbols=[symbol], 
                timeframe=TimeFrame.Day, 
                start=start
            )
            bars_response = stock_data_client.get_stock_bars(request)
        
        # Convert to DataFrame and handle the multi-index
        df = bars_response.df
        
        # If there's a symbol level in the index, get data for our symbol
        if symbol in df.index.get_level_values(0):
            df = df.loc[symbol].copy()
        elif hasattr(df.index, 'get_level_values') and len(df.index.get_level_values(0)) > 0:
            # If multi-index but different structure, try to extract our symbol
            df = df.xs(symbol, level=0) if symbol in df.index.get_level_values(0) else df
        
        logger.info(f"Fetched {len(df)} bars for {symbol} ({'crypto' if is_crypto else 'stock'})")
        logger.info(f"Data columns: {list(df.columns)}")
        
        if len(df) > 0:
            logger.info(f"Latest close price: ${df['close'].iloc[-1]:.2f}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def check_risk_management_exit(symbol, current_qty):
    """Check if position should be closed based on take-profit or stop-loss"""
    try:
        if current_qty == 0:
            return False, "No position to check"
            
        # Get current position details
        position = trading_client.get_position(symbol)
        if not position:
            return False, "No position found"
            
        entry_price = float(position.avg_fill_price)
        current_price = float(position.market_value) / float(position.qty)
        
        # Calculate profit/loss percentage
        if current_qty > 0:  # Long position
            pnl_percent = (current_price - entry_price) / entry_price
        else:  # Short position
            pnl_percent = (entry_price - current_price) / entry_price
            
        logger.info(f"Position check - Entry: ${entry_price:.2f}, Current: ${current_price:.2f}, P&L: {pnl_percent:.2%}")
        
        # Check take-profit
        if pnl_percent >= TAKE_PROFIT_PERCENT:
            return True, f"TAKE PROFIT triggered - Gain: {pnl_percent:.2%} (target: {TAKE_PROFIT_PERCENT:.2%})"
            
        # Check stop-loss
        if pnl_percent <= -STOP_LOSS_PERCENT:
            return True, f"STOP LOSS triggered - Loss: {pnl_percent:.2%} (limit: {-STOP_LOSS_PERCENT:.2%})"
            
        return False, f"Position within range - P&L: {pnl_percent:.2%}"
        
    except Exception as e:
        logger.error(f"Error checking risk management: {e}")
        return False, "Error checking position"

def analyze(df):
    """Analyze data and generate trading signals"""
    try:
        if df is None or len(df) < LONG_SMA:
            logger.warning("Insufficient data for analysis")
            return 0
            
        df["SMA_Short"] = df["close"].rolling(SHORT_SMA).mean()
        df["SMA_Long"] = df["close"].rolling(LONG_SMA).mean()
        
        # Get latest values
        latest_close = df["close"].iloc[-1]
        latest_short_sma = df["SMA_Short"].iloc[-1]
        latest_long_sma = df["SMA_Long"].iloc[-1]
        
        logger.info(f"Current price: ${latest_close:.2f}, Short SMA: ${latest_short_sma:.2f}, Long SMA: ${latest_long_sma:.2f}")
        
        # Generate signals
        if latest_short_sma > latest_long_sma:
            return 1  # Bullish signal
        elif latest_short_sma < latest_long_sma:
            return -1  # Bearish signal
        
        return 0
        
    except Exception as e:
        logger.error(f"Error in analysis: {e}")
        return 0

def place_order(signal, symbol, qty):
    """Place orders based on signals and risk management"""
    try:
        # Check current position first
        try:
            position = trading_client.get_position(symbol)
            current_qty = float(position.qty) if position else 0
            logger.info(f"Current position in {symbol}: {current_qty}")
        except Exception:
            current_qty = 0
            logger.info(f"No current position in {symbol}")
        
        # Check risk management exits first
        should_exit, exit_reason = check_risk_management_exit(symbol, current_qty)
        if should_exit and current_qty != 0:
            # Force exit regardless of SMA signal
            side = OrderSide.SELL if current_qty > 0 else OrderSide.BUY
            order = MarketOrderRequest(symbol=symbol, qty=abs(current_qty), side=side, time_in_force=TimeInForce.GTC)
            result = trading_client.submit_order(order)
            logger.info(f"🚨 RISK MANAGEMENT EXIT: {exit_reason}")
            logger.info(f"📈 Placed {side.value.upper()} order for {symbol} - Order ID: {result.id}")
            return
            
        # Regular SMA-based trading logic
        if signal == 0:
            logger.info("✅ No signal.")
            return
            
        # Determine if we should place order based on SMA signals
        if signal == 1 and current_qty <= 0:  # Buy signal and no long position
            side = OrderSide.BUY
        elif signal == -1 and current_qty >= 0:  # Sell signal and no short position
            side = OrderSide.SELL
        else:
            logger.info(f"SMA signal {signal} ignored due to existing position - {exit_reason}")
            return
            
        order = MarketOrderRequest(symbol=symbol, qty=qty, side=side, time_in_force=TimeInForce.GTC)
        result = trading_client.submit_order(order)
        logger.info(f"📈 Placed {side.value.upper()} order for {symbol} - Order ID: {result.id}")
        
    except Exception as e:
        logger.error(f"Error placing order: {e}")

def run():
    """Main trading loop execution"""
    try:
        symbol, qty, asset_type, market_status = get_current_symbol_and_qty()
        is_crypto = asset_type == "CRYPTO"
        
        logger.info(f"🚦 Running bot for {symbol} ({asset_type}) - {market_status}")
        
        df = fetch_data(symbol, is_crypto)
        if df is None:
            logger.error("Failed to fetch data")
            return
            
        signal = analyze(df)
        place_order(signal, symbol, qty)
        
        logger.info("Trading cycle completed successfully")
        
    except Exception as e:
        logger.error(f"Error in trading cycle: {e}")

def main():
    """Main function"""
    # Create logs directory if it doesn't exist
    os.makedirs('/app/logs', exist_ok=True)
    
    logger.info("🚀 Starting Alpaca Multi-Asset Swing Trading Bot with Risk Management")
    logger.info(f"Stock symbol (market hours): {STOCK_SYMBOL}")
    logger.info(f"Crypto symbol (after hours): {CRYPTO_SYMBOL}")
    logger.info(f"📊 Risk Management - Stop Loss: {STOP_LOSS_PERCENT:.1%}, Take Profit: {TAKE_PROFIT_PERCENT:.1%}")
    logger.info("⚠️  LIVE TRADING MODE - Real money at risk!")
    
    # Run immediately
    run()
    
    # Schedule to run every 10 minutes
    schedule.every(10).minutes.do(run)
    
    # Keep the script running
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    main()
