import os
import pandas as pd
import logging
import time
import schedule
import pytz
from datetime import datetime, timedelta
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.historical.screener import ScreenerClient
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest, MarketMoversRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
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

# Debug: Check if API keys are loaded
if not API_KEY or not SECRET_KEY:
    logger.error("❌ API keys not found in environment variables!")
    logger.error(f"API_KEY loaded: {'✅' if API_KEY else '❌'}")
    logger.error(f"SECRET_KEY loaded: {'✅' if SECRET_KEY else '❌'}")
    logger.error("💡 Check Portainer stack environment variables or .env file mounting")
else:
    logger.info(f"✅ API keys loaded: API_KEY={API_KEY[:8]}..., SECRET_KEY={'*' * len(SECRET_KEY) if SECRET_KEY else 'None'}")

# Trading symbols - Dynamic via screener
STOCK_SYMBOL = os.getenv("STOCK_SYMBOL", "NVDA")  # Fallback symbol
CRYPTO_SYMBOL = os.getenv("CRYPTO_SYMBOL", "LTC/USD")  # Fallback symbol
STOCK_QTY = int(os.getenv("STOCK_QUANTITY", "1"))

# Screener Configuration
MAX_STOCK_SYMBOLS = int(os.getenv("MAX_STOCK_SYMBOLS", "10"))  # Max stocks to trade
MAX_CRYPTO_SYMBOLS = int(os.getenv("MAX_CRYPTO_SYMBOLS", "5"))  # Max crypto to trade
MIN_DAILY_VOLUME = int(os.getenv("MIN_DAILY_VOLUME", "1000000"))  # Min $1M daily volume
MIN_PRICE = float(os.getenv("MIN_PRICE", "5.0"))  # Min $5 per share
MAX_PRICE = float(os.getenv("MAX_PRICE", "500.0"))  # Max $500 per share
SCREENER_UPDATE_HOURS = int(os.getenv("SCREENER_UPDATE_HOURS", "1"))  # Update symbols every hour

# Dynamic symbol lists
active_stock_symbols = [STOCK_SYMBOL]  # Start with fallback
active_crypto_symbols = [CRYPTO_SYMBOL]  # Start with fallback
last_screener_update = datetime.min

# Portfolio Allocation Settings
MAX_POSITION_PERCENT = float(os.getenv("MAX_POSITION_PERCENT", "0.20"))  # 20% max per position
MAX_CASH_PER_TRADE = float(os.getenv("MAX_CASH_PER_TRADE", "0.10"))  # 10% of available cash per trade
STOP_LOSS_PERCENT = float(os.getenv("STOP_LOSS_PERCENT", "0.005"))  # 0.5% stop loss
TAKE_PROFIT_1 = float(os.getenv("TAKE_PROFIT_1", "0.005"))  # 0.5% first target
TAKE_PROFIT_2 = float(os.getenv("TAKE_PROFIT_2", "0.010"))  # 1.0% second target  
TAKE_PROFIT_3 = float(os.getenv("TAKE_PROFIT_3", "0.015"))  # 1.5% third target

# Initialize clients (LIVE trading)
stock_data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
crypto_data_client = CryptoHistoricalDataClient(API_KEY, SECRET_KEY)
screener_client = ScreenerClient(API_KEY, SECRET_KEY)
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=False)

# EMA Strategy parameters
FAST_EMA = 9
SLOW_EMA = 21

# Global position tracking - removed old LTC-specific tracking
# Now using generic position management for all symbols

class PositionTracker:
    """Generic position tracker for any symbol with trailing stop support"""
    def __init__(self, symbol, entry_price, quantity, order_id):
        self.symbol = symbol
        self.entry_price = entry_price
        self.quantity = quantity
        self.order_id = order_id
        self.highest_price = entry_price  # Track highest price for trailing stop
        self.trailing_stop_price = None   # Current trailing stop level

def is_market_hours():
    """Check if US stock market is currently open"""
    try:
        et_tz = pytz.timezone('US/Eastern')
        now_et = datetime.now(et_tz)
        
        if now_et.weekday() >= 5:  # Weekend
            return False
            
        market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= now_et <= market_close
    except Exception as e:
        logger.error(f"Error checking market hours: {e}")
        return False

def get_portfolio_value():
    """Get current portfolio value"""
    try:
        account = trading_client.get_account()
        return float(account.portfolio_value)
    except Exception as e:
        logger.error(f"Error getting portfolio value: {e}")
        return 100000  # Default fallback

def calculate_ema(data, period):
    """Calculate Exponential Moving Average"""
    return data.ewm(span=period, adjust=False).mean()

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD, Signal line, and Histogram"""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def validate_api_access():
    """Validate API access for both stock and crypto data"""
    try:
        # Debug: Show environment variables
        logger.info("🔍 Environment Variables Debug:")
        alpaca_vars = {k: v for k, v in os.environ.items() if 'ALPACA' in k.upper()}
        if alpaca_vars:
            for key, value in alpaca_vars.items():
                masked_value = value[:8] + "..." if len(value) > 8 else "***" if value else "None"
                logger.info(f"  {key}: {masked_value}")
        else:
            logger.error("❌ No ALPACA environment variables found!")
            
        if not API_KEY or not SECRET_KEY:
            logger.error("❌ Cannot proceed without valid API credentials")
            return False
        
        # Test account access
        account = trading_client.get_account()
        logger.info(f"✅ Account access validated. Portfolio: ${float(account.portfolio_value):.2f}")
        
        # Test stock data access
        try:
            test_stock_request = StockBarsRequest(
                symbol_or_symbols=["AAPL"], 
                timeframe=TimeFrame(1, TimeFrameUnit.Day),
                start=datetime.now() - timedelta(days=2)
            )
            stock_data_client.get_stock_bars(test_stock_request)
            logger.info("✅ Stock data access validated")
        except Exception as e:
            logger.error(f"❌ Stock data access failed: {e}")
            
        # Test crypto data access
        try:
            test_crypto_request = CryptoBarsRequest(
                symbol_or_symbols=["BTC/USD"], 
                timeframe=TimeFrame(1, TimeFrameUnit.Hour),
                start=datetime.now() - timedelta(days=1)
            )
            crypto_data_client.get_crypto_bars(test_crypto_request)
            logger.info("✅ Crypto data access validated")
        except Exception as e:
            logger.error(f"❌ Crypto data access failed: {e}")
            logger.error("💡 Your API keys may not have crypto data access. Check Alpaca account permissions.")
            
    except Exception as e:
        logger.error(f"❌ API validation failed: {e}")
        return False
    return True

def check_existing_positions():
    """Check for existing positions on startup"""
    try:
        logger.info("📋 Checking existing positions...")
        
        # Get all positions
        positions = trading_client.get_all_positions()
        
        if not positions:
            logger.info("📭 No existing positions found")
            return
            
        logger.info(f"📊 Found {len(positions)} existing position(s):")
        
        total_position_value = 0
        for position in positions:
            symbol = position.symbol
            qty = float(position.qty)
            side = "LONG" if qty > 0 else "SHORT"
            market_value = float(position.market_value)
            unrealized_pl = float(position.unrealized_pl)
            unrealized_plpc = float(position.unrealized_plpc) * 100
            
            # Try different attribute names for entry price
            avg_entry_price = None
            if hasattr(position, 'avg_fill_price'):
                avg_entry_price = float(position.avg_fill_price)
            elif hasattr(position, 'avg_entry_price'):
                avg_entry_price = float(position.avg_entry_price)
            elif hasattr(position, 'cost_basis'):
                avg_entry_price = float(position.cost_basis) / abs(qty)
            else:
                avg_entry_price = abs(float(market_value)) / abs(qty)  # Fallback calculation
            
            total_position_value += abs(market_value)
            
            logger.info(f"  📈 {symbol}: {side} {abs(qty)} shares @ ${avg_entry_price:.2f}")
            logger.info(f"      💰 Market Value: ${market_value:.2f}")
            logger.info(f"      📊 P&L: ${unrealized_pl:.2f} ({unrealized_plpc:+.2f}%)")
            
            # Check if it's one of our tracked symbols
            if symbol == STOCK_SYMBOL:
                logger.info(f"      🎯 [NVDA] Tracked position - will manage with 6% TP / 3% SL")
            elif symbol == CRYPTO_SYMBOL:
                logger.info(f"      🎯 [LTC] Tracked position - NOTE: Manual position, not in LTC scalping tracker")
                
        # Portfolio allocation info
        portfolio_value = get_portfolio_value()
        allocation_percent = (total_position_value / portfolio_value) * 100
        logger.info(f"💼 Total Position Value: ${total_position_value:.2f} ({allocation_percent:.1f}% of portfolio)")
        logger.info(f"💼 Available for new positions: ${portfolio_value - total_position_value:.2f}")
        
    except Exception as e:
        logger.error(f"Error checking existing positions: {e}")
        # Debug: Show position attributes
        try:
            positions = trading_client.get_all_positions()
            if positions:
                sample_position = positions[0]
                logger.info(f"🔍 Position attributes: {[attr for attr in dir(sample_position) if not attr.startswith('_')]}")
        except:
            pass

# =============================================================================
# SCREENER FUNCTIONS - Dynamic Symbol Selection
# =============================================================================

def screen_top_stock_movers():
    """Screen for top stock movers with good volume and trending direction"""
    try:
        logger.info("🔍 Screening for top stock movers...")
        
        # Use Alpaca's market movers API for gainers
        movers_request = MarketMoversRequest(
            top=50  # Get top 50 movers
        )
        
        movers_response = screener_client.get_market_movers(movers_request)
        
        if not movers_response or not hasattr(movers_response, 'gainers'):
            logger.warning("⚠️ No movers results received")
            return [STOCK_SYMBOL]  # Return fallback
            
        filtered_symbols = []
        
        # Check gainers first
        if hasattr(movers_response, 'gainers') and movers_response.gainers:
            for stock in movers_response.gainers:
                symbol = stock.symbol
                
                # Skip if already have enough symbols
                if len(filtered_symbols) >= MAX_STOCK_SYMBOLS:
                    break
                    
                try:
                    # Check price criteria (volume checking will be done separately)
                    if (hasattr(stock, 'price') and 
                        MIN_PRICE <= float(stock.price) <= MAX_PRICE):
                        
                        filtered_symbols.append(symbol)
                        logger.info(f"  ✅ {symbol}: ${float(stock.price):.2f}, Change: {float(stock.change) if hasattr(stock, 'change') else 'N/A'}%")
                    else:
                        logger.debug(f"  ❌ {symbol}: Failed price criteria")
                        
                except Exception as e:
                    logger.debug(f"  ❌ {symbol}: Error checking criteria - {e}")
                    continue
        
        if not filtered_symbols:
            logger.warning("⚠️ No stocks met screening criteria, using fallback")
            return [STOCK_SYMBOL]
            
        logger.info(f"📊 Selected {len(filtered_symbols)} stocks: {filtered_symbols}")
        return filtered_symbols
        
    except Exception as e:
        logger.error(f"Error in stock screening: {e}")
        return [STOCK_SYMBOL]  # Return fallback on error

def screen_top_crypto_movers():
    """Screen for top crypto movers based on recent performance"""
    try:
        logger.info("🔍 Screening for top crypto movers...")
        
        # Major crypto pairs available on Alpaca
        crypto_candidates = [
            "BTC/USD", "ETH/USD", "LTC/USD", "BCH/USD", "LINK/USD",
            "AAVE/USD", "UNI/USD", "SUSHI/USD", "ALGO/USD", "DOT/USD"
        ]
        
        crypto_performance = []
        
        # Analyze recent performance for each crypto
        for symbol in crypto_candidates:
            try:
                # Get 24-hour data
                start = datetime.now() - timedelta(days=1)
                request = CryptoBarsRequest(
                    symbol_or_symbols=[symbol],
                    timeframe=TimeFrame(1, TimeFrameUnit.Hour),
                    start=start
                )
                
                response = crypto_data_client.get_crypto_bars(request)
                if not response.df.empty:
                    df = response.df
                    if symbol in df.index.get_level_values(0):
                        df = df.loc[symbol]
                    
                    if len(df) >= 2:
                        # Calculate 24h performance
                        first_price = df['close'].iloc[0]
                        last_price = df['close'].iloc[-1]
                        performance = (last_price - first_price) / first_price
                        
                        # Calculate volatility (helpful for swing trading)
                        volatility = df['close'].pct_change().std()
                        
                        crypto_performance.append({
                            'symbol': symbol,
                            'performance': performance,
                            'volatility': volatility,
                            'price': last_price,
                            'volume': df['volume'].iloc[-1] if 'volume' in df.columns else 0
                        })
                        
                        logger.debug(f"  📈 {symbol}: {performance:.2%} (Vol: {volatility:.4f})")
                        
            except Exception as e:
                logger.debug(f"  ❌ {symbol}: Error getting data - {e}")
                continue
                
        if not crypto_performance:
            logger.warning("⚠️ No crypto data available, using fallback")
            return [CRYPTO_SYMBOL]
            
        # Sort by combination of positive performance and good volatility
        crypto_performance.sort(key=lambda x: x['performance'] + (x['volatility'] * 0.5), reverse=True)
        
        # Select top performers
        selected_cryptos = [item['symbol'] for item in crypto_performance[:MAX_CRYPTO_SYMBOLS]]
        
        logger.info(f"📊 Selected {len(selected_cryptos)} cryptos: {selected_cryptos}")
        for crypto in crypto_performance[:MAX_CRYPTO_SYMBOLS]:
            logger.info(f"  🎯 {crypto['symbol']}: {crypto['performance']:.2%} performance, {crypto['volatility']:.4f} volatility")
            
        return selected_cryptos if selected_cryptos else [CRYPTO_SYMBOL]
        
    except Exception as e:
        logger.error(f"Error in crypto screening: {e}")
        return [CRYPTO_SYMBOL]  # Return fallback on error

def update_symbol_lists():
    """Update active symbol lists using screener"""
    global active_stock_symbols, active_crypto_symbols, last_screener_update
    
    try:
        current_time = datetime.now()
        
        # Check if we need to update (every X hours)
        if (current_time - last_screener_update).total_seconds() < SCREENER_UPDATE_HOURS * 3600:
            logger.debug(f"⏳ Screener update not due yet (last: {last_screener_update})")
            return False
            
        logger.info("🔄 Updating symbol lists via screener...")
        
        # Update stock symbols
        new_stock_symbols = screen_top_stock_movers()
        if new_stock_symbols != active_stock_symbols:
            logger.info(f"📈 Stock symbols updated: {active_stock_symbols} → {new_stock_symbols}")
            active_stock_symbols = new_stock_symbols
            
        # Update crypto symbols  
        new_crypto_symbols = screen_top_crypto_movers()
        if new_crypto_symbols != active_crypto_symbols:
            logger.info(f"💰 Crypto symbols updated: {active_crypto_symbols} → {new_crypto_symbols}")
            active_crypto_symbols = new_crypto_symbols
            
        last_screener_update = current_time
        
        logger.info(f"✅ Screener update complete. Tracking {len(active_stock_symbols)} stocks, {len(active_crypto_symbols)} cryptos")
        return True
        
    except Exception as e:
        logger.error(f"Error updating symbol lists: {e}")
        return False

# =============================================================================
# MULTI-SYMBOL POSITION MANAGEMENT
# =============================================================================

def calculate_position_size_per_symbol(current_price, symbol_type="stock"):
    """Calculate position size with 20% portfolio max and 10% cash max constraints - NO MARGIN"""
    try:
        # Get account info
        account = trading_client.get_account()
        portfolio_value = float(account.portfolio_value)
        available_cash = float(account.cash)
        
        # CRITICAL: Never trade if we don't have cash
        if available_cash <= 0:
            logger.warning(f"⚠️ No available cash (${available_cash:.2f}) - skipping trade to avoid margin")
            return 0
        
        # Calculate maximum investment based on constraints
        max_by_portfolio = portfolio_value * MAX_POSITION_PERCENT  # 20% of total portfolio
        max_by_cash = available_cash * MAX_CASH_PER_TRADE  # 10% of available cash
        
        # Use the smaller of the two limits
        max_investment = min(max_by_portfolio, max_by_cash)
        
        # SAFETY CHECK: Never exceed available cash (no borrowing)
        max_investment = min(max_investment, available_cash)
        
        # Ensure we have enough for minimum trade (Alpaca minimum is $1 for fractional shares)
        min_trade_value = 1.0 if symbol_type == "stock" else current_price * 0.01  # $1 min for stocks, crypto varies
        if max_investment < min_trade_value:
            logger.warning(f"⚠️ Insufficient funds: ${max_investment:.2f} < ${min_trade_value:.2f} minimum")
            return 0
        
        # Calculate position size
        if symbol_type == "stock":
            # For stocks, use fractional shares (Alpaca supports fractional trading)
            max_shares = max_investment / current_price
            position_size = round(max_shares, 6) if max_shares >= 0.000001 else 0  # 6 decimal precision
        else:
            # For crypto, allow fractional shares
            max_shares = max_investment / current_price
            position_size = round(max_shares, 4) if max_shares >= 0.01 else 0
        
        # Final check: Ensure total cost doesn't exceed cash
        total_cost = position_size * current_price
        if total_cost > available_cash:
            logger.warning(f"⚠️ Position cost ${total_cost:.2f} exceeds cash ${available_cash:.2f} - reducing")
            if symbol_type == "stock":
                position_size = round(available_cash / current_price, 6)  # Fractional shares for stocks
            else:
                position_size = round(available_cash / current_price, 4)
        
        logger.info(f"💼 Position Sizing for {symbol_type.upper()} @ ${current_price:.2f}:")
        logger.info(f"   Portfolio: ${portfolio_value:.2f}, Cash: ${available_cash:.2f}")
        logger.info(f"   Max by portfolio (20%): ${max_by_portfolio:.2f}")
        logger.info(f"   Max by cash (10%): ${max_by_cash:.2f}")
        logger.info(f"   Using limit: ${max_investment:.2f} → {position_size} units")
        logger.info(f"   Total cost: ${position_size * current_price:.2f} (within cash: ✅)")
        
        return position_size
        
    except Exception as e:
        logger.error(f"Error calculating {symbol_type} position size: {e}")
        return 0  # Return 0 on error to prevent any trade

# =============================================================================
# MULTI-SYMBOL TRADING SYSTEM
# =============================================================================

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function with screener-based multi-symbol execution"""
    os.makedirs('/app/logs', exist_ok=True)
    
    logger.info("🚀 Starting Screener-Based Multi-Asset Trading Bot")
    logger.info(f"📊 Max Symbols: {MAX_STOCK_SYMBOLS} stocks, {MAX_CRYPTO_SYMBOLS} cryptos")
    logger.info(f"📈 Strategy: EMA {FAST_EMA}/{SLOW_EMA}")
    logger.info(f"💰 Position Limits: {MAX_POSITION_PERCENT:.0%} of portfolio per position, {MAX_CASH_PER_TRADE:.0%} of cash per trade")
    logger.info(f"� Screener Criteria: ${MIN_PRICE}-${MAX_PRICE}, ${MIN_DAILY_VOLUME:,} min volume")
    logger.info(f"� Symbol update frequency: Every {SCREENER_UPDATE_HOURS} hours")
    logger.info("⚠️  LIVE TRADING MODE - Real money at risk!")
    
    # Validate API access before starting
    logger.info("🔍 Validating API access...")
    if not validate_api_access():
        logger.error("❌ API validation failed. Exiting.")
        return
    
    # Initial screener update
    logger.info("🔄 Running initial screener update...")
    update_symbol_lists()
    
    # Check existing positions
    check_existing_positions()
    
    # Run initial cycles
    run_multi_symbol_trading()
    
    # Schedule trading cycles
    schedule.every(5).minutes.do(run_multi_symbol_trading)  # Every 5 minutes
    schedule.every(1).hours.do(update_symbol_lists)  # Check for symbol updates hourly
    
    # Keep running
    while True:
        schedule.run_pending()
        time.sleep(30)

def run_multi_symbol_trading():
    """Run trading strategy across all screened symbols"""
    try:
        logger.info("🔄 Running multi-symbol trading cycle")
        
        # Update symbols if needed
        update_symbol_lists()
        
        # Trade crypto symbols (24/7)
        for crypto_symbol in active_crypto_symbols:
            try:
                trade_crypto_symbol(crypto_symbol)
            except Exception as e:
                logger.error(f"Error trading {crypto_symbol}: {e}")
                
        # Trade stock symbols (market hours only)
        if is_market_hours():
            for stock_symbol in active_stock_symbols:
                try:
                    trade_stock_symbol(stock_symbol)
                except Exception as e:
                    logger.error(f"Error trading {stock_symbol}: {e}")
        else:
            logger.info("📅 Market closed - skipping stock trading")
            
    except Exception as e:
        logger.error(f"Error in multi-symbol trading: {e}")

def trade_crypto_symbol(symbol):
    """Trade a specific crypto symbol"""
    try:
        # Fetch data
        df = fetch_symbol_data(symbol, "crypto")
        if df is None:
            return
            
        current_price = df['close'].iloc[-1]
        
        # Check for entry signal
        should_enter, reason = check_entry_signal(df, symbol)
        
        if should_enter:
            position_size = calculate_position_size_per_symbol(current_price, "crypto")
            
            if position_size > 0:
                place_order(symbol, OrderSide.BUY, position_size, f"CRYPTO ENTRY - {reason}")
            else:
                logger.info(f"⚠️ [{symbol}] Position size too small")
        else:
            logger.debug(f"⏳ [{symbol}] No entry signal - {reason}")
            
        # Manage existing position with technical indicators
        manage_position_with_indicators(symbol, current_price, "crypto", df)
        
    except Exception as e:
        logger.error(f"Error trading crypto {symbol}: {e}")

def trade_stock_symbol(symbol):
    """Trade a specific stock symbol"""
    try:
        # Fetch data
        df = fetch_symbol_data(symbol, "stock")
        if df is None:
            return
            
        current_price = df['close'].iloc[-1]
        
        # Check for entry signal
        should_enter, reason = check_entry_signal(df, symbol)
        
        if should_enter:
            position_size = calculate_position_size_per_symbol(current_price, "stock")
            
            if position_size >= 1:
                place_order(symbol, OrderSide.BUY, position_size, f"STOCK ENTRY - {reason}")
            else:
                logger.info(f"⚠️ [{symbol}] Position size too small")
        else:
            logger.debug(f"⏳ [{symbol}] No entry signal - {reason}")
            
        # Manage existing position with technical indicators
        manage_position_with_indicators(symbol, current_price, "stock", df)
        
    except Exception as e:
        logger.error(f"Error trading stock {symbol}: {e}")

def fetch_symbol_data(symbol, symbol_type):
    """Fetch 5-minute data for any symbol"""
    try:
        start = datetime.now() - timedelta(days=7)
        
        if symbol_type == "crypto":
            request = CryptoBarsRequest(
                symbol_or_symbols=[symbol], 
                timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                start=start
            )
            bars_response = crypto_data_client.get_crypto_bars(request)
        else:
            request = StockBarsRequest(
                symbol_or_symbols=[symbol], 
                timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                start=start
            )
            bars_response = stock_data_client.get_stock_bars(request)
            
        df = bars_response.df
        
        if symbol in df.index.get_level_values(0):
            df = df.loc[symbol].copy()
        elif hasattr(df.index, 'get_level_values') and len(df.index.get_level_values(0)) > 0:
            df = df.xs(symbol, level=0) if symbol in df.index.get_level_values(0) else df
        
        if len(df) == 0:
            logger.warning(f"No 5-minute data for {symbol}")
            return None
            
        logger.debug(f"📊 [{symbol}] Fetched {len(df)} 5-minute bars")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return None

def check_entry_signal(df, symbol):
    """Check entry signal using MACD, RSI, and EMA indicators"""
    try:
        if len(df) < 30:  # Need more data for MACD
            return False, "Insufficient data"
            
        # Calculate indicators
        df['EMA_9'] = calculate_ema(df['close'], FAST_EMA)
        df['EMA_21'] = calculate_ema(df['close'], SLOW_EMA)
        
        macd, signal, histogram = calculate_macd(df['close'])
        df['MACD'] = macd
        df['MACD_Signal'] = signal
        df['MACD_Histogram'] = histogram
        
        df['RSI'] = calculate_rsi(df['close'])
        
        # Current values
        current_price = df['close'].iloc[-1]
        current_ema_9 = df['EMA_9'].iloc[-1]
        current_ema_21 = df['EMA_21'].iloc[-1]
        current_macd = df['MACD'].iloc[-1]
        current_signal = df['MACD_Signal'].iloc[-1]
        current_histogram = df['MACD_Histogram'].iloc[-1]
        current_rsi = df['RSI'].iloc[-1]
        
        # Previous values for crossover detection
        previous_ema_9 = df['EMA_9'].iloc[-2]
        previous_ema_21 = df['EMA_21'].iloc[-2]
        previous_macd = df['MACD'].iloc[-2]
        previous_signal = df['MACD_Signal'].iloc[-2]
        previous_histogram = df['MACD_Histogram'].iloc[-2]
        
        # Candle analysis
        current_candle = df.iloc[-1]
        current_green = current_candle['close'] > current_candle['open']
        
        # === ENTRY CONDITIONS (Multiple confirmation) ===
        # 1. EMA conditions
        ema_bullish = current_ema_9 > current_ema_21
        ema_crossover = (previous_ema_9 <= previous_ema_21) and ema_bullish
        
        # 2. MACD conditions
        macd_bullish = current_macd > current_signal
        macd_crossover = (previous_macd <= previous_signal) and macd_bullish
        macd_momentum = current_histogram > previous_histogram  # Increasing momentum
        
        # 3. RSI conditions
        rsi_oversold_bounce = 30 < current_rsi < 70  # Not overbought
        rsi_bullish = current_rsi > df['RSI'].iloc[-2]  # RSI increasing
        
        # Log all indicators
        logger.debug(f"🔍 [{symbol}] Price: ${current_price:.2f}")
        logger.debug(f"📊 [{symbol}] EMA: 9={current_ema_9:.2f}, 21={current_ema_21:.2f}, Bullish={ema_bullish}")
        logger.debug(f"📈 [{symbol}] MACD: {current_macd:.4f}, Signal: {current_signal:.4f}, Hist: {current_histogram:.4f}")
        logger.debug(f"📉 [{symbol}] RSI: {current_rsi:.1f}, Increasing={rsi_bullish}")
        
        # === ENTRY STRATEGIES (Less restrictive, multiple paths) ===
        
        # Strategy 1: Strong MACD crossover with RSI support
        if macd_crossover and rsi_oversold_bounce and current_green:
            return True, f"MACD crossover + RSI {current_rsi:.1f} + Green candle"
            
        # Strategy 2: EMA crossover with MACD confirmation
        if ema_crossover and macd_bullish and current_green:
            return True, f"EMA crossover + MACD bullish + Green candle"
            
        # Strategy 3: Oversold bounce (RSI recovery)
        if current_rsi > 35 and df['RSI'].iloc[-3] < 30 and macd_momentum and current_green:
            return True, f"RSI oversold bounce {current_rsi:.1f} + MACD momentum"
            
        # Strategy 4: Strong momentum play
        if ema_bullish and macd_bullish and macd_momentum and rsi_bullish and current_green:
            return True, f"Strong momentum: All indicators bullish"
        
        # No signal reasons
        reasons = []
        if not ema_bullish:
            reasons.append("EMA bearish")
        if not macd_bullish:
            reasons.append("MACD bearish")
        if current_rsi > 70:
            reasons.append(f"RSI overbought {current_rsi:.1f}")
        if not current_green:
            reasons.append("Red candle")
            
        return False, f"No signal: {', '.join(reasons) if reasons else 'Conditions not met'}"
            
    except Exception as e:
        logger.error(f"Error checking entry signal for {symbol}: {e}")
        return False, "Error in analysis"

def check_exit_signal(df, symbol, pnl_percent):
    """Check exit signal using MACD, RSI, and EMA indicators"""
    try:
        if len(df) < 30:
            return False, "Insufficient data for exit analysis"
            
        # Calculate indicators
        df['EMA_9'] = calculate_ema(df['close'], FAST_EMA)
        df['EMA_21'] = calculate_ema(df['close'], SLOW_EMA)
        
        macd, signal, histogram = calculate_macd(df['close'])
        df['MACD'] = macd
        df['MACD_Signal'] = signal
        df['MACD_Histogram'] = histogram
        
        df['RSI'] = calculate_rsi(df['close'])
        
        # Current values
        current_price = df['close'].iloc[-1]
        current_macd = df['MACD'].iloc[-1]
        current_signal = df['MACD_Signal'].iloc[-1]
        current_histogram = df['MACD_Histogram'].iloc[-1]
        current_rsi = df['RSI'].iloc[-1]
        
        # Previous values
        previous_macd = df['MACD'].iloc[-2]
        previous_signal = df['MACD_Signal'].iloc[-2]
        previous_histogram = df['MACD_Histogram'].iloc[-2]
        
        # === EXIT CONDITIONS ===
        
        # 1. RSI Overbought (take profits)
        if current_rsi > 75 and pnl_percent > 0.01:  # RSI very overbought + in profit
            return True, f"RSI overbought {current_rsi:.1f} - Take profit"
            
        # 2. MACD bearish crossover
        macd_bearish_cross = (previous_macd >= previous_signal) and (current_macd < current_signal)
        if macd_bearish_cross and pnl_percent > -0.01:  # MACD cross with small loss tolerance
            return True, f"MACD bearish crossover - Exit signal"
            
        # 3. Momentum loss (histogram declining)
        momentum_loss = (current_histogram < previous_histogram < df['MACD_Histogram'].iloc[-3])
        if momentum_loss and current_rsi > 60 and pnl_percent > 0:
            return True, f"Momentum declining - RSI {current_rsi:.1f}"
            
        # 4. EMA death cross (bearish)
        current_ema_9 = df['EMA_9'].iloc[-1]
        current_ema_21 = df['EMA_21'].iloc[-1]
        previous_ema_9 = df['EMA_9'].iloc[-2]
        previous_ema_21 = df['EMA_21'].iloc[-2]
        
        ema_death_cross = (previous_ema_9 >= previous_ema_21) and (current_ema_9 < current_ema_21)
        if ema_death_cross:
            return True, f"EMA death cross - Bearish signal"
            
        return False, "No exit signal"
        
    except Exception as e:
        logger.error(f"Error checking exit signal for {symbol}: {e}")
        return False, "Error in exit analysis"

def manage_position_with_indicators(symbol, current_price, symbol_type, df):
    """Enhanced position management with technical indicators"""
    try:
        try:
            position = trading_client.get_position(symbol)
            if not position:
                return False, f"No {symbol} position"
                
            # Get entry price and P&L
            if hasattr(position, 'avg_fill_price'):
                entry_price = float(position.avg_fill_price)
            elif hasattr(position, 'avg_entry_price'):
                entry_price = float(position.avg_entry_price)
            elif hasattr(position, 'cost_basis'):
                entry_price = float(position.cost_basis) / abs(float(position.qty))
            else:
                entry_price = abs(float(position.market_value)) / abs(float(position.qty))
                
            current_qty = float(position.qty)
            pnl_percent = (current_price - entry_price) / entry_price if current_qty > 0 else 0
            
            # Check technical exit signals
            should_exit, exit_reason = check_exit_signal(df, symbol, pnl_percent)
            
            if should_exit:
                logger.info(f"📊 [{symbol}] Technical exit signal: {exit_reason}")
                place_order(symbol, OrderSide.SELL, abs(current_qty), f"TECHNICAL EXIT - {exit_reason}")
                return True, f"Technical exit: {exit_reason}"
            
            # Continue with existing stop-loss/take-profit logic
            return manage_position(symbol, current_price, symbol_type)
            
        except Exception as e:
            # No position exists
            return False, f"No {symbol} position"
            
    except Exception as e:
        logger.error(f"Error in enhanced position management for {symbol}: {e}")
        return False, "Error"

def manage_position(symbol, current_price, symbol_type):
    """Manage existing position with trailing stop-loss support"""
    try:
        try:
            position = trading_client.get_position(symbol)
            if not position:
                return False, f"No {symbol} position"
                
            # Get entry price
            if hasattr(position, 'avg_fill_price'):
                entry_price = float(position.avg_fill_price)
            elif hasattr(position, 'avg_entry_price'):
                entry_price = float(position.avg_entry_price)
            elif hasattr(position, 'cost_basis'):
                entry_price = float(position.cost_basis) / abs(float(position.qty))
            else:
                entry_price = abs(float(position.market_value)) / abs(float(position.qty))
                
            current_qty = float(position.qty)
            
            if current_qty > 0:  # Long position
                pnl_percent = (current_price - entry_price) / entry_price
            else:  # Short position  
                pnl_percent = (entry_price - current_price) / entry_price
                
            # === TRAILING STOP-LOSS IMPLEMENTATION ===
            # Track highest price achieved (for longs) or lowest (for shorts)
            position_id = f"{symbol}_{entry_price}"  # Simple position identifier
            
            # Initialize or get tracking data (you'd want to persist this properly)
            if not hasattr(manage_position, 'trailing_data'):
                manage_position.trailing_data = {}
            
            if position_id not in manage_position.trailing_data:
                manage_position.trailing_data[position_id] = {
                    'highest_price': current_price if current_qty > 0 else entry_price,
                    'lowest_price': current_price if current_qty < 0 else entry_price,
                    'trailing_stop': None
                }
            
            tracking = manage_position.trailing_data[position_id]
            
            # Risk management parameters
            if symbol_type == "crypto":
                # Tighter stops for crypto (more volatile)
                take_profit_threshold = 0.03  # 3%
                initial_stop_loss = 0.015  # 1.5%
                trailing_stop_distance = 0.01  # Trail by 1%
            else:
                # Standard stops for stocks
                take_profit_threshold = 0.06  # 6%
                initial_stop_loss = 0.03  # 3%
                trailing_stop_distance = 0.02  # Trail by 2%
            
            # Update highest/lowest price
            if current_qty > 0:  # Long position
                if current_price > tracking['highest_price']:
                    tracking['highest_price'] = current_price
                    # Update trailing stop to maintain distance from new high
                    tracking['trailing_stop'] = current_price * (1 - trailing_stop_distance)
                    logger.info(f"📈 [{symbol}] New high ${current_price:.2f}, trailing stop updated to ${tracking['trailing_stop']:.2f}")
                
                # Calculate stop price (maximum of initial stop and trailing stop)
                initial_stop_price = entry_price * (1 - initial_stop_loss)
                effective_stop_price = max(initial_stop_price, tracking['trailing_stop'] or 0)
                
                # Check if stop should trigger
                if current_price <= effective_stop_price:
                    stop_type = "TRAILING STOP" if effective_stop_price > initial_stop_price else "STOP LOSS"
                    place_order(symbol, OrderSide.SELL, abs(current_qty), 
                              f"{stop_type} at ${current_price:.2f} (stop: ${effective_stop_price:.2f})")
                    del manage_position.trailing_data[position_id]  # Clean up tracking
                    return True, f"{stop_type} triggered"
                    
            else:  # Short position
                if current_price < tracking['lowest_price']:
                    tracking['lowest_price'] = current_price
                    # Update trailing stop for short position
                    tracking['trailing_stop'] = current_price * (1 + trailing_stop_distance)
                    logger.info(f"📉 [{symbol}] New low ${current_price:.2f}, trailing stop updated to ${tracking['trailing_stop']:.2f}")
                
                # Calculate stop price for short (minimum of initial stop and trailing stop)
                initial_stop_price = entry_price * (1 + initial_stop_loss)
                effective_stop_price = min(initial_stop_price, tracking['trailing_stop'] or float('inf'))
                
                # Check if stop should trigger for short
                if current_price >= effective_stop_price:
                    stop_type = "TRAILING STOP" if effective_stop_price < initial_stop_price else "STOP LOSS"
                    place_order(symbol, OrderSide.BUY, abs(current_qty), 
                              f"{stop_type} at ${current_price:.2f} (stop: ${effective_stop_price:.2f})")
                    del manage_position.trailing_data[position_id]  # Clean up tracking
                    return True, f"{stop_type} triggered"
            
            # Log position status with trailing info
            logger.info(f"📊 [{symbol}] Entry: ${entry_price:.2f}, Current: ${current_price:.2f}, P&L: {pnl_percent:.2%}")
            if tracking['trailing_stop']:
                logger.info(f"    🎯 Trailing stop: ${tracking['trailing_stop']:.2f}, High: ${tracking['highest_price']:.2f}")
            
            # === SCALING OUT / PARTIAL PROFIT TAKING ===
            # Define profit levels for scaling out
            if symbol_type == "crypto":
                scale_out_levels = [
                    (0.015, 0.33),  # At 1.5% profit, sell 33%
                    (0.025, 0.50),  # At 2.5% profit, sell 50% of remaining
                    (0.035, 0.50),  # At 3.5% profit, sell 50% of remaining
                ]
            else:  # stocks
                scale_out_levels = [
                    (0.03, 0.25),   # At 3% profit, sell 25%
                    (0.05, 0.33),   # At 5% profit, sell 33% of remaining
                    (0.08, 0.50),   # At 8% profit, sell 50% of remaining
                ]
            
            # Track which levels have been executed
            if 'scale_out_executed' not in tracking:
                tracking['scale_out_executed'] = []
            
            # Check for scale-out opportunities
            for level_idx, (profit_threshold, sell_portion) in enumerate(scale_out_levels):
                level_key = f"level_{level_idx}"
                
                if pnl_percent >= profit_threshold and level_key not in tracking['scale_out_executed']:
                    # Calculate quantity to sell (portion of current position)
                    qty_to_sell = abs(current_qty) * sell_portion
                    
                    # Round appropriately
                    if symbol_type == "crypto":
                        qty_to_sell = round(qty_to_sell, 6)  # Crypto allows fractional
                    else:
                        qty_to_sell = max(1, int(qty_to_sell))  # Stocks need whole shares
                    
                    # Only execute if we have enough quantity
                    if qty_to_sell > 0 and qty_to_sell <= abs(current_qty):
                        place_order(symbol, OrderSide.SELL if current_qty > 0 else OrderSide.BUY,
                                   qty_to_sell, f"SCALE OUT {int(sell_portion*100)}% at {pnl_percent:.2%} profit")
                        tracking['scale_out_executed'].append(level_key)
                        logger.info(f"💰 [{symbol}] Scaled out {qty_to_sell} shares at {pnl_percent:.2%} profit")
                        
                        # Don't delete tracking - position still open
                        return True, f"Partial profit taken at {pnl_percent:.2%}"
            
            # Full take profit check (for remaining position or if scaling not triggered)
            if pnl_percent >= take_profit_threshold:
                place_order(symbol, OrderSide.SELL if current_qty > 0 else OrderSide.BUY, 
                           abs(current_qty), f"TAKE PROFIT (final) at {pnl_percent:.2%}")
                if position_id in manage_position.trailing_data:
                    del manage_position.trailing_data[position_id]  # Clean up tracking
                return True, "Take profit triggered"
                
            return False, f"Position held - P&L: {pnl_percent:.2%}"
            
        except Exception as e:
            logger.warning(f"No {symbol} position found: {e}")
            return False, f"No {symbol} position found"
            
    except Exception as e:
        logger.error(f"Error managing {symbol} position: {e}")
        return False, "Error"

def place_order(symbol, side, quantity, reason):
    """Place order for any symbol - with cash verification"""
    try:
        if quantity <= 0:
            return None
        
        # For BUY orders, verify we have sufficient cash
        if side == OrderSide.BUY:
            account = trading_client.get_account()
            available_cash = float(account.cash)
            
            # Get current price for the symbol
            if "/" in symbol:  # Crypto
                bars_request = CryptoBarsRequest(
                    symbol_or_symbols=[symbol],
                    timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                    limit=1
                )
                bars = crypto_data_client.get_crypto_bars(bars_request)
            else:  # Stock
                bars_request = StockBarsRequest(
                    symbol_or_symbols=[symbol],
                    timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                    limit=1
                )
                bars = stock_data_client.get_stock_bars(bars_request)
            
            if bars and not bars.df.empty:
                current_price = bars.df['close'].iloc[-1]
                estimated_cost = quantity * current_price
                
                if estimated_cost > available_cash:
                    logger.error(f"❌ MARGIN PREVENTION: Order cost ${estimated_cost:.2f} > Cash ${available_cash:.2f}")
                    logger.error(f"   Blocking order to prevent margin usage")
                    return None
            
        order = MarketOrderRequest(symbol=symbol, qty=quantity, side=side, time_in_force=TimeInForce.GTC)
        result = trading_client.submit_order(order)
        logger.info(f"🚀 [{symbol}] {side.value.upper()}: {quantity} - {reason} - Order ID: {result.id}")
        return result
        
    except Exception as e:
        logger.error(f"Error placing {symbol} order: {e}")
        return None


def check_existing_positions():
    """Check and log existing positions"""
    try:
        positions = trading_client.get_all_positions()
        if positions:
            logger.info(f"📋 Found {len(positions)} existing positions:")
            for pos in positions:
                pnl = float(pos.unrealized_pl)
                pnl_pct = float(pos.unrealized_plpc) * 100
                symbol_emoji = "₿" if pos.symbol.endswith("USD") else "📈"
                logger.info(f"   {symbol_emoji} {pos.symbol}: {pos.qty} @ ${pos.avg_entry_price} (P&L: ${pnl:.2f}, {pnl_pct:.1f}%)")
        else:
            logger.info("📋 No existing positions found")
    except Exception as e:
        logger.warning(f"Could not check existing positions: {e}")

if __name__ == "__main__":
    main()
