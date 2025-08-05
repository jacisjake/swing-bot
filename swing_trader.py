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

# LTC Scalping Strategy Settings
MAX_PORTFOLIO_PERCENT = float(os.getenv("MAX_PORTFOLIO_PERCENT", "0.10"))  # 10% max per asset
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
    """Generic position tracker for any symbol"""
    def __init__(self, symbol, entry_price, quantity, order_id):
        self.symbol = symbol
        self.entry_price = entry_price
        self.quantity = quantity
        self.order_id = order_id

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
    """Calculate position size per symbol with equal allocation"""
    try:
        portfolio_value = get_portfolio_value()
        
        if symbol_type == "stock":
            total_allocation = MAX_PORTFOLIO_PERCENT * 0.6  # 60% of total for stocks
            symbols_count = len(active_stock_symbols)
        else:  # crypto
            total_allocation = MAX_PORTFOLIO_PERCENT * 0.4  # 40% of total for crypto
            symbols_count = len(active_crypto_symbols)
            
        per_symbol_allocation = total_allocation / symbols_count if symbols_count > 0 else 0
        max_investment = portfolio_value * per_symbol_allocation
        
        if symbol_type == "stock":
            # For stocks, use whole shares
            max_shares = int(max_investment / current_price)
            position_size = max(1, max_shares)  # At least 1 share
        else:
            # For crypto, allow fractional shares
            max_shares = max_investment / current_price
            position_size = max(0.01, round(max_shares, 2))  # At least 0.01
            
        logger.debug(f"💼 {symbol_type.upper()} position size: {position_size} (${max_investment:.2f} allocation)")
        return position_size
        
    except Exception as e:
        logger.error(f"Error calculating {symbol_type} position size: {e}")
        return 1 if symbol_type == "stock" else 0.01

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
    logger.info(f"📈 Strategy: EMA {FAST_EMA}/{SLOW_EMA}, Portfolio limit: {MAX_PORTFOLIO_PERCENT:.0%}")
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
            
        # Manage existing position
        manage_position(symbol, current_price, "crypto")
        
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
            
        # Manage existing position
        manage_position(symbol, current_price, "stock")
        
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
    """Check entry signal for any symbol using EMA crossover"""
    try:
        if len(df) < SLOW_EMA + 2:
            return False, "Insufficient data"
            
        df['EMA_9'] = calculate_ema(df['close'], FAST_EMA)
        df['EMA_21'] = calculate_ema(df['close'], SLOW_EMA)
        
        current_ema_9 = df['EMA_9'].iloc[-1]
        current_ema_21 = df['EMA_21'].iloc[-1]
        previous_ema_9 = df['EMA_9'].iloc[-2]
        previous_ema_21 = df['EMA_21'].iloc[-2]
        
        current_candle = df.iloc[-1]
        previous_candle = df.iloc[-2]
        
        # Entry conditions
        ema_crossover = (previous_ema_9 <= previous_ema_21) and (current_ema_9 > current_ema_21)
        previous_green = previous_candle['close'] > previous_candle['open']
        current_green = current_candle['close'] > current_candle['open']
        
        logger.debug(f"🔍 [{symbol}] EMA 9: {current_ema_9:.4f}, EMA 21: {current_ema_21:.4f}")
        logger.debug(f"🔍 [{symbol}] Crossover: {ema_crossover}, Prev Green: {previous_green}, Curr Green: {current_green}")
        
        if ema_crossover and previous_green and current_green:
            return True, "EMA crossover with bullish candles"
        else:
            return False, "Entry conditions not satisfied"
            
    except Exception as e:
        logger.error(f"Error checking entry signal for {symbol}: {e}")
        return False, "Error in analysis"

def manage_position(symbol, current_price, symbol_type):
    """Manage existing position for any symbol"""
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
                
            logger.debug(f"📊 [{symbol}] Position: Entry ${entry_price:.2f}, Current ${current_price:.2f}, P&L: {pnl_percent:.2%}")
            
            # Risk management
            if symbol_type == "crypto":
                # Tighter stops for crypto (more volatile)
                take_profit_threshold = 0.03  # 3%
                stop_loss_threshold = -0.015  # 1.5%
            else:
                # Standard stops for stocks
                take_profit_threshold = 0.06  # 6%
                stop_loss_threshold = -0.03  # 3%
                
            if pnl_percent >= take_profit_threshold:
                place_order(symbol, OrderSide.SELL if current_qty > 0 else OrderSide.BUY, 
                           abs(current_qty), f"TAKE PROFIT at {pnl_percent:.2%}")
                return True, "Take profit triggered"
            elif pnl_percent <= stop_loss_threshold:
                place_order(symbol, OrderSide.SELL if current_qty > 0 else OrderSide.BUY,
                           abs(current_qty), f"STOP LOSS at {pnl_percent:.2%}")
                return True, "Stop loss triggered"
                
            return False, f"Position within range - P&L: {pnl_percent:.2%}"
            
        except Exception:
            return False, f"No {symbol} position found"
            
    except Exception as e:
        logger.error(f"Error managing {symbol} position: {e}")
        return False, "Error"

def place_order(symbol, side, quantity, reason):
    """Place order for any symbol"""
    try:
        if quantity <= 0:
            return None
            
        order = MarketOrderRequest(symbol=symbol, qty=quantity, side=side, time_in_force=TimeInForce.GTC)
        result = trading_client.submit_order(order)
        logger.info(f"🚀 [{symbol}] {side.value.upper()}: {quantity} - {reason} - Order ID: {result.id}")
        return result
        
    except Exception as e:
        logger.error(f"Error placing {symbol} order: {e}")
        return None

def calculate_position_size_per_symbol(price, symbol_type):
    """Calculate position size per symbol based on portfolio allocation"""
    try:
        account = trading_client.get_account()
        portfolio_value = float(account.portfolio_value)
        
        if symbol_type == "crypto":
            max_symbols = MAX_CRYPTO_SYMBOLS
        else:
            max_symbols = MAX_STOCK_SYMBOLS
            
        # Total allocation per asset class
        if symbol_type == "crypto":
            total_allocation = portfolio_value * (MAX_PORTFOLIO_PERCENT / 2)  # 50% of allocation for crypto
        else:
            total_allocation = portfolio_value * (MAX_PORTFOLIO_PERCENT / 2)  # 50% of allocation for stocks
            
        # Per symbol allocation
        per_symbol_allocation = total_allocation / max_symbols
        
        if symbol_type == "crypto":
            quantity = per_symbol_allocation / price
            return round(quantity, 6)  # Crypto can have fractional shares
        else:
            quantity = int(per_symbol_allocation / price)
            return quantity if quantity >= 1 else 0  # Stocks need whole shares
            
    except Exception as e:
        logger.error(f"Error calculating position size: {e}")
        return 0

def is_market_hours():
    """Check if market is open"""
    try:
        clock = trading_client.get_clock()
        return clock.is_open
    except Exception as e:
        logger.warning(f"Could not check market hours: {e}")
        # Fallback to basic hours check
        now = datetime.now().time()
        return time(9, 30) <= now <= time(16, 0)  # 9:30 AM to 4:00 PM EST

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
