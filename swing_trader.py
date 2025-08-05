import os
import pandas as pd
import logging
import time
import schedule
import pytz
from datetime import datetime, timedelta
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
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

# Trading symbols
STOCK_SYMBOL = os.getenv("STOCK_SYMBOL", "NVDA")
CRYPTO_SYMBOL = os.getenv("CRYPTO_SYMBOL", "LTC/USD")
STOCK_QTY = int(os.getenv("STOCK_QUANTITY", "1"))

# LTC Scalping Strategy Settings
MAX_PORTFOLIO_PERCENT = float(os.getenv("MAX_PORTFOLIO_PERCENT", "0.10"))  # 10% max per asset
STOP_LOSS_PERCENT = float(os.getenv("STOP_LOSS_PERCENT", "0.005"))  # 0.5% stop loss
TAKE_PROFIT_1 = float(os.getenv("TAKE_PROFIT_1", "0.005"))  # 0.5% first target
TAKE_PROFIT_2 = float(os.getenv("TAKE_PROFIT_2", "0.010"))  # 1.0% second target  
TAKE_PROFIT_3 = float(os.getenv("TAKE_PROFIT_3", "0.015"))  # 1.5% third target

# Initialize clients (LIVE trading)
stock_data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
crypto_data_client = CryptoHistoricalDataClient(API_KEY, SECRET_KEY)
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=False)

# EMA Strategy parameters
FAST_EMA = 9
SLOW_EMA = 21

# Global position tracking for LTC scalping
ltc_positions = []

class LTCPosition:
    def __init__(self, entry_price, initial_quantity, order_id):
        self.entry_price = entry_price
        self.initial_quantity = initial_quantity
        self.remaining_quantity = initial_quantity
        self.order_id = order_id
        self.exit_1_filled = False
        self.exit_2_filled = False
        self.exit_3_filled = False
        self.stop_loss_price = entry_price * (1 - STOP_LOSS_PERCENT)
        
    def get_target_prices(self):
        return {
            'target_1': self.entry_price * (1 + TAKE_PROFIT_1),
            'target_2': self.entry_price * (1 + TAKE_PROFIT_2), 
            'target_3': self.entry_price * (1 + TAKE_PROFIT_3),
            'stop_loss': self.stop_loss_price
        }

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
# LTC SCALPING STRATEGY (24/7)
# =============================================================================

def calculate_ltc_position_size(current_price):
    """Calculate LTC position size based on 10% portfolio limit"""
    try:
        portfolio_value = get_portfolio_value()
        max_investment = portfolio_value * MAX_PORTFOLIO_PERCENT
        
        # Get current LTC exposure
        current_ltc_value = 0
        try:
            ltc_position = trading_client.get_position(CRYPTO_SYMBOL)
            if ltc_position:
                current_ltc_value = float(ltc_position.market_value)
        except:
            pass
            
        available_investment = max_investment - current_ltc_value
        
        if available_investment <= 0:
            logger.info(f"💼 LTC position limit reached. Current exposure: ${current_ltc_value:.2f}")
            return 0
            
        max_shares = available_investment / current_price
        position_size = max(0.01, round(max_shares, 2))
        
        logger.info(f"💼 Portfolio: ${portfolio_value:.2f}, Max LTC: ${max_investment:.2f}, Available: ${available_investment:.2f}")
        return position_size
        
    except Exception as e:
        logger.error(f"Error calculating LTC position size: {e}")
        return 0.1

def fetch_ltc_data():
    """Fetch 5-minute LTC data"""
    try:
        start = datetime.now() - timedelta(days=7)
        
        logger.info(f"🔗 [LTC] Requesting data from {start} to now")
        
        request = CryptoBarsRequest(
            symbol_or_symbols=[CRYPTO_SYMBOL], 
            timeframe=TimeFrame(5, TimeFrameUnit.Minute),
            start=start
        )
        
        logger.info(f"🔗 [LTC] Making API call for {CRYPTO_SYMBOL}")
        bars_response = crypto_data_client.get_crypto_bars(request)
        df = bars_response.df
        
        if CRYPTO_SYMBOL in df.index.get_level_values(0):
            df = df.loc[CRYPTO_SYMBOL].copy()
        elif hasattr(df.index, 'get_level_values') and len(df.index.get_level_values(0)) > 0:
            df = df.xs(CRYPTO_SYMBOL, level=0) if CRYPTO_SYMBOL in df.index.get_level_values(0) else df
        
        if len(df) == 0:
            logger.error(f"No 5-minute data for {CRYPTO_SYMBOL}")
            return None
            
        logger.info(f"📊 [LTC] Fetched {len(df)} 5-minute bars")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching LTC data: {e}")
        return None

def check_ltc_entry_signal(df):
    """Check for LTC scalping entry signal"""
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
        
        logger.info(f"🔍 [LTC] EMA 9: {current_ema_9:.4f}, EMA 21: {current_ema_21:.4f}")
        logger.info(f"🔍 [LTC] Crossover: {ema_crossover}, Prev Green: {previous_green}, Curr Green: {current_green}")
        
        if ema_crossover and previous_green and current_green:
            return True, "All entry conditions met"
        else:
            return False, "Entry conditions not satisfied"
            
    except Exception as e:
        logger.error(f"Error checking LTC entry: {e}")
        return False, "Error in analysis"

def manage_ltc_positions(current_price):
    """Manage existing LTC positions with scaled exits"""
    global ltc_positions
    
    try:
        for position in ltc_positions[:]:
            targets = position.get_target_prices()
            
            # Stop loss check
            if current_price <= targets['stop_loss']:
                if position.remaining_quantity > 0:
                    place_ltc_order(OrderSide.SELL, position.remaining_quantity, 
                                  f"STOP LOSS at ${current_price:.4f}")
                    ltc_positions.remove(position)
                continue
                
            # Take profit levels
            if not position.exit_1_filled and current_price >= targets['target_1']:
                exit_qty = round(position.initial_quantity * 0.33, 2)
                place_ltc_order(OrderSide.SELL, exit_qty, f"TP1 (33%) at ${current_price:.4f}")
                position.remaining_quantity -= exit_qty
                position.exit_1_filled = True
                
            elif not position.exit_2_filled and current_price >= targets['target_2']:
                exit_qty = round(position.initial_quantity * 0.33, 2)
                place_ltc_order(OrderSide.SELL, exit_qty, f"TP2 (33%) at ${current_price:.4f}")
                position.remaining_quantity -= exit_qty
                position.exit_2_filled = True
                
            elif not position.exit_3_filled and current_price >= targets['target_3']:
                place_ltc_order(OrderSide.SELL, position.remaining_quantity, f"TP3 (34%) at ${current_price:.4f}")
                ltc_positions.remove(position)
                
            # Log position status
            pnl = (current_price - position.entry_price) / position.entry_price * 100
            logger.info(f"📊 [LTC] Position: Entry ${position.entry_price:.4f}, Remaining: {position.remaining_quantity}, P&L: {pnl:.2f}%")
            
    except Exception as e:
        logger.error(f"Error managing LTC positions: {e}")

def place_ltc_order(side, quantity, reason):
    """Place LTC order"""
    try:
        if quantity <= 0:
            return None
            
        order = MarketOrderRequest(symbol=CRYPTO_SYMBOL, qty=quantity, side=side, time_in_force=TimeInForce.GTC)
        result = trading_client.submit_order(order)
        logger.info(f"🚀 [LTC] {side.value.upper()}: {quantity} - {reason} - Order ID: {result.id}")
        return result
        
    except Exception as e:
        logger.error(f"Error placing LTC order: {e}")
        return None

def run_ltc_scalping():
    """Run LTC 5-minute scalping strategy"""
    global ltc_positions
    
    try:
        logger.info("🔄 [LTC] Running 5-minute scalping cycle")
        
        df = fetch_ltc_data()
        if df is None:
            logger.warning("⚠️ [LTC] No data available - skipping cycle")
            return
            
        current_price = df['close'].iloc[-1]
        logger.info(f"💰 [LTC] Current Price: ${current_price:.4f}")
        
        # Manage existing positions
        manage_ltc_positions(current_price)
        
        # Check for new entry
        should_enter, reason = check_ltc_entry_signal(df)
        
        if should_enter:
            position_size = calculate_ltc_position_size(current_price)
            
            if position_size > 0:
                result = place_ltc_order(OrderSide.BUY, position_size, f"SCALP ENTRY - {reason}")
                
                if result:
                    new_position = LTCPosition(current_price, position_size, result.id)
                    ltc_positions.append(new_position)
                    
                    targets = new_position.get_target_prices()
                    logger.info(f"🎯 [LTC] Targets: TP1: ${targets['target_1']:.4f}, TP2: ${targets['target_2']:.4f}, TP3: ${targets['target_3']:.4f}, SL: ${targets['stop_loss']:.4f}")
            else:
                logger.info("⚠️ [LTC] Portfolio limit reached")
        else:
            logger.info(f"⏳ [LTC] No entry signal - {reason}")
            
        logger.info(f"📈 [LTC] Active positions: {len(ltc_positions)}")
        
    except Exception as e:
        logger.error(f"Error in LTC scalping: {e}")
        logger.info("⚠️ [LTC] Continuing with reduced functionality...")

# =============================================================================
# NVDA STOCK STRATEGY (Market Hours Only)
# =============================================================================

def fetch_nvda_data():
    """Fetch 5-minute NVDA data"""
    try:
        start = datetime.now() - timedelta(days=7)
        
        request = StockBarsRequest(
            symbol_or_symbols=[STOCK_SYMBOL], 
            timeframe=TimeFrame(5, TimeFrameUnit.Minute),
            start=start
        )
        bars_response = stock_data_client.get_stock_bars(request)
        df = bars_response.df
        
        if STOCK_SYMBOL in df.index.get_level_values(0):
            df = df.loc[STOCK_SYMBOL].copy()
        elif hasattr(df.index, 'get_level_values') and len(df.index.get_level_values(0)) > 0:
            df = df.xs(STOCK_SYMBOL, level=0) if STOCK_SYMBOL in df.index.get_level_values(0) else df
        
        if len(df) == 0:
            logger.error(f"No 5-minute data for {STOCK_SYMBOL}")
            return None
            
        logger.info(f"📊 [NVDA] Fetched {len(df)} 5-minute bars")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching NVDA data: {e}")
        return None

def check_nvda_signal(df):
    """Check NVDA trading signal using EMA crossover"""
    try:
        if len(df) < SLOW_EMA:
            return 0, "Insufficient data"
            
        df['EMA_9'] = calculate_ema(df['close'], FAST_EMA)
        df['EMA_21'] = calculate_ema(df['close'], SLOW_EMA)
        
        current_ema_9 = df['EMA_9'].iloc[-1]
        current_ema_21 = df['EMA_21'].iloc[-1]
        current_price = df['close'].iloc[-1]
        
        logger.info(f"🔍 [NVDA] Price: ${current_price:.2f}, EMA 9: {current_ema_9:.2f}, EMA 21: {current_ema_21:.2f}")
        
        if current_ema_9 > current_ema_21:
            return 1, "Bullish - EMA 9 > EMA 21"
        elif current_ema_9 < current_ema_21:
            return -1, "Bearish - EMA 9 < EMA 21"
        else:
            return 0, "Neutral"
            
    except Exception as e:
        logger.error(f"Error analyzing NVDA: {e}")
        return 0, "Error"

def manage_nvda_position(current_price):
    """Manage NVDA position with 6% TP / 3% SL"""
    try:
        try:
            position = trading_client.get_position(STOCK_SYMBOL)
            if not position:
                return False, "No NVDA position"
                
            # Try different attribute names for entry price
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
                
            logger.info(f"📊 [NVDA] Position: Entry ${entry_price:.2f}, Current ${current_price:.2f}, P&L: {pnl_percent:.2%}")
            
            # Risk management (6% TP / 3% SL)
            if pnl_percent >= 0.06:  # 6% take profit
                place_nvda_order(OrderSide.SELL if current_qty > 0 else OrderSide.BUY, 
                               abs(current_qty), f"TAKE PROFIT at {pnl_percent:.2%}")
                return True, "Take profit triggered"
            elif pnl_percent <= -0.03:  # 3% stop loss
                place_nvda_order(OrderSide.SELL if current_qty > 0 else OrderSide.BUY,
                               abs(current_qty), f"STOP LOSS at {pnl_percent:.2%}")
                return True, "Stop loss triggered"
                
            return False, f"Position within range - P&L: {pnl_percent:.2%}"
            
        except Exception:
            return False, "No NVDA position found"
            
    except Exception as e:
        logger.error(f"Error managing NVDA position: {e}")
        return False, "Error"

def place_nvda_order(side, quantity, reason):
    """Place NVDA order"""
    try:
        if quantity <= 0:
            return None
            
        order = MarketOrderRequest(symbol=STOCK_SYMBOL, qty=quantity, side=side, time_in_force=TimeInForce.GTC)
        result = trading_client.submit_order(order)
        logger.info(f"🚀 [NVDA] {side.value.upper()}: {quantity} shares - {reason} - Order ID: {result.id}")
        return result
        
    except Exception as e:
        logger.error(f"Error placing NVDA order: {e}")
        return None

def run_nvda_trading():
    """Run NVDA stock trading (market hours only)"""
    try:
        if not is_market_hours():
            logger.info("📅 [NVDA] Market closed")
            return
            
        logger.info("🔄 [NVDA] Running stock trading cycle")
        
        df = fetch_nvda_data()
        if df is None:
            return
            
        current_price = df['close'].iloc[-1]
        
        # Check existing position first
        position_exited, exit_reason = manage_nvda_position(current_price)
        if position_exited:
            logger.info(f"🔄 [NVDA] {exit_reason}")
            return
        
        # Check for new signals
        signal, reason = check_nvda_signal(df)
        
        if signal != 0:
            # Check current position
            try:
                position = trading_client.get_position(STOCK_SYMBOL)
                current_qty = float(position.qty) if position else 0
            except:
                current_qty = 0
                
            # Determine if we should place order
            if signal == 1 and current_qty <= 0:  # Buy signal and no long position
                place_nvda_order(OrderSide.BUY, STOCK_QTY, f"BUY signal - {reason}")
            elif signal == -1 and current_qty >= 0:  # Sell signal and no short position
                place_nvda_order(OrderSide.SELL, STOCK_QTY, f"SELL signal - {reason}")
            else:
                logger.info(f"[NVDA] Signal {signal} ignored due to existing position")
        else:
            logger.info(f"✅ [NVDA] No signal - {reason}")
        
    except Exception as e:
        logger.error(f"Error in NVDA trading: {e}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function with dual-strategy execution"""
    os.makedirs('/app/logs', exist_ok=True)
    
    logger.info("🚀 Starting Dual-Asset Trading Bot")
    logger.info(f"📊 LTC 5-min scalping (24/7): EMA {FAST_EMA}/{SLOW_EMA}, Portfolio limit: {MAX_PORTFOLIO_PERCENT:.0%}")
    logger.info(f"📈 NVDA 5-min trading (market hours): EMA {FAST_EMA}/{SLOW_EMA}, 6% TP / 3% SL")
    logger.info(f"💰 LTC Targets: {TAKE_PROFIT_1:.1%}, {TAKE_PROFIT_2:.1%}, {TAKE_PROFIT_3:.1%} | SL: {STOP_LOSS_PERCENT:.1%}")
    logger.info("⚠️  LIVE TRADING MODE - Real money at risk!")
    
    # Validate API access before starting
    logger.info("🔍 Validating API access...")
    if not validate_api_access():
        logger.error("❌ API validation failed. Exiting.")
        return
    
    # Check existing positions
    check_existing_positions()
    
    # Run initial cycles
    run_ltc_scalping()
    time.sleep(150)  # 2.5 minute offset
    run_nvda_trading()
    
    # Schedule both strategies
    schedule.every(5).minutes.do(run_ltc_scalping)      # LTC every 5 minutes
    schedule.every(5).minutes.do(run_nvda_trading)      # NVDA every 5 minutes (2.5 min offset)
    
    # Keep running
    while True:
        schedule.run_pending()
        time.sleep(30)

if __name__ == "__main__":
    main()
