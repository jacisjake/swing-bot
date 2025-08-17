#!/usr/bin/env python3
"""
Quick script to check current positions and test position management logic
"""
import os
from alpaca.trading.client import TradingClient

# Load API credentials from .env file
def load_env():
    env_vars = {}
    try:
        with open('.env', 'r') as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    env_vars[key] = value
    except FileNotFoundError:
        print("❌ .env file not found!")
    return env_vars

env_vars = load_env()
API_KEY = env_vars.get("ALPACA_LIVE_API_KEY")
SECRET_KEY = env_vars.get("ALPACA_LIVE_API_SECRET")

print(f"🔑 API Key: {API_KEY[:8]}..." if API_KEY else "❌ No API Key")
print(f"🔐 Secret: {'*' * 10}" if SECRET_KEY else "❌ No Secret Key")

if not API_KEY or not SECRET_KEY:
    print("❌ API keys not found!")
    exit(1)

# Initialize trading client
try:
    trading_client = TradingClient(API_KEY, SECRET_KEY, paper=False)
    print("✅ Trading client initialized")
except Exception as e:
    print(f"❌ Failed to initialize trading client: {e}")
    exit(1)

try:
    print("🔍 Checking current positions...")
    
    # Get account info
    account = trading_client.get_account()
    print(f"💼 Portfolio Value: ${float(account.portfolio_value):.2f}")
    print(f"💰 Buying Power: ${float(account.buying_power):.2f}")
    
    # Get all positions
    positions = trading_client.get_all_positions()
    
    if not positions:
        print("📭 No current positions found")
    else:
        print(f"\n📊 Found {len(positions)} position(s):")
        print("-" * 60)
        
        for position in positions:
            symbol = position.symbol
            qty = float(position.qty)
            market_value = float(position.market_value)
            unrealized_pl = float(position.unrealized_pl)
            unrealized_plpc = float(position.unrealized_plpc) * 100
            
            # Get entry price
            entry_price = None
            if hasattr(position, 'avg_fill_price'):
                entry_price = float(position.avg_fill_price)
            elif hasattr(position, 'avg_entry_price'):
                entry_price = float(position.avg_entry_price)
            elif hasattr(position, 'cost_basis'):
                entry_price = float(position.cost_basis) / abs(qty)
            else:
                entry_price = abs(market_value) / abs(qty)
            
            side = "LONG" if qty > 0 else "SHORT"
            symbol_type = "crypto" if "/" in symbol else "stock"
            
            # Calculate thresholds
            if symbol_type == "crypto":
                take_profit_threshold = 3.0  # 3%
                stop_loss_threshold = -1.5   # 1.5%
            else:
                take_profit_threshold = 6.0  # 6%
                stop_loss_threshold = -3.0   # 3%
            
            print(f"📈 {symbol} ({symbol_type.upper()}): {side} {abs(qty)} @ ${entry_price:.2f}")
            print(f"   💰 Market Value: ${market_value:.2f}")
            print(f"   📊 P&L: ${unrealized_pl:.2f} ({unrealized_plpc:+.2f}%)")
            print(f"   🎯 Take Profit at: +{take_profit_threshold}% | Stop Loss at: {stop_loss_threshold}%")
            
            # Check if exit conditions are met
            if unrealized_plpc >= take_profit_threshold:
                print(f"   🚀 SHOULD TAKE PROFIT! ({unrealized_plpc:.2f}% >= {take_profit_threshold}%)")
            elif unrealized_plpc <= stop_loss_threshold:
                print(f"   🚨 SHOULD STOP LOSS! ({unrealized_plpc:.2f}% <= {stop_loss_threshold}%)")
            else:
                print(f"   ⏳ Within range ({stop_loss_threshold}% < {unrealized_plpc:.2f}% < {take_profit_threshold}%)")
            
            print("-" * 60)

except Exception as e:
    print(f"❌ Error checking positions: {e}")
