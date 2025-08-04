import sys
import subprocess

print("Python version:", sys.version)
print("Installed packages:")
result = subprocess.run([sys.executable, "-m", "pip", "list"], capture_output=True, text=True)
print(result.stdout)

print("\nTrying to import alpaca modules...")
try:
    import alpaca
    print("✅ alpaca module found")
    print("alpaca module path:", alpaca.__file__)
    print("alpaca module contents:", dir(alpaca))
except ImportError as e:
    print("❌ alpaca module not found:", e)

try:
    from alpaca.data.historical import StockHistoricalDataClient
    print("✅ StockHistoricalDataClient import successful")
except ImportError as e:
    print("❌ StockHistoricalDataClient import failed:", e)

try:
    from alpaca.trading.client import TradingClient
    print("✅ TradingClient import successful")  
except ImportError as e:
    print("❌ TradingClient import failed:", e)
