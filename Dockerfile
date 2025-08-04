FROM python:3.11-slim

WORKDIR /app

# Install system dependencies that might be needed
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Create logs directory
RUN mkdir -p /app/logs

# Force cache bust - Update timestamp
RUN echo "Build timestamp: $(date)" > /tmp/build-timestamp

# Upgrade pip and install requirements
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY debug_imports.py .
COPY swing_trader.py .

# Run the trading bot
CMD ["python", "swing_trader.py"]
