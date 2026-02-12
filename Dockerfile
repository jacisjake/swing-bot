FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/
COPY config/ config/
COPY scripts/ scripts/

# Create directories for state and logs
RUN mkdir -p /app/state /app/logs

# Environment
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose dashboard port
EXPOSE 8080

# Run the bot with dashboard
CMD ["python", "scripts/run_bot.py"]
