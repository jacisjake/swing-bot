#!/bin/bash

# Simple deployment script for Simple Swinger Bot
# Run this on your VPS after cloning the repository

set -e

echo "🚀 Starting Simple Swinger Bot Deployment"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose not found. Please install Docker Compose first."
    exit 1
fi

# Create logs directory
mkdir -p logs
chmod 755 logs

echo "📁 Created logs directory"

# Build the Docker image
echo "🏗️  Building Docker image..."
docker-compose build

# Start the service
echo "▶️  Starting Simple Swinger Bot..."
docker-compose up -d

# Wait a moment for startup
sleep 5

# Check if container is running
if docker ps | grep -q "alpaca-swing-bot"; then
    echo "✅ Simple Swinger Bot is running successfully!"
    echo "📊 View logs with: docker logs -f alpaca-swing-bot"
    echo "🛑 Stop with: docker-compose down"
    echo "🔄 Restart with: docker-compose restart"
else
    echo "❌ Bot failed to start. Check logs:"
    docker logs alpaca-swing-bot
    exit 1
fi

echo "🎉 Deployment complete!"
echo ""
echo "Next steps:"
echo "1. Monitor logs: docker logs -f alpaca-swing-bot"
echo "2. Check Alpaca web interface for trades"
echo "3. Set up monitoring alerts"
echo ""
echo "⚠️  Remember: This is LIVE TRADING with real money!"
