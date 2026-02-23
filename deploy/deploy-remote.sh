#!/bin/bash
# Deploy swing-trader to remote Podman server
# Usage: ./deploy-remote.sh <remote-host> [--build]

set -e

REMOTE_HOST="${1:-}"
BUILD_FLAG="${2:-}"

if [ -z "$REMOTE_HOST" ]; then
    echo "Usage: ./deploy-remote.sh <user@remote-host> [--build]"
    echo "Example: ./deploy-remote.sh jacob@trading-server --build"
    exit 1
fi

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_DIR="/opt/swing-trader"

echo "=== Deploying swing-trader to $REMOTE_HOST ==="

# Create remote directory structure
echo "Creating remote directories..."
ssh "$REMOTE_HOST" "sudo mkdir -p $REMOTE_DIR && sudo chown \$(whoami) $REMOTE_DIR"

# Sync project files (excluding venv, __pycache__, etc.)
echo "Syncing files..."
rsync -avz --delete \
    --exclude 'venv/' \
    --exclude '.venv/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '.git/' \
    --exclude 'logs/' \
    --exclude 'state/' \
    --exclude '.env' \
    "$PROJECT_DIR/" "$REMOTE_HOST:$REMOTE_DIR/"

# Copy .env only if remote doesn't have one (never overwrite server secrets)
if ssh "$REMOTE_HOST" "[ ! -f $REMOTE_DIR/.env ]" 2>/dev/null; then
    if [ -f "$PROJECT_DIR/.env" ]; then
        echo "Copying .env file (first deploy)..."
        scp "$PROJECT_DIR/.env" "$REMOTE_HOST:$REMOTE_DIR/.env"
    else
        echo "WARNING: No .env file found locally or on server"
    fi
else
    echo "Using existing .env on server (not overwriting)"
fi

# Build and run on remote
echo "Building and starting container..."
ssh "$REMOTE_HOST" << EOF
    cd $REMOTE_DIR/deploy

    # Stop existing container if running
    podman-compose down 2>/dev/null || true

    # Build if requested or if image doesn't exist
    if [ "$BUILD_FLAG" = "--build" ] || ! podman image exists swing-trader-bot:latest; then
        echo "Building image..."
        podman-compose build
    fi

    # Start the container
    echo "Starting container..."
    podman-compose up -d

    # Show status
    echo ""
    echo "=== Container Status ==="
    podman ps -a --filter "name=swing-trader"

    echo ""
    echo "=== Recent Logs ==="
    sleep 2
    podman logs --tail 20 swing-trader-bot
EOF

echo ""
echo "=== Deployment Complete ==="
echo "View logs: ssh $REMOTE_HOST 'podman logs -f swing-trader-bot'"
echo "Stop: ssh $REMOTE_HOST 'cd $REMOTE_DIR/deploy && podman-compose down'"
