#!/bin/bash
# Health check for swing-trader bot on ut.gitsum.rest

HOST="ut.gitsum.rest"

echo "=== Swing Trader Health Check ==="
echo ""

# Container status
echo "Container:"
ssh $HOST 'podman ps --filter name=swing-trader --format "  Status: {{.Status}}"' 2>/dev/null

echo ""
echo "Latest sync:"
ssh $HOST 'podman logs --tail 100 swing-trader-bot 2>&1 | grep "Synced:" | tail -1' 2>/dev/null

echo ""
echo "Recent activity:"
ssh $HOST 'podman logs --tail 20 swing-trader-bot 2>&1 | grep -E "(Signal|Exit|Filled|Error|position)" | tail -5' 2>/dev/null

echo ""
echo "Jobs last run:"
ssh $HOST 'podman logs --tail 200 swing-trader-bot 2>&1 | grep -E "Checking (stock|crypto)|Refreshing watchlist|monitor" | tail -5' 2>/dev/null
