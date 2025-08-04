<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# Simple Swinger - Alpaca Trading Bot

This is a minimal, production-ready Alpaca swing trading bot using Docker and Portainer for deployment.

## Key Components
- **swing_trader.py**: Main trading logic with SMA crossover strategy
- **Dockerfile**: Containerization for production deployment
- **docker-compose.yml**: Orchestration with Portainer support
- **Requirements**: Minimal dependencies focused on Alpaca API and scheduling

## Development Guidelines
- **Security First**: Never commit API keys or sensitive data
- **Live Trading**: This bot trades with real money - emphasize safety
- **Error Handling**: All functions should have comprehensive error handling
- **Logging**: Use Python logging module for all output, not print statements
- **Docker Focus**: Prioritize containerized deployment patterns

## Trading Strategy
- Simple SMA crossover (10-day vs 20-day)
- Runs every 10 minutes during market hours
- Market orders for immediate execution
- NVDA as default symbol (configurable)

## Deployment Pattern
- Use Portainer stacks for deployment
- Volume mount for persistent logs
- Environment variables for configuration
- Production-ready with restart policies
