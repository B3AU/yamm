#!/usr/bin/env python3
"""Main orchestration script for live trading bot."""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading.config import TradingConfig, IBConfig, StrategyConfig, DataConfig
from trading.data_pipeline import DataPipeline, load_cached_fundamentals
from trading.model import ModelInference
from trading.ib_client import IBClient
from trading.strategy import ShortStrategy
from trading.risk import RiskManager, CircuitBreaker


# Setup logging
def setup_logging(log_level: str = "INFO", log_file: Path | None = None):
    """Configure logging."""
    handlers = [logging.StreamHandler()]

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )


logger = logging.getLogger(__name__)


class TradingBot:
    """Main trading bot orchestrator."""

    def __init__(self, config: TradingConfig):
        self.config = config

        # Initialize components
        logger.info("Initializing trading bot...")

        # Data pipeline
        self.data_pipeline = DataPipeline(config.data)

        # Model
        if config.data.model_path.exists():
            self.model = ModelInference(config.data.model_path)
        else:
            logger.warning(f"Model not found at {config.data.model_path}")
            self.model = None

        # IB client
        self.ib_client = IBClient(config.ib)

        # Strategy
        self.strategy = ShortStrategy(config.strategy, self.ib_client)

        # Risk management
        self.risk_manager = RiskManager(config.strategy, self.ib_client)
        self.circuit_breaker = CircuitBreaker()

        # State
        self.is_connected = False
        self.universe: list[str] = []

    def connect(self) -> bool:
        """Connect to IB."""
        if self.ib_client.connect_sync():
            self.is_connected = True
            logger.info("Connected to Interactive Brokers")

            # Get account info
            account = self.ib_client.get_account_values()
            logger.info(f"Account value: ${account.get('NetLiquidation', 0):,.0f}")

            # Initialize circuit breaker
            self.circuit_breaker.reset_all(account.get("NetLiquidation", 0))

            return True
        return False

    def disconnect(self):
        """Disconnect from IB."""
        if self.is_connected:
            self.ib_client.disconnect()
            self.is_connected = False

    def load_universe(self) -> list[str]:
        """Load trading universe."""
        self.universe = self.data_pipeline.load_universe()
        logger.info(f"Loaded universe: {len(self.universe)} symbols")
        return self.universe

    def prepare_candidates(self) -> pd.DataFrame:
        """Prepare candidate stocks with features and scores."""
        if not self.universe:
            self.load_universe()

        # Fetch quotes and compute features
        logger.info("Fetching market data...")
        features_df = self.data_pipeline.prepare_features(
            symbols=self.universe,
            fundamentals_df=load_cached_fundamentals(self.config.data.data_dir),
        )

        if features_df.empty:
            logger.error("Failed to prepare features")
            return pd.DataFrame()

        # Score with model
        if self.model:
            logger.info("Scoring candidates...")
            features_df = self.model.rank_stocks(features_df, ascending=True)
        else:
            logger.warning("No model loaded, using random scores")
            features_df["score"] = 0

        # Apply risk filters
        candidates = self.risk_manager.apply_filters(features_df)

        logger.info(f"Prepared {len(candidates)} candidates")
        return candidates

    def run_trading_cycle(self, dry_run: bool = False) -> dict:
        """Run one trading cycle (rebalance if needed)."""
        logger.info("=" * 50)
        logger.info("Starting trading cycle")
        logger.info("=" * 50)

        # Check circuit breaker
        current_value = self.ib_client.get_net_liquidation()
        if self.circuit_breaker.update(current_value):
            logger.critical("Circuit breaker triggered - halting trading")
            return {"status": "circuit_breaker", "value": current_value}

        # Check if rebalance is needed
        if not self.strategy.should_rebalance():
            logger.info("No rebalance needed (within hold period)")
            return {"status": "skip", "reason": "hold_period"}

        # Prepare candidates
        candidates = self.prepare_candidates()
        if candidates.empty:
            return {"status": "error", "reason": "no_candidates"}

        # Run pre-trade checks
        passed, checks = self.risk_manager.run_pre_trade_checks(
            candidates, current_value
        )
        for check in checks:
            level = logging.INFO if check.passed else logging.WARNING
            logger.log(level, f"Risk check: {check.message}")

        if not passed:
            logger.warning("Pre-trade risk checks failed")
            if not dry_run:
                return {"status": "risk_check_failed", "checks": checks}

        # Execute rebalance
        result = self.strategy.rebalance(candidates, dry_run=dry_run)

        logger.info(f"Trading cycle complete: {result.get('status')}")
        return result

    def check_stop_losses(self) -> list:
        """Check and execute stop-losses."""
        triggers = self.risk_manager.check_stop_losses()

        if not triggers:
            return []

        logger.warning(f"Stop-loss triggered for {len(triggers)} positions")

        # Close triggered positions
        for symbol, loss_pct, action in triggers:
            logger.warning(f"Closing {symbol} due to {action}: {loss_pct*100:.1f}% loss")
            # Execute close order
            positions = self.ib_client.get_portfolio_df()
            if symbol in positions:
                qty = positions[symbol]["quantity"]
                if qty < 0:  # Short position
                    self.ib_client.place_market_order(symbol, -qty, "BUY")
                else:
                    self.ib_client.place_market_order(symbol, qty, "SELL")

        return triggers

    def status(self) -> dict:
        """Get current bot status."""
        if not self.is_connected:
            return {"status": "disconnected"}

        state = self.strategy.get_current_state()
        return {
            "status": "connected",
            "is_paper": self.config.ib.use_paper,
            "portfolio_value": state["total_value"],
            "available_funds": state["available_funds"],
            "n_positions": state["n_positions"],
            "short_exposure_pct": state["short_pct"] * 100,
            "positions": state["positions"],
            "last_rebalance": self.strategy.last_rebalance_date,
            "circuit_breaker": self.circuit_breaker.is_triggered,
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Trading bot for short strategy")
    parser.add_argument(
        "--paper", action="store_true", default=True,
        help="Use paper trading account (default: True)"
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Use live trading account (WARNING: real money)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Don't execute trades, just show what would happen"
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Show current status and exit"
    )
    parser.add_argument(
        "--rebalance", action="store_true",
        help="Run rebalance now"
    )
    parser.add_argument(
        "--close-all", action="store_true",
        help="Close all positions"
    )
    parser.add_argument(
        "--log-level", default="INFO",
        help="Logging level"
    )

    args = parser.parse_args()

    # Determine paper vs live
    use_paper = not args.live

    if not use_paper:
        confirm = input("WARNING: You are about to trade with REAL MONEY. Type 'CONFIRM' to proceed: ")
        if confirm != "CONFIRM":
            print("Aborted.")
            return

    # Setup logging
    setup_logging(args.log_level)

    # Create config
    config = TradingConfig(
        ib=IBConfig(use_paper=use_paper),
        strategy=StrategyConfig(),
        data=DataConfig(
            fmp_api_key=os.environ.get("FMP_API_KEY", ""),
            data_dir=Path("data"),
            model_path=Path("data/model_final.pt"),
        ),
    )

    # Create bot
    bot = TradingBot(config)

    try:
        # Connect
        if not bot.connect():
            logger.error("Failed to connect to IB")
            return

        if args.status:
            # Show status and exit
            status = bot.status()
            print("\n" + "=" * 50)
            print("TRADING BOT STATUS")
            print("=" * 50)
            for key, value in status.items():
                if key == "positions":
                    print(f"\nPositions:")
                    for sym, pos in value.items():
                        print(f"  {sym}: {pos['quantity']} @ ${pos['market_price']:.2f}")
                else:
                    print(f"  {key}: {value}")
            return

        if args.close_all:
            # Close all positions
            result = bot.strategy.close_all_positions(dry_run=args.dry_run)
            print(f"Close all result: {result}")
            return

        if args.rebalance:
            # Run rebalance
            result = bot.run_trading_cycle(dry_run=args.dry_run)
            print(f"Rebalance result: {result}")
            return

        # Default: show status
        status = bot.status()
        print(f"\nBot Status: {status['status']}")
        print(f"Portfolio Value: ${status.get('portfolio_value', 0):,.0f}")
        print(f"Positions: {status.get('n_positions', 0)}")
        print(f"\nUse --rebalance to run a trading cycle")
        print("Use --status for detailed status")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.exception(f"Error: {e}")
    finally:
        bot.disconnect()


if __name__ == "__main__":
    main()
