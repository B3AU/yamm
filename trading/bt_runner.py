"""Unified runner for backtest and live trading with backtrader."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import backtrader as bt
import pandas as pd
import numpy as np

from trading.config import TradingConfig, DEFAULT_CONFIG
from trading.model import ModelInference
from trading.bt_strategy import ShortRankerStrategy, ShortRankerStrategyLive
from trading.bt_data import add_data_feeds, prepare_backtest_features, load_features_data


logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    final_value: float
    total_return: float
    sharpe_ratio: float | None
    max_drawdown: float
    n_trades: int
    analyzers: dict = field(default_factory=dict)
    trades: list = field(default_factory=list)


def run_backtest(
    config: TradingConfig = DEFAULT_CONFIG,
    model_path: Path | str | None = None,
    data_dir: Path | str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    initial_cash: float = 100_000,
    use_dd_scaling: bool = True,
    use_confidence: bool = False,
    max_symbols: int | None = None,
    plot: bool = False,
) -> BacktestResult:
    """Run backtest using backtrader.

    Args:
        config: Trading configuration
        model_path: Path to trained model checkpoint
        data_dir: Path to data directory with prices.pqt and ml_dataset.pqt
        start_date: Backtest start date (YYYY-MM-DD)
        end_date: Backtest end date (YYYY-MM-DD)
        initial_cash: Starting capital
        use_dd_scaling: Enable drawdown-based position scaling
        use_confidence: Enable confidence-weighted sizing
        max_symbols: Limit number of symbols (for testing)
        plot: Generate plot after backtest

    Returns:
        BacktestResult with performance metrics
    """
    # Resolve paths
    data_dir = Path(data_dir) if data_dir else config.data.data_dir
    model_path = Path(model_path) if model_path else config.data.model_path

    logger.info(f"Starting backtest")
    logger.info(f"  Model: {model_path}")
    logger.info(f"  Data: {data_dir}")
    logger.info(f"  Period: {start_date} to {end_date}")

    # Load model
    model = ModelInference(model_path)

    # Create cerebro
    cerebro = bt.Cerebro()

    # Add data feeds for all symbols
    symbols = add_data_feeds(
        cerebro=cerebro,
        data_dir=data_dir,
        symbols=None,  # Load all symbols
        start_date=start_date,
        end_date=end_date,
        max_symbols=max_symbols,
    )

    if not symbols:
        raise ValueError("No data feeds added")

    # Prepare features
    features_df = prepare_backtest_features(
        data_dir=data_dir,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
    )

    if features_df.empty:
        raise ValueError("No features loaded")

    # Add strategy
    cerebro.addstrategy(
        ShortRankerStrategy,
        # Position sizing
        k_short=config.strategy.k_short,
        max_position_pct=config.strategy.max_position_pct,
        max_portfolio_short=config.strategy.max_portfolio_short,
        # Holding period
        hold_days=config.strategy.hold_days,
        # Drawdown scaling
        use_dd_scaling=use_dd_scaling,
        dd_threshold=0.10,
        dd_max=0.20,
        dd_min_scale=0.25,
        # Confidence
        use_confidence=use_confidence,
        # Stop loss
        stop_loss_pct=config.strategy.stop_loss_pct,
        # Model and features
        model=model,
        features_df=features_df,
    )

    # Set broker parameters
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=config.strategy.fee_per_share)

    # Enable short selling
    cerebro.broker.set_shortcash(True)

    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=0.0)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")

    # Calculate expected trading days for progress estimation
    n_trading_days = features_df["date"].nunique()
    logger.info(f"Running backtest: {n_trading_days} trading days, {len(symbols):,} symbols...")
    logger.info(f"  Initial cash: ${initial_cash:,.0f}")

    # Run backtest
    results = cerebro.run()
    strat = results[0]

    # Extract results
    final_value = cerebro.broker.getvalue()
    total_return = (final_value - initial_cash) / initial_cash

    # Analyzers
    try:
        sharpe = strat.analyzers.sharpe.get_analysis().get("sharperatio")
    except Exception:
        sharpe = None

    try:
        drawdown = strat.analyzers.drawdown.get_analysis()
        max_dd = drawdown.get("max", {}).get("drawdown", 0) / 100
    except Exception:
        max_dd = 0

    try:
        trade_analysis = strat.analyzers.trades.get_analysis()
        n_trades = trade_analysis.get("total", {}).get("total", 0)
    except Exception:
        n_trades = 0

    logger.info(f"Backtest complete")
    logger.info(f"  Final value: ${final_value:,.2f}")
    logger.info(f"  Total return: {total_return*100:.1f}%")
    logger.info(f"  Sharpe ratio: {sharpe:.2f}" if sharpe else "  Sharpe ratio: N/A")
    logger.info(f"  Max drawdown: {max_dd*100:.1f}%")
    logger.info(f"  Total trades: {n_trades}")

    # Plot if requested
    if plot:
        cerebro.plot(style="candlestick")

    return BacktestResult(
        final_value=final_value,
        total_return=total_return,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        n_trades=n_trades,
        analyzers={
            "sharpe": strat.analyzers.sharpe.get_analysis(),
            "drawdown": strat.analyzers.drawdown.get_analysis(),
            "trades": strat.analyzers.trades.get_analysis(),
            "returns": strat.analyzers.returns.get_analysis(),
        },
    )


def run_live(
    config: TradingConfig = DEFAULT_CONFIG,
    model_path: Path | str | None = None,
    data_dir: Path | str | None = None,
    paper: bool = True,
    use_dd_scaling: bool = True,
    use_confidence: bool = False,
):
    """Run live trading via Interactive Brokers.

    Args:
        config: Trading configuration
        model_path: Path to trained model checkpoint
        data_dir: Path to data directory
        paper: Use paper trading (True) or live (False)
        use_dd_scaling: Enable drawdown-based position scaling
        use_confidence: Enable confidence-weighted sizing
    """
    try:
        from backtrader_ib_insync import IBStore
    except ImportError:
        try:
            # Fallback to bt's built-in IB store
            IBStore = bt.stores.IBStore
        except AttributeError:
            raise ImportError(
                "IB integration requires backtrader_ib_insync or ib_insync. "
                "Install with: pip install backtrader-ib-insync"
            )

    # Resolve paths
    data_dir = Path(data_dir) if data_dir else config.data.data_dir
    model_path = Path(model_path) if model_path else config.data.model_path

    logger.info(f"Starting live trading")
    logger.info(f"  Paper mode: {paper}")
    logger.info(f"  Model: {model_path}")

    # Load model
    model = ModelInference(model_path)

    # Create cerebro
    cerebro = bt.Cerebro()

    # IB connection
    port = config.ib.paper_port if paper else config.ib.live_port
    store = IBStore(
        host=config.ib.host,
        port=port,
        clientId=config.ib.client_id,
    )

    # Load universe
    from trading.data_pipeline import DataPipeline, load_cached_fundamentals

    pipeline = DataPipeline(config.data)
    symbols = pipeline.load_universe()

    logger.info(f"Trading universe: {len(symbols)} symbols")

    # Add data feeds from IB
    for symbol in symbols[:50]:  # Limit for IB data limits
        try:
            data = store.getdata(
                dataname=symbol,
                sectype="STK",
                exchange="SMART",
                currency="USD",
                historical=True,
                what="TRADES",
            )
            cerebro.adddata(data, name=symbol)
        except Exception as e:
            logger.warning(f"Failed to add data for {symbol}: {e}")

    # Load cached fundamentals
    fundamentals_df = load_cached_fundamentals(data_dir)

    # Add live strategy
    cerebro.addstrategy(
        ShortRankerStrategyLive,
        # Position sizing
        k_short=config.strategy.k_short,
        max_position_pct=config.strategy.max_position_pct,
        max_portfolio_short=config.strategy.max_portfolio_short,
        # Holding period
        hold_days=config.strategy.hold_days,
        # Drawdown scaling
        use_dd_scaling=use_dd_scaling,
        dd_threshold=0.10,
        dd_max=0.20,
        dd_min_scale=0.25,
        # Confidence
        use_confidence=use_confidence,
        # Stop loss
        stop_loss_pct=config.strategy.stop_loss_pct,
        # Model
        model=model,
        # Data pipeline for live data fetching
        data_pipeline=pipeline,
        fundamentals_df=fundamentals_df,
    )

    # Set broker to IB
    cerebro.setbroker(store.getbroker())

    # Run
    logger.info("Starting live trading loop...")
    cerebro.run()


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Backtrader runner for short strategy")
    parser.add_argument("mode", choices=["backtest", "paper", "live"], help="Running mode")
    parser.add_argument("--model", type=str, help="Path to model checkpoint")
    parser.add_argument("--data", type=str, help="Path to data directory")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--cash", type=float, default=100_000, help="Initial cash")
    parser.add_argument("--k", type=int, default=5, help="Number of shorts (K)")
    parser.add_argument("--no-dd-scale", action="store_true", help="Disable DD scaling")
    parser.add_argument("--confidence", action="store_true", help="Enable confidence weighting")
    parser.add_argument("--plot", action="store_true", help="Plot results")
    parser.add_argument("--max-symbols", type=int, help="Limit symbols (for testing)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load config
    config = DEFAULT_CONFIG

    if args.mode == "backtest":
        result = run_backtest(
            config=config,
            model_path=args.model,
            data_dir=args.data,
            start_date=args.start,
            end_date=args.end,
            initial_cash=args.cash,
            use_dd_scaling=not args.no_dd_scale,
            use_confidence=args.confidence,
            max_symbols=args.max_symbols,
            plot=args.plot,
        )
        print(f"\n=== Backtest Results ===")
        print(f"Final Value:  ${result.final_value:,.2f}")
        print(f"Total Return: {result.total_return*100:.1f}%")
        print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}" if result.sharpe_ratio else "Sharpe Ratio: N/A")
        print(f"Max Drawdown: {result.max_drawdown*100:.1f}%")
        print(f"Total Trades: {result.n_trades}")

    elif args.mode == "paper":
        run_live(
            config=config,
            model_path=args.model,
            data_dir=args.data,
            paper=True,
            use_dd_scaling=not args.no_dd_scale,
            use_confidence=args.confidence,
        )

    elif args.mode == "live":
        # Safety confirmation
        confirm = input("WARNING: Live trading with real money. Type 'YES' to confirm: ")
        if confirm != "YES":
            print("Aborted.")
            return

        run_live(
            config=config,
            model_path=args.model,
            data_dir=args.data,
            paper=False,
            use_dd_scaling=not args.no_dd_scale,
            use_confidence=args.confidence,
        )


if __name__ == "__main__":
    main()
