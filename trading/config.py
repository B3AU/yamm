"""Configuration for live trading bot."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class IBConfig:
    """Interactive Brokers connection settings."""
    host: str = "127.0.0.1"
    paper_port: int = 7497  # TWS paper trading
    live_port: int = 7496   # TWS live trading
    client_id: int = 1

    # Use paper trading by default
    use_paper: bool = True

    @property
    def port(self) -> int:
        return self.paper_port if self.use_paper else self.live_port


@dataclass(frozen=True)
class StrategyConfig:
    """Trading strategy parameters.

    Best backtest results (Simple K=5):
    - Sharpe: 4.03
    - Total Return: 648.5% over 178 days
    - Win rate: 60.7%
    - Max drawdown: -28.7%
    """
    # Position sizing (K=5 performed best)
    k_short: int = 5                     # Number of stocks to short
    initial_capital: float = 100_000     # Starting capital
    max_position_pct: float = 0.20       # Max 20% per position (100%/5)

    # Holding period (daily rebalance performed best)
    hold_days: int = 1                   # Daily rebalance

    # Filters
    min_market_cap: float = 500_000_000  # $500M minimum
    min_price: float = 5.0               # $5 minimum price
    min_volume: float = 500_000          # 500K shares avg volume

    # Risk limits
    max_portfolio_short: float = 1.0     # Max 100% short exposure
    stop_loss_pct: float = 0.15          # 15% stop loss per position

    # Fees (for tracking)
    fee_per_share: float = 0.005         # $0.005 per share
    max_fee_pct: float = 0.01            # Max 1% of trade value


@dataclass(frozen=True)
class DataConfig:
    """Data source settings."""
    # FMP API
    fmp_api_key: str = ""  # Set via environment variable

    # Paths
    data_dir: Path = field(default_factory=lambda: Path("data"))
    model_path: Path = field(default_factory=lambda: Path("data/model_final.pt"))

    # Universe
    universe_file: Path = field(default_factory=lambda: Path("data/universe.csv"))

    # Feature columns (must match training)
    price_feat_cols: tuple[str, ...] = (
        "overnight_gap_z", "intraday_ret_z",
        "ret_1d_z", "ret_2d_z", "ret_3d_z", "ret_5d_z",
        "vol_5d_z", "dist_from_high_5d_z", "dist_from_low_5d_z",
    )


@dataclass(frozen=True)
class TradingConfig:
    """Combined configuration."""
    ib: IBConfig = field(default_factory=IBConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    data: DataConfig = field(default_factory=DataConfig)

    # Scheduling (matches training data: 15:30 ET cutoff, close-to-close returns)
    trade_time_et: str = "15:30"  # 30 min before close
    timezone: str = "US/Eastern"

    # Logging
    log_level: str = "INFO"
    log_file: Path = field(default_factory=lambda: Path("trading/logs/trading.log"))


# Default config - override via environment or config file
DEFAULT_CONFIG = TradingConfig()
