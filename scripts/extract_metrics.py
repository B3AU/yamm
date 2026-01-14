#!/usr/bin/env python3
"""Extract and track metrics from ML pipeline runs.

Usage:
    # Extract metrics from latest run
    python3 scripts/extract_metrics.py

    # Extract from specific run directory
    python3 scripts/extract_metrics.py --run-dir notebooks/runs/20260114_122228

    # Show metrics history
    python3 scripts/extract_metrics.py --history
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path
import argparse

import pandas as pd
import numpy as np


DATA_DIR = Path(__file__).parent.parent / "data" / "earnings"
MODELS_DIR = Path(__file__).parent.parent / "models"
METRICS_FILE = MODELS_DIR / "metrics_history.json"


def get_git_info() -> dict:
    """Get current git commit and branch."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()

        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()

        # Check for uncommitted changes
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = len(status) > 0

        return {
            "commit": commit,
            "branch": branch,
            "dirty": dirty,
        }
    except Exception:
        return {"commit": "unknown", "branch": "unknown", "dirty": True}


def compute_sharpe(trades_df: pd.DataFrame) -> float:
    """Compute annualized Sharpe ratio from trades."""
    if len(trades_df) == 0:
        return 0.0

    daily_pnl = trades_df.groupby('earnings_date')['pnl'].sum()
    if len(daily_pnl) < 2 or daily_pnl.std() == 0:
        return 0.0

    n_days = len(daily_pnl)
    date_range = (trades_df['earnings_date'].max() - trades_df['earnings_date'].min()).days
    trades_per_year = n_days * 252 / date_range if date_range > 0 else 100

    return (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(trades_per_year)


def simulate_strategy(df: pd.DataFrame,
                     edge_threshold: float = 0.0,
                     implied_move_multiplier: float = 1.0,
                     spread_cost_pct: float = 0.03,
                     max_trades_per_day: int = 5) -> pd.DataFrame:
    """Simulate straddle trading strategy."""
    df = df.copy()

    # Implied move with multiplier
    df['implied_move'] = df['hist_move_mean'] * implied_move_multiplier
    df['edge'] = df['pred_q75'] - df['implied_move']
    df['tradeable'] = df['edge'] > edge_threshold

    # Limit trades per day
    df = df.sort_values(['earnings_date', 'edge'], ascending=[True, False])
    df['trade_rank'] = df.groupby('earnings_date').cumcount() + 1
    df['take_trade'] = df['tradeable'] & (df['trade_rank'] <= max_trades_per_day)

    # P&L
    commission_pct = (1.30 * 2 * 2) / 5000  # IBKR commissions
    df['payoff'] = df['target_move']
    df['total_cost'] = df['implied_move'] + spread_cost_pct + commission_pct
    df['pnl'] = df['payoff'] - df['total_cost']

    return df


def extract_metrics(oos_path: Path = None) -> dict:
    """Extract metrics from OOS predictions."""
    if oos_path is None:
        oos_path = DATA_DIR / "oos_predictions.parquet"

    if not oos_path.exists():
        raise FileNotFoundError(f"OOS predictions not found: {oos_path}")

    oos = pd.read_parquet(oos_path)
    oos['earnings_date'] = pd.to_datetime(oos['earnings_date'])

    metrics = {
        "timestamp": datetime.now().isoformat(),
        "git": get_git_info(),
        "data": {
            "oos_samples": len(oos),
            "date_range_start": str(oos['earnings_date'].min().date()),
            "date_range_end": str(oos['earnings_date'].max().date()),
        },
        "calibration": {},
        "strategy_1.0x": {},
        "strategy_1.3x": {},
    }

    # Calibration metrics
    for q in [0.50, 0.75, 0.90, 0.95]:
        col = f'pred_q{int(q*100)}'
        if col in oos.columns:
            expected = 1 - q
            actual = (oos['target_move'] > oos[col]).mean()
            metrics["calibration"][f"q{int(q*100)}_exceedance"] = round(actual, 4)
            metrics["calibration"][f"q{int(q*100)}_error"] = round(actual - expected, 4)

    # Strategy metrics at different thresholds
    for multiplier, key in [(1.0, "strategy_1.0x"), (1.3, "strategy_1.3x")]:
        for threshold in [0.05, 0.06, 0.07, 0.08, 0.10]:
            sim = simulate_strategy(
                oos,
                edge_threshold=threshold,
                implied_move_multiplier=multiplier,
            )
            trades = sim[sim['take_trade']].copy()

            if len(trades) > 0:
                sharpe = compute_sharpe(trades)
                thresh_key = f"{int(threshold*100)}pct"
                metrics[key][thresh_key] = {
                    "trades": len(trades),
                    "mean_pnl": round(trades['pnl'].mean(), 4),
                    "win_rate": round((trades['pnl'] > 0).mean(), 4),
                    "sharpe": round(sharpe, 2),
                }

    return metrics


def load_history() -> list:
    """Load metrics history from file."""
    if METRICS_FILE.exists():
        with open(METRICS_FILE) as f:
            return json.load(f)
    return []


def save_metrics(metrics: dict):
    """Save metrics to history file."""
    history = load_history()
    history.append(metrics)

    # Keep last 100 entries
    if len(history) > 100:
        history = history[-100:]

    METRICS_FILE.parent.mkdir(exist_ok=True)
    with open(METRICS_FILE, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"Metrics saved to {METRICS_FILE}")


def print_metrics(metrics: dict):
    """Print metrics in a readable format."""
    print("\n" + "=" * 60)
    print("ML PIPELINE METRICS")
    print("=" * 60)

    print(f"\nTimestamp: {metrics['timestamp']}")
    print(f"Git: {metrics['git']['commit']} ({metrics['git']['branch']})"
          + (" [dirty]" if metrics['git']['dirty'] else ""))

    print(f"\nData:")
    print(f"  OOS samples: {metrics['data']['oos_samples']}")
    print(f"  Date range: {metrics['data']['date_range_start']} to {metrics['data']['date_range_end']}")

    print(f"\nCalibration (exceedance rate, error):")
    for q in [50, 75, 90, 95]:
        exc = metrics['calibration'].get(f'q{q}_exceedance', 'N/A')
        err = metrics['calibration'].get(f'q{q}_error', 'N/A')
        expected = (100 - q) / 100
        print(f"  q{q}: {exc:.1%} (expected {expected:.0%}, error {err:+.1%})"
              if isinstance(exc, float) else f"  q{q}: N/A")

    print(f"\nStrategy Performance (1.3x realistic):")
    print(f"  {'Threshold':<10} {'Trades':<8} {'Mean P&L':<10} {'Win Rate':<10} {'Sharpe':<8}")
    print(f"  {'-'*46}")
    for thresh in ['5pct', '6pct', '7pct', '8pct', '10pct']:
        data = metrics['strategy_1.3x'].get(thresh, {})
        if data:
            print(f"  {thresh:<10} {data['trades']:<8} {data['mean_pnl']:+.2%}     "
                  f"{data['win_rate']:.1%}      {data['sharpe']:.2f}")

    print("\n" + "=" * 60)


def print_history():
    """Print metrics history comparison."""
    history = load_history()
    if not history:
        print("No metrics history found.")
        return

    print("\n" + "=" * 70)
    print("METRICS HISTORY (1.3x realistic, 6% threshold)")
    print("=" * 70)
    print(f"{'Date':<12} {'Commit':<8} {'OOS':<8} {'Trades':<8} {'Sharpe':<8} {'q75 err':<10}")
    print("-" * 70)

    for entry in history[-10:]:  # Last 10 entries
        date = entry['timestamp'][:10]
        commit = entry['git']['commit'][:7]
        oos = entry['data']['oos_samples']

        strat = entry.get('strategy_1.3x', {}).get('6pct', {})
        trades = strat.get('trades', 'N/A')
        sharpe = strat.get('sharpe', 'N/A')

        q75_err = entry.get('calibration', {}).get('q75_error', 'N/A')

        sharpe_str = f"{sharpe:.2f}" if isinstance(sharpe, (int, float)) else str(sharpe)
        q75_str = f"{q75_err:+.2%}" if isinstance(q75_err, (int, float)) else str(q75_err)

        print(f"{date:<12} {commit:<8} {oos:<8} {trades:<8} {sharpe_str:<8} {q75_str:<10}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Extract and track ML pipeline metrics")
    parser.add_argument("--run-dir", type=Path, help="Specific run directory to extract from")
    parser.add_argument("--history", action="store_true", help="Show metrics history")
    parser.add_argument("--no-save", action="store_true", help="Don't save to history file")

    args = parser.parse_args()

    if args.history:
        print_history()
        return

    # Determine OOS path
    if args.run_dir:
        # Look for OOS predictions in run directory or use default
        oos_path = args.run_dir / "oos_predictions.parquet"
        if not oos_path.exists():
            oos_path = DATA_DIR / "oos_predictions.parquet"
    else:
        oos_path = DATA_DIR / "oos_predictions.parquet"

    print(f"Extracting metrics from: {oos_path}")

    metrics = extract_metrics(oos_path)
    print_metrics(metrics)

    if not args.no_save:
        save_metrics(metrics)


if __name__ == "__main__":
    main()
