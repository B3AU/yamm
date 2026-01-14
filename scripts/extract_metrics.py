#!/usr/bin/env python3
"""Extract and track comprehensive metrics from ML pipeline runs.

Usage:
    # Extract metrics from latest run
    python3 scripts/extract_metrics.py

    # Extract from specific run directory
    python3 scripts/extract_metrics.py --run-dir notebooks/runs/20260114_122228

    # Show metrics history
    python3 scripts/extract_metrics.py --history

    # Full bootstrap analysis (slower)
    python3 scripts/extract_metrics.py --bootstrap
"""

import json
import hashlib
import subprocess
from datetime import datetime
from pathlib import Path
import argparse

import pandas as pd
import numpy as np


DATA_DIR = Path(__file__).parent.parent / "data" / "earnings"
MODELS_DIR = Path(__file__).parent.parent / "models"
METRICS_FILE = MODELS_DIR / "metrics_history.json"

# Bootstrap settings
N_BOOTSTRAP = 500
BOOTSTRAP_CI = 0.95


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


def get_model_hashes() -> dict:
    """Get MD5 hashes of model files for traceability."""
    hashes = {}
    for q in [50, 75, 90, 95]:
        model_file = MODELS_DIR / f"earnings_q{q}.txt"
        if model_file.exists():
            with open(model_file, 'rb') as f:
                hashes[f"q{q}"] = hashlib.md5(f.read()).hexdigest()[:8]
    return hashes


def compute_sharpe(daily_pnl: pd.Series, trades_per_year: float = None) -> float:
    """Compute annualized Sharpe ratio from daily P&L series."""
    if len(daily_pnl) < 2 or daily_pnl.std() == 0:
        return 0.0

    if trades_per_year is None:
        trades_per_year = len(daily_pnl) * 252 / 365  # rough estimate

    return (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(trades_per_year)


def compute_max_drawdown(cum_pnl: pd.Series) -> tuple:
    """Compute max drawdown and duration."""
    high_water = cum_pnl.cummax()
    drawdown = cum_pnl - high_water
    max_dd = drawdown.min()

    # Find drawdown duration (in number of trades)
    if max_dd < 0:
        dd_end_idx = drawdown.idxmin()
        # Find when drawdown started (last time at high water before dd_end)
        before_dd = cum_pnl.loc[:dd_end_idx]
        dd_start_idx = before_dd[before_dd == before_dd.max()].index[-1]
        dd_duration = len(cum_pnl.loc[dd_start_idx:dd_end_idx])
    else:
        dd_duration = 0

    return max_dd, dd_duration


def bootstrap_sharpe(trades_df: pd.DataFrame, n_bootstrap: int = N_BOOTSTRAP) -> dict:
    """Bootstrap confidence intervals for Sharpe ratio."""
    if len(trades_df) < 10:
        return {"point": 0, "mean": 0, "std": 0, "ci_low": 0, "ci_high": 0}

    daily_pnl = trades_df.groupby('earnings_date')['pnl'].sum()
    n_days = len(daily_pnl)

    if n_days < 5:
        return {"point": 0, "mean": 0, "std": 0, "ci_low": 0, "ci_high": 0}

    date_range = (trades_df['earnings_date'].max() - trades_df['earnings_date'].min()).days
    trades_per_year = n_days * 252 / date_range if date_range > 0 else 100

    # Point estimate
    point_sharpe = compute_sharpe(daily_pnl, trades_per_year)

    # Bootstrap
    bootstrap_sharpes = []
    daily_pnl_values = daily_pnl.values

    for _ in range(n_bootstrap):
        # Resample daily P&L with replacement
        sample = np.random.choice(daily_pnl_values, size=n_days, replace=True)
        if np.std(sample) > 0:
            sharpe = (np.mean(sample) / np.std(sample)) * np.sqrt(trades_per_year)
            bootstrap_sharpes.append(sharpe)

    if len(bootstrap_sharpes) == 0:
        return {"point": point_sharpe, "mean": 0, "std": 0, "ci_low": 0, "ci_high": 0}

    bootstrap_sharpes = np.array(bootstrap_sharpes)
    ci_low = np.percentile(bootstrap_sharpes, (1 - BOOTSTRAP_CI) / 2 * 100)
    ci_high = np.percentile(bootstrap_sharpes, (1 + BOOTSTRAP_CI) / 2 * 100)

    return {
        "point": round(point_sharpe, 3),
        "mean": round(np.mean(bootstrap_sharpes), 3),
        "std": round(np.std(bootstrap_sharpes), 3),
        "ci_low": round(ci_low, 3),
        "ci_high": round(ci_high, 3),
    }


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

    # P&L calculation
    commission_pct = (1.30 * 2 * 2) / 5000  # IBKR commissions
    df['payoff'] = df['target_move']
    df['total_cost'] = df['implied_move'] + spread_cost_pct + commission_pct
    df['pnl'] = df['payoff'] - df['total_cost']

    return df


def compute_strategy_metrics(trades_df: pd.DataFrame, do_bootstrap: bool = False) -> dict:
    """Compute comprehensive metrics for a set of trades."""
    if len(trades_df) == 0:
        return None

    trades = trades_df.copy()
    trades_sorted = trades.sort_values('earnings_date')
    trades_sorted['cum_pnl'] = trades_sorted['pnl'].cumsum()

    # Basic stats
    n_trades = len(trades)
    mean_pnl = trades['pnl'].mean()
    std_pnl = trades['pnl'].std()
    median_pnl = trades['pnl'].median()

    # Win/loss analysis
    wins = trades[trades['pnl'] > 0]
    losses = trades[trades['pnl'] <= 0]
    win_rate = len(wins) / n_trades if n_trades > 0 else 0

    avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
    avg_loss = losses['pnl'].mean() if len(losses) > 0 else 0

    # Profit factor
    gross_profit = wins['pnl'].sum() if len(wins) > 0 else 0
    gross_loss = abs(losses['pnl'].sum()) if len(losses) > 0 else 1e-9
    profit_factor = gross_profit / gross_loss

    # Payoff ratio (avg win / avg loss)
    payoff_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    # Consecutive wins/losses
    pnl_signs = (trades_sorted['pnl'] > 0).astype(int)
    streaks = pnl_signs.groupby((pnl_signs != pnl_signs.shift()).cumsum())
    max_consecutive_wins = max([len(g) for _, g in streaks if g.iloc[0] == 1], default=0)
    max_consecutive_losses = max([len(g) for _, g in streaks if g.iloc[0] == 0], default=0)

    # Drawdown
    max_dd, dd_duration = compute_max_drawdown(trades_sorted['cum_pnl'])

    # Sharpe
    daily_pnl = trades.groupby('earnings_date')['pnl'].sum()
    n_days = len(daily_pnl)
    date_range = (trades['earnings_date'].max() - trades['earnings_date'].min()).days
    trades_per_year = n_days * 252 / date_range if date_range > 0 else 100
    sharpe = compute_sharpe(daily_pnl, trades_per_year)

    # Calmar ratio (annualized return / max drawdown)
    total_return = trades_sorted['cum_pnl'].iloc[-1]
    years = date_range / 365 if date_range > 0 else 1
    annual_return = total_return / years
    calmar = abs(annual_return / max_dd) if max_dd < 0 else 0

    # Sortino ratio (return / downside deviation)
    negative_returns = daily_pnl[daily_pnl < 0]
    downside_std = negative_returns.std() if len(negative_returns) > 1 else daily_pnl.std()
    sortino = (daily_pnl.mean() / downside_std) * np.sqrt(trades_per_year) if downside_std > 0 else 0

    metrics = {
        "trades": n_trades,
        "mean_pnl": round(mean_pnl, 5),
        "std_pnl": round(std_pnl, 5),
        "median_pnl": round(median_pnl, 5),
        "win_rate": round(win_rate, 4),
        "avg_win": round(avg_win, 5),
        "avg_loss": round(avg_loss, 5),
        "profit_factor": round(profit_factor, 3),
        "payoff_ratio": round(payoff_ratio, 3),
        "max_consecutive_wins": max_consecutive_wins,
        "max_consecutive_losses": max_consecutive_losses,
        "total_pnl": round(total_return, 4),
        "max_drawdown": round(max_dd, 4),
        "max_dd_duration": dd_duration,
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "calmar": round(calmar, 3),
        "trading_days": n_days,
        "trades_per_day": round(n_trades / n_days, 2) if n_days > 0 else 0,
    }

    # Bootstrap confidence intervals (optional, slower)
    if do_bootstrap and n_trades >= 20:
        bootstrap = bootstrap_sharpe(trades)
        metrics["sharpe_bootstrap"] = bootstrap

    return metrics


def compute_timing_breakdown(trades_df: pd.DataFrame, timing_col: str = 'timing') -> dict:
    """Compute metrics breakdown by BMO/AMC timing."""
    if timing_col not in trades_df.columns:
        return {}

    breakdown = {}
    for timing in ['BMO', 'AMC', 'unknown']:
        subset = trades_df[trades_df[timing_col] == timing]
        if len(subset) >= 10:
            breakdown[timing] = {
                "trades": len(subset),
                "mean_pnl": round(subset['pnl'].mean(), 5),
                "win_rate": round((subset['pnl'] > 0).mean(), 4),
            }

    return breakdown


def compute_monthly_breakdown(trades_df: pd.DataFrame) -> dict:
    """Compute metrics by month."""
    trades = trades_df.copy()
    trades['month'] = trades['earnings_date'].dt.to_period('M')

    monthly = trades.groupby('month').agg({
        'pnl': ['count', 'sum', 'mean'],
    })
    monthly.columns = ['trades', 'total_pnl', 'mean_pnl']
    monthly['win_rate'] = trades.groupby('month')['pnl'].apply(lambda x: (x > 0).mean())

    # Return last 12 months
    recent = monthly.tail(12)
    return {
        str(idx): {
            "trades": int(row['trades']),
            "total_pnl": round(row['total_pnl'], 4),
            "mean_pnl": round(row['mean_pnl'], 5),
            "win_rate": round(row['win_rate'], 4),
        }
        for idx, row in recent.iterrows()
    }


def compute_edge_variance_map(trades_df: pd.DataFrame, pnl_col: str = 'pnl_pct',
                              edge_buckets: list = None) -> dict:
    """Compute historical P&L variance by edge bucket (for Kelly sizing)."""
    if edge_buckets is None:
        edge_buckets = [0.06, 0.08, 0.10, 0.15, 1.0]  # Upper bounds

    trades = trades_df.copy()
    variance_map = {}

    for i, upper in enumerate(edge_buckets):
        lower = edge_buckets[i - 1] if i > 0 else 0
        mask = (trades['edge'] > lower) & (trades['edge'] <= upper)
        bucket_trades = trades[mask]

        if len(bucket_trades) >= 5:
            variance_map[(lower, upper)] = {
                'std': bucket_trades[pnl_col].std(),
                'mean': bucket_trades[pnl_col].mean(),
                'count': len(bucket_trades),
            }

    return variance_map


def compute_kelly_multiplier(edge: float, variance_map: dict, kelly_fraction: float = 0.5) -> float:
    """Compute Kelly position multiplier based on edge and historical variance.

    Kelly criterion: f* = edge / variance
    We use fractional Kelly (kelly_fraction < 1.0) for safety.
    """
    # Fixed sizing: always return 1.0
    if kelly_fraction == 0:
        return 1.0

    # Find the right variance bucket
    bucket_std = None
    for (lower, upper), stats in variance_map.items():
        if lower < edge <= upper:
            bucket_std = stats['std']
            break

    if bucket_std is None or bucket_std == 0:
        return 1.0  # Default to base position

    # Kelly formula: f* = edge / variance
    # Since edge and std are in same units (%), we can use edge / std^2
    # But simpler: use edge / std as a scaling factor
    kelly_raw = edge / bucket_std if bucket_std > 0 else 0

    # Apply Kelly fraction and cap
    kelly_mult = kelly_raw * kelly_fraction
    return max(0.5, min(kelly_mult, 3.0))  # Cap between 0.5x and 3x


def simulate_kelly_strategy(df: pd.DataFrame,
                           edge_threshold: float = 0.06,
                           implied_move_multiplier: float = 1.3,
                           spread_cost_pct: float = 0.03,
                           max_trades_per_day: int = 5,
                           base_risk_pct: float = 0.02,
                           kelly_fraction: float = 0.5,
                           initial_bankroll: float = 100000) -> dict:
    """Simulate Kelly-sized position strategy and return metrics."""
    df = df.copy()

    # Setup same as fixed strategy
    df['implied_move'] = df['hist_move_mean'] * implied_move_multiplier
    df['edge'] = df['pred_q75'] - df['implied_move']
    df['tradeable'] = df['edge'] > edge_threshold

    df = df.sort_values(['earnings_date', 'edge'], ascending=[True, False])
    df['trade_rank'] = df.groupby('earnings_date').cumcount() + 1
    df['take_trade'] = df['tradeable'] & (df['trade_rank'] <= max_trades_per_day)

    # P&L calculation (percentage terms)
    commission_pct = (1.30 * 2 * 2) / 5000
    df['payoff'] = df['target_move']
    df['total_cost'] = df['implied_move'] + spread_cost_pct + commission_pct
    df['pnl_pct'] = df['payoff'] - df['total_cost']

    trades = df[df['take_trade']].copy()
    if len(trades) < 20:
        return None

    # Compute edge-specific variance from training portion (first 70%)
    n_train = int(len(trades) * 0.7)
    train_trades = trades.iloc[:n_train]
    variance_map = compute_edge_variance_map(train_trades)

    if not variance_map:
        return None

    # Simulate with variable position sizing
    trades_sorted = trades.sort_values('earnings_date')
    bankroll = initial_bankroll
    daily_returns = []
    position_sizes = []

    for date, day_trades in trades_sorted.groupby('earnings_date'):
        day_return = 0
        for _, trade in day_trades.iterrows():
            # Compute Kelly multiplier for this trade's edge
            kelly_mult = compute_kelly_multiplier(trade['edge'], variance_map, kelly_fraction)
            position_size = base_risk_pct * kelly_mult

            # Dollar P&L
            dollar_pnl = bankroll * position_size * trade['pnl_pct']
            day_return += dollar_pnl
            position_sizes.append(position_size)

        bankroll += day_return
        daily_returns.append(day_return / (bankroll - day_return))  # Return as %

    daily_returns = pd.Series(daily_returns)
    cum_returns = (1 + daily_returns).cumprod() - 1

    # Compute metrics
    total_return = cum_returns.iloc[-1]
    n_days = len(daily_returns)
    date_range = (trades['earnings_date'].max() - trades['earnings_date'].min()).days
    years = date_range / 365 if date_range > 0 else 1
    trades_per_year = n_days * 252 / date_range if date_range > 0 else 100

    # Max drawdown
    cum_wealth = (1 + daily_returns).cumprod()
    high_water = cum_wealth.cummax()
    drawdown = (cum_wealth - high_water) / high_water
    max_dd = drawdown.min()

    # Sharpe and Sortino
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(trades_per_year) if daily_returns.std() > 0 else 0
    negative = daily_returns[daily_returns < 0]
    downside_std = negative.std() if len(negative) > 1 else daily_returns.std()
    sortino = (daily_returns.mean() / downside_std) * np.sqrt(trades_per_year) if downside_std > 0 else 0

    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    calmar = abs(cagr / max_dd) if max_dd < 0 else 0

    return {
        "trades": len(trades),
        "total_return": round(total_return, 4),
        "cagr": round(cagr, 4),
        "max_drawdown": round(max_dd, 4),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "calmar": round(calmar, 3),
        "avg_position_size": round(np.mean(position_sizes), 4),
        "max_position_size": round(np.max(position_sizes), 4),
        "kelly_fraction": kelly_fraction,
        "base_risk_pct": base_risk_pct,
    }


def compute_yearly_breakdown(trades_df: pd.DataFrame) -> dict:
    """Compute metrics by year."""
    trades = trades_df.copy()
    trades['year'] = trades['earnings_date'].dt.year

    yearly = {}
    for year, grp in trades.groupby('year'):
        if len(grp) >= 10:
            grp_sorted = grp.sort_values('earnings_date')
            grp_sorted['cum_pnl'] = grp_sorted['pnl'].cumsum()
            max_dd, _ = compute_max_drawdown(grp_sorted['cum_pnl'])

            yearly[str(year)] = {
                "trades": len(grp),
                "total_pnl": round(grp['pnl'].sum(), 4),
                "mean_pnl": round(grp['pnl'].mean(), 5),
                "win_rate": round((grp['pnl'] > 0).mean(), 4),
                "max_drawdown": round(max_dd, 4),
            }

    return yearly


def extract_metrics(oos_path: Path = None, do_bootstrap: bool = False) -> dict:
    """Extract comprehensive metrics from OOS predictions."""
    if oos_path is None:
        oos_path = DATA_DIR / "oos_predictions.parquet"

    if not oos_path.exists():
        raise FileNotFoundError(f"OOS predictions not found: {oos_path}")

    oos = pd.read_parquet(oos_path)
    oos['earnings_date'] = pd.to_datetime(oos['earnings_date'])

    # Try to load timing info from ml_features
    ml_features_path = DATA_DIR / "ml_features.parquet"
    if ml_features_path.exists():
        ml_features = pd.read_parquet(ml_features_path)
        ml_features['earnings_date'] = pd.to_datetime(ml_features['earnings_date'])
        if 'timing' in ml_features.columns:
            timing_map = ml_features.drop_duplicates(['symbol', 'earnings_date'])[['symbol', 'earnings_date', 'timing']]
            oos = oos.merge(timing_map, on=['symbol', 'earnings_date'], how='left')

    metrics = {
        "timestamp": datetime.now().isoformat(),
        "git": get_git_info(),
        "model_hashes": get_model_hashes(),
        "data": {
            "oos_samples": len(oos),
            "date_range_start": str(oos['earnings_date'].min().date()),
            "date_range_end": str(oos['earnings_date'].max().date()),
            "unique_symbols": oos['symbol'].nunique(),
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
            metrics["calibration"][f"q{int(q*100)}"] = {
                "expected": expected,
                "actual": round(actual, 4),
                "error": round(actual - expected, 4),
            }

    # Strategy metrics at different thresholds
    thresholds = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12]

    for multiplier, key in [(1.0, "strategy_1.0x"), (1.3, "strategy_1.3x")]:
        threshold_metrics = {}

        for threshold in thresholds:
            sim = simulate_strategy(
                oos,
                edge_threshold=threshold,
                implied_move_multiplier=multiplier,
            )
            trades = sim[sim['take_trade']].copy()

            if len(trades) >= 10:
                thresh_key = f"{int(threshold*100)}pct"
                strat_metrics = compute_strategy_metrics(trades, do_bootstrap=do_bootstrap)

                # Add timing breakdown for 1.3x only (to keep JSON smaller)
                if multiplier == 1.3 and 'timing' in trades.columns:
                    strat_metrics['by_timing'] = compute_timing_breakdown(trades)

                threshold_metrics[thresh_key] = strat_metrics

        metrics[key] = threshold_metrics

    # Add yearly breakdown for recommended threshold (6% at 1.3x)
    sim_6pct = simulate_strategy(oos, edge_threshold=0.06, implied_move_multiplier=1.3)
    trades_6pct = sim_6pct[sim_6pct['take_trade']].copy()
    if len(trades_6pct) > 0:
        metrics["yearly_breakdown_6pct_1.3x"] = compute_yearly_breakdown(trades_6pct)
        metrics["monthly_breakdown_6pct_1.3x"] = compute_monthly_breakdown(trades_6pct)

    # Kelly position sizing metrics - compare across thresholds
    kelly_thresholds = [0.05, 0.06, 0.07, 0.08]
    kelly_fractions = [
        ("fixed", 0.0),
        ("half_kelly", 0.5),
        ("full_kelly", 1.0),
    ]

    metrics["kelly_sizing"] = {}
    for threshold in kelly_thresholds:
        thresh_key = f"{int(threshold*100)}pct"
        metrics["kelly_sizing"][thresh_key] = {}

        for name, kelly_frac in kelly_fractions:
            result = simulate_kelly_strategy(
                oos,
                edge_threshold=threshold,
                implied_move_multiplier=1.3,
                base_risk_pct=0.02,
                kelly_fraction=kelly_frac,
            )
            if result:
                metrics["kelly_sizing"][thresh_key][name] = result

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
    print("\n" + "=" * 70)
    print("ML PIPELINE METRICS (Comprehensive)")
    print("=" * 70)

    print(f"\nTimestamp: {metrics['timestamp']}")
    git = metrics['git']
    print(f"Git: {git['commit']} ({git['branch']})" + (" [dirty]" if git['dirty'] else ""))

    if metrics.get('model_hashes'):
        hashes = metrics['model_hashes']
        print(f"Models: q50={hashes.get('q50', '?')}, q75={hashes.get('q75', '?')}, "
              f"q90={hashes.get('q90', '?')}, q95={hashes.get('q95', '?')}")

    data = metrics['data']
    print(f"\nData:")
    print(f"  OOS samples: {data['oos_samples']:,}")
    print(f"  Unique symbols: {data.get('unique_symbols', 'N/A'):,}")
    print(f"  Date range: {data['date_range_start']} to {data['date_range_end']}")

    print(f"\nCalibration:")
    print(f"  {'Quantile':<10} {'Expected':<10} {'Actual':<10} {'Error':<10}")
    print(f"  {'-'*40}")
    for q in [50, 75, 90, 95]:
        cal = metrics['calibration'].get(f'q{q}', {})
        if cal:
            print(f"  q{q:<9} {cal['expected']:<10.0%} {cal['actual']:<10.1%} {cal['error']:+.2%}")

    print(f"\nStrategy Performance (1.3x realistic):")
    print(f"  {'Thresh':<8} {'Trades':<8} {'Mean':<9} {'WinRate':<9} {'Sharpe':<8} "
          f"{'Sortino':<8} {'MaxDD':<9} {'PF':<6}")
    print(f"  {'-'*73}")

    for thresh in ['3pct', '4pct', '5pct', '6pct', '7pct', '8pct', '10pct', '12pct']:
        data = metrics['strategy_1.3x'].get(thresh, {})
        if data:
            print(f"  {thresh:<8} {data['trades']:<8} {data['mean_pnl']:+.2%}   "
                  f"{data['win_rate']:.1%}     {data['sharpe']:<8.2f} "
                  f"{data['sortino']:<8.2f} {data['max_drawdown']:+.1%}    "
                  f"{data['profit_factor']:.2f}")

            # Print bootstrap CI if available
            if 'sharpe_bootstrap' in data:
                bs = data['sharpe_bootstrap']
                print(f"           └── Sharpe 95% CI: [{bs['ci_low']:.2f}, {bs['ci_high']:.2f}] "
                      f"(bootstrap mean: {bs['mean']:.2f})")

    # Timing breakdown
    strat_6pct = metrics['strategy_1.3x'].get('6pct', {})
    if strat_6pct and 'by_timing' in strat_6pct:
        print(f"\n  By Timing (6% threshold):")
        for timing, data in strat_6pct['by_timing'].items():
            print(f"    {timing}: {data['trades']} trades, {data['mean_pnl']:+.2%} mean, "
                  f"{data['win_rate']:.1%} win rate")

    # Yearly breakdown
    if 'yearly_breakdown_6pct_1.3x' in metrics:
        print(f"\n  Yearly Performance (6% threshold, 1.3x):")
        for year, data in metrics['yearly_breakdown_6pct_1.3x'].items():
            print(f"    {year}: {data['trades']} trades, {data['total_pnl']:+.1%} total, "
                  f"{data['win_rate']:.1%} win, {data['max_drawdown']:+.1%} DD")

    # Kelly position sizing comparison across thresholds
    if 'kelly_sizing' in metrics and metrics['kelly_sizing']:
        print(f"\nKelly Position Sizing Comparison (1.3x realistic):")
        print(f"  {'Threshold':<10} {'Strategy':<14} {'Trades':<8} {'Return':<10} {'MaxDD':<9} {'Sharpe':<8}")
        print(f"  {'-'*65}")

        for thresh_key in ['5pct', '6pct', '7pct', '8pct']:
            thresh_data = metrics['kelly_sizing'].get(thresh_key, {})
            for strat_name in ['fixed', 'half_kelly', 'full_kelly']:
                data = thresh_data.get(strat_name)
                if data:
                    print(f"  {thresh_key:<10} {strat_name:<14} {data['trades']:<8} "
                          f"{data['total_return']:+7.1%}   {data['max_drawdown']:+7.1%}  "
                          f"{data['sharpe']:.2f}")
            if thresh_data:
                print()  # blank line between thresholds

    print("\n" + "=" * 70)


def print_history():
    """Print metrics history comparison."""
    history = load_history()
    if not history:
        print("No metrics history found.")
        return

    print("\n" + "=" * 80)
    print("METRICS HISTORY (1.3x realistic, 6% threshold)")
    print("=" * 80)
    print(f"{'Date':<12} {'Commit':<8} {'OOS':<8} {'Trades':<8} {'Sharpe':<8} "
          f"{'Sortino':<8} {'MaxDD':<9} {'q75err':<8}")
    print("-" * 80)

    for entry in history[-15:]:  # Last 15 entries
        date = entry['timestamp'][:10]
        commit = entry['git']['commit'][:7]
        oos = entry['data']['oos_samples']

        strat = entry.get('strategy_1.3x', {}).get('6pct', {})
        trades = strat.get('trades', '-')
        sharpe = strat.get('sharpe', '-')
        sortino = strat.get('sortino', '-')
        max_dd = strat.get('max_drawdown', '-')

        q75_err = entry.get('calibration', {}).get('q75', {}).get('error', '-')

        sharpe_str = f"{sharpe:.2f}" if isinstance(sharpe, (int, float)) else str(sharpe)
        sortino_str = f"{sortino:.2f}" if isinstance(sortino, (int, float)) else str(sortino)
        max_dd_str = f"{max_dd:+.1%}" if isinstance(max_dd, (int, float)) else str(max_dd)
        q75_str = f"{q75_err:+.2%}" if isinstance(q75_err, (int, float)) else str(q75_err)

        print(f"{date:<12} {commit:<8} {oos:<8} {trades:<8} {sharpe_str:<8} "
              f"{sortino_str:<8} {max_dd_str:<9} {q75_str:<8}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Extract comprehensive ML pipeline metrics")
    parser.add_argument("--run-dir", type=Path, help="Specific run directory")
    parser.add_argument("--history", action="store_true", help="Show metrics history")
    parser.add_argument("--no-save", action="store_true", help="Don't save to history")
    parser.add_argument("--bootstrap", action="store_true",
                       help="Include bootstrap confidence intervals (slower)")

    args = parser.parse_args()

    if args.history:
        print_history()
        return

    oos_path = None
    if args.run_dir:
        oos_path = args.run_dir / "oos_predictions.parquet"
        if not oos_path.exists():
            oos_path = DATA_DIR / "oos_predictions.parquet"

    print(f"Extracting metrics from: {oos_path or DATA_DIR / 'oos_predictions.parquet'}")
    if args.bootstrap:
        print(f"Running bootstrap analysis ({N_BOOTSTRAP} samples)...")

    metrics = extract_metrics(oos_path, do_bootstrap=args.bootstrap)
    print_metrics(metrics)

    if not args.no_save:
        save_metrics(metrics)


if __name__ == "__main__":
    main()
