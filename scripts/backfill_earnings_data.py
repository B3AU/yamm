#!/usr/bin/env python3
"""Backfill missing earnings and price data.

Fetches:
1. Prices for symbols in earnings calendar but missing prices
2. Earnings calendar for symbols with prices but no calendar entry
3. Recomputes historical moves for all symbols with both

Usage:
    python scripts/backfill_earnings_data.py
"""
import pandas as pd
import requests
from datetime import datetime, timedelta
import time
from pathlib import Path
import os
from dotenv import load_dotenv
import sys

# Load environment
load_dotenv(Path(__file__).parent.parent / '.env')

DATA_DIR = Path('data/earnings')
FMP_API_KEY = os.getenv('FMP_API_KEY', '')

NASDAQ_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'application/json',
    'Origin': 'https://www.nasdaq.com',
    'Referer': 'https://www.nasdaq.com/',
}


def fetch_fmp_prices(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch historical prices from FMP."""
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}"
    params = {
        'from': start_date,
        'to': end_date,
        'apikey': FMP_API_KEY,
    }

    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return pd.DataFrame()

        data = r.json()
        if 'historical' not in data:
            return pd.DataFrame()

        df = pd.DataFrame(data['historical'])
        if df.empty:
            return df

        df['symbol'] = symbol
        df['date'] = pd.to_datetime(df['date'])
        return df[['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        print(f"  Warning: Failed to fetch prices for {symbol}: {e}", file=sys.stderr)
        return pd.DataFrame()


def fetch_nasdaq_earnings(from_date: datetime, to_date: datetime, existing_symbols: set = None) -> pd.DataFrame:
    """Fetch earnings calendar from Nasdaq API."""
    all_rows = []
    current_date = from_date

    while current_date <= to_date:
        date_str = current_date.strftime('%Y-%m-%d')
        url = f"https://api.nasdaq.com/api/calendar/earnings?date={date_str}"

        try:
            r = requests.get(url, headers=NASDAQ_HEADERS, timeout=10)
            if r.status_code == 200:
                data = r.json()
                rows = data.get('data', {}).get('rows', [])
                for row in rows:
                    symbol = row.get('symbol', '')
                    # Only add if not already in existing calendar
                    if existing_symbols is None or symbol not in existing_symbols:
                        row['date'] = date_str
                        all_rows.append(row)
        except Exception as e:
            print(f"  Warning: Failed to fetch earnings for {date_str}: {e}", file=sys.stderr)

        current_date += timedelta(days=1)
        time.sleep(0.1)

        if (current_date - from_date).days % 30 == 0:
            print(f"  Fetched up to {current_date.strftime('%Y-%m-%d')}...")

    return pd.DataFrame(all_rows)


def compute_earnings_moves(symbol: str, earnings_dates: list, prices_df: pd.DataFrame) -> list:
    """Compute moves around each earnings date."""
    moves = []

    prices_df = prices_df.set_index('date').sort_index()

    for earn_date in earnings_dates:
        earn_date = pd.to_datetime(earn_date)

        try:
            # Find T-1 (day before earnings)
            t_minus_1_candidates = prices_df[prices_df.index < earn_date].tail(1)
            if t_minus_1_candidates.empty:
                continue
            t_minus_1 = t_minus_1_candidates.index[0]

            # Find T (earnings day or next trading day)
            t_candidates = prices_df[prices_df.index >= earn_date].head(1)
            if t_candidates.empty:
                continue
            t = t_candidates.index[0]

            # Find T+1 (day after earnings reaction)
            t_plus_1_candidates = prices_df[prices_df.index > t].head(1)
            if t_plus_1_candidates.empty:
                continue
            t_plus_1 = t_plus_1_candidates.index[0]

            # Get prices
            close_t_minus_1 = prices_df.loc[t_minus_1, 'close']
            open_t = prices_df.loc[t, 'open']
            close_t = prices_df.loc[t, 'close']
            close_t_plus_1 = prices_df.loc[t_plus_1, 'close']

            # Compute moves
            gap_move = (open_t - close_t_minus_1) / close_t_minus_1
            full_move = (close_t - close_t_minus_1) / close_t_minus_1
            overnight_move = (close_t_plus_1 - close_t_minus_1) / close_t_minus_1

            moves.append({
                'symbol': symbol,
                'earnings_date': earn_date,
                'close_t_minus_1': close_t_minus_1,
                'open_t': open_t,
                'close_t': close_t,
                'close_t_plus_1': close_t_plus_1,
                'gap_move': gap_move,
                'gap_move_abs': abs(gap_move),
                'full_move': full_move,
                'full_move_abs': abs(full_move),
                'overnight_move': overnight_move,
                'overnight_move_abs': abs(overnight_move),
            })
        except Exception as e:
            print(f"  Warning: Failed to compute moves for {symbol} on {earn_date}: {e}", file=sys.stderr)
            continue

    return moves


def main():
    print("=" * 60)
    print("BACKFILL EARNINGS DATA")
    print("=" * 60)

    if not FMP_API_KEY:
        print("ERROR: FMP_API_KEY not set")
        sys.exit(1)

    # Load current data
    print("\nLoading existing data...")
    cal = pd.read_parquet(DATA_DIR / 'earnings_calendar.parquet')
    prices = pd.read_parquet('data/prices.pqt')
    prices['date'] = pd.to_datetime(prices['date'])
    moves = pd.read_parquet(DATA_DIR / 'historical_earnings_moves.parquet')

    cal_symbols = set(cal['symbol'].unique())
    price_symbols = set(prices['symbol'].unique())
    move_symbols = set(moves['symbol'].unique())

    print(f"  Calendar: {len(cal_symbols)} symbols")
    print(f"  Prices: {len(price_symbols)} symbols")
    print(f"  Moves: {len(move_symbols)} symbols")

    # === STEP 1: Fetch prices for symbols in calendar but missing ===
    no_prices = cal_symbols - price_symbols
    # Filter to US-like symbols (no dots, dashes, reasonable length)
    us_no_prices = [s for s in no_prices if isinstance(s, str) and '.' not in s and '-' not in s and 1 <= len(s) <= 5]

    print(f"\n--- STEP 1: Fetch prices for {len(us_no_prices)} symbols ---")

    new_prices = []
    start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')

    for i, symbol in enumerate(us_no_prices[:500]):  # Limit to 500 for now
        if i > 0 and i % 50 == 0:
            print(f"  Progress: {i}/{min(500, len(us_no_prices))} ({len(new_prices)} with data)")

        df = fetch_fmp_prices(symbol, start_date, end_date)
        if not df.empty:
            new_prices.append(df)
        time.sleep(0.12)  # Rate limit

    if new_prices:
        new_prices_df = pd.concat(new_prices, ignore_index=True)
        print(f"  Fetched prices for {new_prices_df['symbol'].nunique()} new symbols")

        # Append to existing prices
        prices = pd.concat([prices, new_prices_df], ignore_index=True)
        prices.to_parquet('data/prices.pqt', index=False)
        print(f"  Updated prices.pqt: {prices['symbol'].nunique()} total symbols")
        price_symbols = set(prices['symbol'].unique())

    # === STEP 2: Fetch earnings for symbols with prices but no calendar ===
    no_calendar = price_symbols - cal_symbols
    print(f"\n--- STEP 2: Need earnings for {len(no_calendar)} symbols ---")
    print("  (Nasdaq API fetches all symbols per day, so this is automatic)")

    # Check if we need to extend the calendar date range
    cal_max_date = pd.to_datetime(cal['date']).max()
    if cal_max_date < datetime.now() - timedelta(days=7):
        print(f"  Calendar ends at {cal_max_date}, fetching recent data...")
        new_cal = fetch_nasdaq_earnings(
            cal_max_date + timedelta(days=1),
            datetime.now(),
            existing_symbols=None  # Get all
        )
        if not new_cal.empty:
            new_cal['date'] = pd.to_datetime(new_cal['date'])
            cal = pd.concat([cal, new_cal], ignore_index=True)
            cal.to_parquet(DATA_DIR / 'earnings_calendar.parquet', index=False)
            print(f"  Updated calendar: {cal['symbol'].nunique()} symbols, {len(cal)} events")
            cal_symbols = set(cal['symbol'].unique())

    # === STEP 3: Recompute all moves ===
    print(f"\n--- STEP 3: Recompute historical moves ---")

    # Build price cache
    common_symbols = cal_symbols & price_symbols
    print(f"  Symbols with both calendar & prices: {len(common_symbols)}")

    price_cache = {}
    for symbol in common_symbols:
        sym_prices = prices[prices['symbol'] == symbol].copy()
        if not sym_prices.empty:
            price_cache[symbol] = sym_prices.sort_values('date')

    # Compute moves
    all_moves = []
    cal['date'] = pd.to_datetime(cal['date'])

    for i, symbol in enumerate(price_cache):
        if i > 0 and i % 200 == 0:
            print(f"  Progress: {i}/{len(price_cache)} ({len(all_moves)} moves)")

        symbol_earnings = cal[cal['symbol'] == symbol]['date'].tolist()
        symbol_moves = compute_earnings_moves(symbol, symbol_earnings, price_cache[symbol])
        all_moves.extend(symbol_moves)

    moves_df = pd.DataFrame(all_moves)
    moves_df.to_parquet(DATA_DIR / 'historical_earnings_moves.parquet', index=False)

    print(f"\n" + "=" * 60)
    print("BACKFILL COMPLETE")
    print("=" * 60)
    print(f"  Historical moves: {len(moves_df)} events, {moves_df['symbol'].nunique()} symbols")
    print(f"  (Was: {len(moves)} events, {move_symbols} symbols)")

    # Check specific symbols
    for sym in ['AIR', 'ACI']:
        sym_moves = moves_df[moves_df['symbol'] == sym]
        print(f"  {sym}: {len(sym_moves)} moves")


if __name__ == '__main__':
    main()
