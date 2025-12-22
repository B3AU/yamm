from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

import pandas as pd
import requests


FMP_BASE_URL = "https://financialmodelingprep.com/stable"


def get_fmp_key() -> str:
    key = os.environ.get("FMP_API_KEY")
    if not key:
        raise RuntimeError("FMP_API_KEY environment variable not set")
    return key


# -----------------------------------------------------------------------------
# Universe loading
# -----------------------------------------------------------------------------

def load_nasdaq_universe() -> pd.DataFrame:
    """Load US common stocks from NASDAQ and NYSE/AMEX, filtering out ETFs and special issues."""
    nasdaq = pd.read_csv(
        "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        sep="|",
    )
    other = pd.read_csv(
        "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
        sep="|",
    )

    nasdaq = nasdaq[(nasdaq["Test Issue"] == "N") & (nasdaq["ETF"] == "N")].copy()
    nasdaq["symbol"] = nasdaq["Symbol"]

    other = other[(other["Test Issue"] == "N") & (other["ETF"] == "N")].copy()
    other["symbol"] = other["ACT Symbol"]

    df = pd.concat(
        [nasdaq[["symbol", "Security Name"]], other[["symbol", "Security Name"]]],
        ignore_index=True,
    )

    df["symbol"] = df["symbol"].astype(str)
    df = df[
        ~df["symbol"].str.contains(r"[.\-]", regex=True)
        & ~df["symbol"].str.endswith(("W", "R", "P"))
    ]

    return df.drop_duplicates().reset_index(drop=True)


# -----------------------------------------------------------------------------
# HTTP helpers
# -----------------------------------------------------------------------------

@dataclass
class RateLimiter:
    max_per_minute: int = 240
    _last_call: float = 0.0

    @property
    def min_interval(self) -> float:
        return 60.0 / self.max_per_minute

    def wait(self) -> None:
        now = time.time()
        elapsed = now - self._last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_call = time.time()


def _request_json(
    session: requests.Session,
    path: str,
    params: dict,
    max_retries: int = 8,
) -> list | dict:
    url = f"{FMP_BASE_URL}{path}"
    for attempt in range(max_retries):
        resp = session.get(url, params=params, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code in (429, 500, 502, 503, 504):
            sleep_time = min(60, (2**attempt) + random.random())
            time.sleep(sleep_time)
            continue
        resp.raise_for_status()
    raise RuntimeError(f"Failed after {max_retries} retries: {url}")


# -----------------------------------------------------------------------------
# Date windowing
# -----------------------------------------------------------------------------

def _date_str(d: datetime) -> str:
    return d.strftime("%Y-%m-%d")


def iter_date_windows(
    years: int = 5,
    window_days: int = 30,
    end_date: datetime | None = None,
) -> Iterator[tuple[str, str]]:
    """Yield (start, end) date string pairs, iterating backwards from end_date."""
    end_dt = end_date or datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=int(365.25 * years))
    cur_end = end_dt

    while cur_end > start_dt:
        cur_start = max(start_dt, cur_end - timedelta(days=window_days))
        yield _date_str(cur_start), _date_str(cur_end)
        cur_end = cur_start


# -----------------------------------------------------------------------------
# Progress tracking
# -----------------------------------------------------------------------------

@dataclass
class DownloadProgress:
    done: dict[str, bool]
    cursor: dict[str, int]

    @classmethod
    def load(cls, path: Path) -> DownloadProgress:
        if path.exists():
            data = json.loads(path.read_text())
            return cls(done=data.get("done", {}), cursor=data.get("cursor", {}))
        return cls(done={}, cursor={})

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"done": self.done, "cursor": self.cursor}, indent=2))

    def is_done(self, job: str) -> bool:
        return self.done.get(job, False)

    def mark_done(self, job: str) -> None:
        self.done[job] = True
        self.cursor.pop(job, None)

    def get_page(self, job: str) -> int:
        return self.cursor.get(job, 0)

    def set_page(self, job: str, page: int) -> None:
        self.cursor[job] = page


# -----------------------------------------------------------------------------
# News download
# -----------------------------------------------------------------------------

def _infer_end_date_from_progress(out_dir: str) -> datetime | None:
    """Infer the original end_date from existing progress.json.

    Uses the most frequent end date to handle runs that crossed midnight.
    """
    progress_path = Path(out_dir) / "progress.json"
    if not progress_path.exists():
        return None
    try:
        data = json.loads(progress_path.read_text())
        # Count end dates to find the most common (handles midnight crossings)
        from collections import Counter
        end_dates: Counter[str] = Counter()
        for job in list(data.get("done", {})) + list(data.get("cursor", {})):
            parts = job.split("|")
            if len(parts) == 3:
                end_dates[parts[2]] += 1
        if not end_dates:
            return None
        # Use most common end date (the anchor date)
        most_common = end_dates.most_common(1)[0][0]
        return datetime.strptime(most_common, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except Exception:
        return None


@dataclass(frozen=True)
class BackfillConfig:
    out_dir: str = "fmp_news_backfill"
    years: int = 5
    window_days: int = 30
    limit: int = 1000
    max_calls_per_minute: int = 240
    end_date: datetime | None = None  # None = infer from progress or use now()


def download_news_backfill(
    universe_df: pd.DataFrame,
    fmp_key: str | None = None,
    config: BackfillConfig | None = None,
    progress_callback: callable | None = None,
) -> None:
    """
    Download news for each symbol over rolling date windows.

    Writes Parquet part files and progress.json for resume capability.
    Automatically infers end_date from existing progress to ensure resume works.
    """
    cfg = config or BackfillConfig()
    api_key = fmp_key or get_fmp_key()

    out_path = Path(cfg.out_dir)
    parts_path = out_path / "parts"
    parts_path.mkdir(parents=True, exist_ok=True)

    progress_file = out_path / "progress.json"
    progress = DownloadProgress.load(progress_file)

    # Infer end_date from existing progress if not specified
    end_date = cfg.end_date
    if end_date is None:
        end_date = _infer_end_date_from_progress(cfg.out_dir)
    if end_date is not None:
        print(f"Using end_date: {_date_str(end_date)}")

    session = requests.Session()
    rate_limiter = RateLimiter(max_per_minute=cfg.max_calls_per_minute)

    symbols = universe_df["symbol"].dropna().astype(str).unique().tolist()
    windows = list(iter_date_windows(years=cfg.years, window_days=cfg.window_days, end_date=end_date))

    # Pre-compute symbols to process (skip fully done)
    symbols_to_process = [
        sym for sym in symbols
        if not all(progress.is_done(f"{sym}|{d_from}|{d_to}") for d_from, d_to in windows)
    ]
    n_skipped = len(symbols) - len(symbols_to_process)

    for si, sym in enumerate(symbols_to_process):
        for wi, (d_from, d_to) in enumerate(windows):
            job = f"{sym}|{d_from}|{d_to}"
            if progress.is_done(job):
                continue

            page = progress.get_page(job)

            while True:
                rate_limiter.wait()
                params = {
                    "apikey": api_key,
                    "symbols": sym,
                    "from": d_from,
                    "to": d_to,
                    "limit": cfg.limit,
                    "page": page,
                }
                data = _request_json(session, "/news/stock", params)
                df = pd.DataFrame(data)

                if df.empty:
                    progress.mark_done(job)
                    progress.save(progress_file)
                    break

                if "symbol" not in df.columns:
                    df["symbol"] = sym
                df["window_from"] = d_from
                df["window_to"] = d_to
                df["page"] = page

                part_file = parts_path / f"symbol={sym}_from={d_from}_to={d_to}_page={page}.parquet"
                df.to_parquet(part_file, index=False)

                page += 1
                progress.set_page(job, page)
                progress.save(progress_file)

            if progress_callback:
                progress_callback(n_skipped + si + 1, len(symbols), sym, wi + 1, len(windows), n_skipped)


def merge_news_parts(
    out_dir: str = "fmp_news_backfill",
    out_file: str = "news_all.parquet",
) -> pd.DataFrame:
    """Merge all part files into a single parquet file."""
    parts_path = Path(out_dir) / "parts"
    files = sorted(parts_path.glob("*.parquet"))

    if not files:
        return pd.DataFrame()

    all_df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    all_df.to_parquet(Path(out_dir) / out_file, index=False)
    return all_df


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    def print_progress(si: int, total_s: int, sym: str, wi: int, total_w: int) -> None:
        print(f"\r[{si}/{total_s}] {sym} window {wi}/{total_w}", end="", flush=True)

    universe = load_nasdaq_universe()
    print(f"Loaded {len(universe)} symbols")

    if "--merge-only" in sys.argv:
        df = merge_news_parts()
        print(f"Merged {len(df)} news articles")
    else:
        download_news_backfill(universe, progress_callback=print_progress)
        print("\nBackfill complete.")
