"""NYSE market calendar using FMP API.

Fetches holidays and early close days from FMP, caches locally.
Falls back to weekends-only logic if FMP unavailable and no cache.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

FMP_API_KEY = os.getenv('FMP_API_KEY', '')
FMP_HOLIDAYS_URL = 'https://financialmodelingprep.com/stable/holidays-by-exchange'
CACHE_FILE = Path(__file__).parent.parent.parent / 'data' / 'nyse_calendar_cache.json'
CACHE_TTL_DAYS = 7  # Refresh cache weekly


class NYSECalendar:
    """NYSE trading calendar with holiday and early close awareness."""

    def __init__(self):
        self._holidays: set[date] = set()
        self._early_close: dict[date, str] = {}  # date -> close time (e.g., "13:00")
        self._loaded = False
        self._load_failed = False

    def _load_cache(self) -> bool:
        """Load calendar from local cache if fresh."""
        if not CACHE_FILE.exists():
            return False

        try:
            data = json.loads(CACHE_FILE.read_text())
            cached_at = datetime.fromisoformat(data['cached_at'])
            if datetime.now() - cached_at > timedelta(days=CACHE_TTL_DAYS):
                logger.info("Calendar cache expired, will refresh from FMP")
                return False  # Cache expired

            self._holidays = {date.fromisoformat(d) for d in data.get('holidays', [])}
            self._early_close = {
                date.fromisoformat(d): t for d, t in data.get('early_close', {}).items()
            }
            self._loaded = True
            logger.debug(
                f"Loaded NYSE calendar from cache: {len(self._holidays)} holidays, "
                f"{len(self._early_close)} early close days"
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to load calendar cache: {e}")
            return False

    def _save_cache(self):
        """Save calendar to local cache."""
        try:
            CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            data = {
                'cached_at': datetime.now().isoformat(),
                'holidays': [d.isoformat() for d in sorted(self._holidays)],
                'early_close': {d.isoformat(): t for d, t in self._early_close.items()},
            }
            CACHE_FILE.write_text(json.dumps(data, indent=2))
            logger.debug(f"Saved NYSE calendar cache to {CACHE_FILE}")
        except Exception as e:
            logger.warning(f"Failed to save calendar cache: {e}")

    def _fetch_from_fmp(self) -> bool:
        """Fetch calendar from FMP API."""
        if not FMP_API_KEY:
            logger.error("FMP_API_KEY not set, cannot fetch NYSE calendar")
            return False

        try:
            resp = requests.get(
                FMP_HOLIDAYS_URL,
                params={'exchange': 'NYSE', 'apikey': FMP_API_KEY},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()

            if not isinstance(data, list):
                logger.error(f"Unexpected FMP response type: {type(data)}")
                return False

            holidays = set()
            early_close = {}

            for item in data:
                try:
                    d = date.fromisoformat(item['date'])
                    if item.get('isClosed'):
                        holidays.add(d)
                    elif item.get('adjCloseTime'):
                        early_close[d] = item['adjCloseTime']
                except (KeyError, ValueError) as e:
                    logger.warning(f"Skipping malformed calendar entry: {item}, error: {e}")
                    continue

            self._holidays = holidays
            self._early_close = early_close
            self._loaded = True
            self._save_cache()
            logger.info(
                f"Loaded NYSE calendar from FMP: {len(self._holidays)} holidays, "
                f"{len(self._early_close)} early close days"
            )
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch NYSE calendar from FMP: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error fetching NYSE calendar: {e}")
            return False

    def _ensure_loaded(self):
        """Ensure calendar data is loaded."""
        if self._loaded:
            return

        # Don't retry if we already failed this session
        if self._load_failed:
            return

        # Try cache first, then FMP
        if not self._load_cache():
            if not self._fetch_from_fmp():
                # Mark as failed so we don't retry repeatedly
                self._load_failed = True
                logger.error(
                    "NYSE calendar unavailable (FMP failed, no cache). "
                    "Falling back to weekends-only logic - holidays will NOT be detected!"
                )

    def is_holiday(self, d: date) -> bool:
        """Check if date is a market holiday."""
        self._ensure_loaded()
        return d in self._holidays

    def is_trading_day(self, d: date) -> bool:
        """Check if date is a trading day (not weekend, not holiday)."""
        if d.weekday() >= 5:  # Saturday or Sunday
            return False
        return not self.is_holiday(d)

    def is_early_close(self, d: date) -> bool:
        """Check if date is an early close day."""
        self._ensure_loaded()
        return d in self._early_close

    def get_close_time(self, d: date) -> str:
        """Get market close time for a date (e.g., '16:00' or '13:00')."""
        self._ensure_loaded()
        return self._early_close.get(d, '16:00')

    def next_trading_day(self, d: date) -> date:
        """Get next trading day (skips weekends and holidays)."""
        next_d = d + timedelta(days=1)
        # Safety limit to avoid infinite loop if calendar is broken
        for _ in range(30):
            if self.is_trading_day(next_d):
                return next_d
            next_d += timedelta(days=1)
        # Fallback: return next weekday if we somehow didn't find a trading day
        logger.warning(f"Could not find trading day within 30 days of {d}, returning next weekday")
        next_d = d + timedelta(days=1)
        while next_d.weekday() >= 5:
            next_d += timedelta(days=1)
        return next_d

    def refresh(self) -> bool:
        """Force refresh calendar from FMP API."""
        self._loaded = False
        self._load_failed = False
        return self._fetch_from_fmp()


# Global singleton instance
_calendar: Optional[NYSECalendar] = None


def get_calendar() -> NYSECalendar:
    """Get the NYSE calendar singleton."""
    global _calendar
    if _calendar is None:
        _calendar = NYSECalendar()
    return _calendar


# Convenience functions
def is_trading_day(d: date) -> bool:
    """Check if date is a trading day (not weekend, not holiday)."""
    return get_calendar().is_trading_day(d)


def is_holiday(d: date) -> bool:
    """Check if date is a market holiday."""
    return get_calendar().is_holiday(d)


def is_early_close(d: date) -> bool:
    """Check if date is an early close day (market closes at 1:00 PM ET)."""
    return get_calendar().is_early_close(d)


def next_trading_day(d: date) -> date:
    """Get next trading day (skips weekends and holidays)."""
    return get_calendar().next_trading_day(d)


def get_close_time(d: date) -> str:
    """Get market close time for a date (e.g., '16:00' or '13:00')."""
    return get_calendar().get_close_time(d)
