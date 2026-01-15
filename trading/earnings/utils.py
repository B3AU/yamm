"""Shared utility functions for earnings trading."""

from datetime import date, datetime, timedelta
import pytz

ET = pytz.timezone('US/Eastern')


def today_et() -> date:
    """Get today's date in Eastern Time."""
    return datetime.now(ET).date()


def should_exit_today(earnings_date: date, timing: str, today: date = None) -> bool:
    """Check if a position should exit today based on earnings date and timing.
    
    Args:
        earnings_date: The earnings announcement date
        timing: 'BMO' (before market open) or 'AMC' (after market close)
        today: Override for today's date (for testing). Defaults to today_et().
    
    Returns:
        True if position should exit today, False otherwise.
    """
    if today is None:
        today = today_et()
    
    if timing == 'BMO' and earnings_date == today:
        return True
    elif timing == 'AMC' and earnings_date == today - timedelta(days=1):
        return True
    # Handle weekend: Friday AMC exits Monday
    elif timing == 'AMC' and today.weekday() == 0:  # Monday
        if earnings_date == today - timedelta(days=3):  # Friday
            return True
    
    return False
