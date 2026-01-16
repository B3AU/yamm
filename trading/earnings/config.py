"""Shared configuration for earnings trading system.

Schedule times are configurable via environment variables.
All times are in ET (Eastern Time), HH:MM format.
"""
import os
from typing import Tuple, List


# === SCHEDULE CONFIGURATION (all times in ET, HH:MM format) ===

# Core schedule times
MORNING_CONNECT_TIME_ET = os.getenv('MORNING_CONNECT_TIME_ET', '09:25')
EXIT_TIME_ET = os.getenv('EXIT_TIME_ET', '14:00')
SCREEN_TIME_ET = os.getenv('SCREEN_TIME_ET', '14:15')
PRICE_IMPROVE_START_ET = os.getenv('PRICE_IMPROVE_START_ET', '14:25')
PRICE_IMPROVE_INTERVAL = int(os.getenv('PRICE_IMPROVE_INTERVAL', '10'))  # minutes between rounds
FINAL_CHECK_TIME_ET = os.getenv('FINAL_CHECK_TIME_ET', '15:55')
FORCE_EXIT_TIME_ET = os.getenv('FORCE_EXIT_TIME_ET', '15:58')
DISCONNECT_TIME_ET = os.getenv('DISCONNECT_TIME_ET', '16:05')
BACKFILL_TIME_ET = os.getenv('BACKFILL_TIME_ET', '16:30')

# Hour ranges for recurring jobs
SNAPSHOT_START_HOUR = int(os.getenv('SNAPSHOT_START_HOUR', '9'))
SNAPSHOT_END_HOUR = int(os.getenv('SNAPSHOT_END_HOUR', '16'))
MONITOR_FILLS_START_HOUR = int(os.getenv('MONITOR_FILLS_START_HOUR', '14'))
MONITOR_FILLS_END_HOUR = int(os.getenv('MONITOR_FILLS_END_HOUR', '16'))


def parse_time_et(time_str: str) -> Tuple[int, int]:
    """Parse HH:MM string to (hour, minute) tuple.
    
    Args:
        time_str: Time in HH:MM format (e.g., "14:00", "09:25")
        
    Returns:
        Tuple of (hour, minute)
        
    Raises:
        ValueError: If time format is invalid
    """
    parts = time_str.split(':')
    if len(parts) != 2:
        raise ValueError(f"Invalid time format: {time_str}. Expected HH:MM")
    try:
        hour, minute = int(parts[0]), int(parts[1])
    except ValueError:
        raise ValueError(f"Invalid time format: {time_str}. Hour and minute must be integers")
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"Invalid time: {time_str}. Hour must be 0-23, minute 0-59")
    return hour, minute


def get_exit_time() -> Tuple[int, int]:
    """Get exit time as (hour, minute) tuple."""
    return parse_time_et(EXIT_TIME_ET)


def get_screen_time() -> Tuple[int, int]:
    """Get screen time as (hour, minute) tuple."""
    return parse_time_et(SCREEN_TIME_ET)


def get_price_improve_times() -> List[Tuple[int, int, float]]:
    """Get price improvement times as list of (hour, minute, aggression) tuples.
    
    Returns 4 rounds with increasing aggression (0.4, 0.5, 0.6, 0.7),
    starting at PRICE_IMPROVE_START_ET with PRICE_IMPROVE_INTERVAL minutes between each.
    """
    start_hour, start_min = parse_time_et(PRICE_IMPROVE_START_ET)
    interval = PRICE_IMPROVE_INTERVAL
    aggressions = [0.4, 0.5, 0.6, 0.7]
    
    result = []
    for i, agg in enumerate(aggressions):
        total_min = start_hour * 60 + start_min + (i * interval)
        hour = total_min // 60
        minute = total_min % 60
        result.append((hour, minute, agg))
    return result
