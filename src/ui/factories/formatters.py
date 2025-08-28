"""
Formatting Utilities Module
===========================
Centralized formatting functions for numbers, percentages, and changes.
"""

from typing import Optional, Union


def fmt_int(n: Union[int, float, str]) -> str:
    """Format integer with thousands separator."""
    try:
        return f"{int(n):,}"
    except (ValueError, TypeError):
        return "0"


def fmt_float(n: Union[int, float, str], decimals: int = 1) -> str:
    """Format float with specified decimal places."""
    try:
        return f"{float(n):,.{decimals}f}"
    except (ValueError, TypeError):
        return f"0.{'0' * decimals}"


def fmt_pct(n: Union[int, float, str], decimals: int = 1) -> str:
    """Format number as percentage with specified decimal places."""
    try:
        return f"{float(n):,.{decimals}f}%"
    except (ValueError, TypeError):
        return f"0.{'0' * decimals}%"


def fmt_change(
    value: Union[int, float],
    previous: Optional[Union[int, float]] = None,
    as_pct: bool = False,
    with_sign: bool = True,
    decimals: int = 1,
) -> str:
    """
    Format a change value with appropriate sign and formatting.

    Parameters:
        value: Current value
        previous: Previous value to calculate change
        as_pct: Whether to format as percentage
        with_sign: Whether to include + sign for positive values
        decimals: Number of decimal places

    Returns:
        Formatted change string
    """
    if previous is not None:
        change = value - previous
        if previous != 0:
            pct_change = (change / abs(previous)) * 100
        else:
            pct_change = 0 if value == 0 else 100
    else:
        change = value
        pct_change = value

    if as_pct:
        formatted = fmt_pct(pct_change, decimals)
        if with_sign and pct_change > 0:
            return f"+{formatted}"
        return formatted
    else:
        formatted = fmt_float(change, decimals)
        if with_sign and change > 0:
            return f"+{formatted}"
        return formatted


def fmt_currency(
    n: Union[int, float, str], decimals: int = 0, symbol: str = "$"
) -> str:
    """Format number as currency."""
    try:
        value = float(n)
        if decimals == 0:
            return f"{symbol}{int(value):,}"
        else:
            return f"{symbol}{value:,.{decimals}f}"
    except (ValueError, TypeError):
        return f"{symbol}0"


def fmt_number(
    n: Union[int, float, str], decimals: Optional[int] = None
) -> str:
    """
    Smart number formatting that automatically determines decimal places.

    Parameters:
        n: Number to format
        decimals: Optional decimal places (auto-determines if None)

    Returns:
        Formatted number string
    """
    try:
        value = float(n)
        if decimals is None:
            # Auto-determine decimals based on value
            if value == int(value):
                return fmt_int(value)
            elif abs(value) >= 100:
                return fmt_float(value, 1)
            elif abs(value) >= 10:
                return fmt_float(value, 2)
            else:
                return fmt_float(value, 3)
        else:
            return fmt_float(value, decimals)
    except (ValueError, TypeError):
        return "0"


def fmt_duration(days: Union[int, float]) -> str:
    """
    Format duration in days to human-readable string.

    Parameters:
        days: Number of days

    Returns:
        Human-readable duration string
    """
    try:
        days = int(days)
        if days < 7:
            return f"{days} day{'s' if days != 1 else ''}"
        elif days < 30:
            weeks = days // 7
            return f"{weeks} week{'s' if weeks != 1 else ''}"
        elif days < 365:
            months = days // 30
            return f"{months} month{'s' if months != 1 else ''}"
        else:
            years = days // 365
            return f"{years} year{'s' if years != 1 else ''}"
    except (ValueError, TypeError):
        return "0 days"


def fmt_ratio(
    numerator: Union[int, float],
    denominator: Union[int, float],
    decimals: int = 1,
) -> str:
    """
    Format a ratio as percentage.

    Parameters:
        numerator: The numerator
        denominator: The denominator
        decimals: Decimal places for percentage

    Returns:
        Formatted percentage string
    """
    try:
        if denominator == 0:
            return "N/A"
        ratio = (numerator / denominator) * 100
        return fmt_pct(ratio, decimals)
    except (ValueError, TypeError):
        return "N/A"


# Export all formatting functions
__all__ = [
    "fmt_int",
    "fmt_float",
    "fmt_pct",
    "fmt_change",
    "fmt_currency",
    "fmt_number",
    "fmt_duration",
    "fmt_ratio",
]
