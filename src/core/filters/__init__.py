"""
Centralized filter management system.

This module provides a unified approach to filter management across all analysis modules,
replacing the scattered filter logic that was previously duplicated.
"""

from .constants import COMMON_FILTER_CONFIGS, FILTER_CATEGORIES
from .manager import FilterConfig, FilterManager

__all__ = [
    "FilterManager",
    "FilterConfig",
    "FILTER_CATEGORIES",
    "COMMON_FILTER_CONFIGS",
]
