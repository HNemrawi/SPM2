"""
Common module patterns and utilities.

This module provides shared functionality for all analysis modules, including:
- Base page classes with common patterns
- Shared date configuration utilities
- Common filter management helpers
- Module initialization utilities
"""

from .base_page import BaseDateConfig, BaseFilterManager, BaseModulePage
from .date_config import InboundDateConfig, OutboundDateConfig, SPM2DateConfig

__all__ = [
    "BaseModulePage",
    "BaseDateConfig",
    "BaseFilterManager",
    "SPM2DateConfig",
    "InboundDateConfig",
    "OutboundDateConfig",
]
