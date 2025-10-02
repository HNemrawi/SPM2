"""
Centralized filter management system.

This module provides a unified approach to filter management that was previously
duplicated across different analysis modules.
"""

import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from src.core.session import SessionKeys
from src.core.utils.helpers import create_multiselect_filter

from .constants import (
    COMMON_FILTER_CONFIGS,
    FILTER_CATEGORIES,
    REQUIRED_COLUMNS,
)


@dataclass
class FilterConfig:
    """Configuration for a single filter."""

    key: str
    column: str
    label: str
    help: Optional[str] = None
    default: Optional[List[str]] = None
    required: bool = False


class FilterManager:
    """
    Centralized filter management system.

    This class provides a unified interface for creating, managing, and applying
    filters across different analysis modules.
    """

    def __init__(self, module_name: str, state_manager: Any = None):
        """
        Initialize the filter manager.

        Args:
            module_name: Name of the module using this filter manager
            state_manager: Optional state manager for persistence
        """
        self.module_name = module_name
        self.state_manager = state_manager
        self._filters = {}
        self._applied_filters = {}

    def hash_data(self, data: Any) -> str:
        """
        Return a short MD5 hash of data for widget keys / cache keys.

        Args:
            data: Any data to hash

        Returns:
            8-character hash string
        """
        return hashlib.md5(str(data).encode()).hexdigest()[:8]

    def init_section_state(
        self, key: str, defaults: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Initialize or retrieve a per-section state dictionary.

        Args:
            key: Unique key for the section state
            defaults: Default values to initialize state with

        Returns:
            Section state dictionary
        """
        state_key = f"state_{key}"
        if state_key not in st.session_state:
            st.session_state[state_key] = defaults or {}

        # Get the section state
        section_state = st.session_state[state_key]

        # Special handling for filter form state - restore PH destinations
        if (
            key == "filter_form"
            and "selected_ph_destinations" in section_state
        ):
            if "selected_ph_destinations" not in st.session_state:
                st.session_state[
                    SessionKeys.SELECTED_PH_DESTINATIONS
                ] = section_state["selected_ph_destinations"]

        return section_state

    def get_filter_timestamp(self) -> str:
        """
        Generate a timestamp for filter change tracking.

        Returns:
            Formatted timestamp string
        """
        return datetime.now().strftime("%H:%M:%S.%f")[:-3]

    def render_filter_section(
        self,
        config_key: str,
        df: pd.DataFrame,
        expanded: bool = False,
        custom_filters: Optional[List[FilterConfig]] = None,
    ) -> Dict[str, List[str]]:
        """
        Render a section of filters based on configuration.

        Args:
            config_key: Key from COMMON_FILTER_CONFIGS
            df: DataFrame to filter
            expanded: Whether section is expanded by default
            custom_filters: Optional custom filter configurations

        Returns:
            Dictionary mapping filter keys to selected values
        """
        if config_key in COMMON_FILTER_CONFIGS:
            config = COMMON_FILTER_CONFIGS[config_key]
            title = config["title"]
            icon = config["icon"]
            filters = [FilterConfig(**f) for f in config["filters"]]
        elif custom_filters:
            title = config_key.replace("_", " ").title()
            icon = "🔍"
            filters = custom_filters
        else:
            raise ValueError(f"Unknown filter config: {config_key}")

        results = {}

        with st.sidebar.expander(f"{icon} **{title}**", expanded=expanded):
            for filter_config in filters:
                if filter_config.column in df.columns:
                    # Get saved state
                    saved_values = self._get_saved_filter_state(
                        filter_config.key
                    )

                    # Create the filter
                    selected_values = create_multiselect_filter(
                        df=df,
                        column_name=filter_config.column,
                        label=filter_config.label,
                        key=f"{self.module_name.lower()}_{filter_config.key}",
                        help=filter_config.help,
                        default=saved_values or filter_config.default or [],
                    )

                    results[filter_config.key] = selected_values

                    # Save state
                    self._save_filter_state(filter_config.key, selected_values)

        return results

    def render_category_filters(
        self,
        category: str,
        df: pd.DataFrame,
        expanded: bool = False,
        title_override: Optional[str] = None,
        icon_override: Optional[str] = None,
    ) -> Dict[str, List[str]]:
        """
        Render filters for a category from FILTER_CATEGORIES.

        Args:
            category: Category key from FILTER_CATEGORIES
            df: DataFrame to filter
            expanded: Whether section is expanded by default
            title_override: Optional title override
            icon_override: Optional icon override

        Returns:
            Dictionary mapping filter keys to selected values
        """
        if category not in FILTER_CATEGORIES:
            raise ValueError(f"Unknown filter category: {category}")

        category_config = FILTER_CATEGORIES[category]
        title = title_override or category
        icon = icon_override or "🔍"

        results = {}

        with st.sidebar.expander(f"{icon} **{title}**", expanded=expanded):
            for display_name, column_name in category_config.items():
                if column_name in df.columns:
                    filter_key = column_name.lower().replace(" ", "_")

                    # Get saved state
                    saved_values = self._get_saved_filter_state(filter_key)

                    # Create the filter
                    selected_values = create_multiselect_filter(
                        df=df,
                        column_name=column_name,
                        label=display_name,
                        key=f"{self.module_name.lower()}_{filter_key}",
                        default=saved_values or [],
                    )

                    results[filter_key] = selected_values

                    # Save state
                    self._save_filter_state(filter_key, selected_values)

        return results

    def apply_filters(
        self, df: pd.DataFrame, filter_dict: Dict[str, List[str]]
    ) -> pd.DataFrame:
        """
        Apply filters to a dataframe.

        Args:
            df: DataFrame to filter
            filter_dict: Dictionary mapping column names to selected values

        Returns:
            Filtered dataframe
        """
        filtered_df = df.copy()

        for filter_key, selected_values in filter_dict.items():
            if selected_values:  # Only apply filter if values are selected
                # Try to find the column name
                column_name = self._resolve_column_name(filter_key, df)

                if column_name and column_name in filtered_df.columns:
                    filtered_df = filtered_df[
                        filtered_df[column_name].isin(selected_values)
                    ]

        return filtered_df

    def get_filter_summary(self, filter_dict: Dict[str, List[str]]) -> str:
        """
        Generate a summary of applied filters.

        Args:
            filter_dict: Dictionary of applied filters

        Returns:
            Summary string
        """
        active_filters = {k: v for k, v in filter_dict.items() if v}

        if not active_filters:
            return "No filters applied"

        summary_parts = []
        for filter_key, values in active_filters.items():
            if len(values) == 1:
                summary_parts.append(f"{filter_key}: {values[0]}")
            else:
                summary_parts.append(f"{filter_key}: {len(values)} items")

        return " | ".join(summary_parts)

    def clear_filters(self) -> None:
        """Clear all saved filter states."""
        if self.state_manager:
            # Clear from state manager if available
            if hasattr(self.state_manager, "clear_filters"):
                self.state_manager.clear_filters()

        # Clear from session state
        keys_to_clear = [
            key
            for key in st.session_state.keys()
            if key.startswith(f"{self.module_name.lower()}_")
            and "filter" in key
        ]

        for key in keys_to_clear:
            del st.session_state[key]

        self._filters.clear()
        self._applied_filters.clear()

    def validate_dataframe(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate that a dataframe has required columns for filtering.

        Args:
            df: DataFrame to validate

        Returns:
            Tuple of (is_valid, missing_columns)
        """
        missing_columns = []

        for column in REQUIRED_COLUMNS:
            if column not in df.columns:
                missing_columns.append(column)

        return len(missing_columns) == 0, missing_columns

    def _get_saved_filter_state(self, filter_key: str) -> Optional[List[str]]:
        """Get saved filter state from state manager or session state."""
        if self.state_manager and hasattr(
            self.state_manager, "get_widget_state"
        ):
            return self.state_manager.get_widget_state(
                f"{filter_key}_filter", []
            )

        # Fallback to session state
        session_key = f"{self.module_name.lower()}_{filter_key}"
        return st.session_state.get(session_key, [])

    def _save_filter_state(self, filter_key: str, values: List[str]) -> None:
        """Save filter state to state manager or session state."""
        if self.state_manager and hasattr(
            self.state_manager, "set_widget_state"
        ):
            self.state_manager.set_widget_state(f"{filter_key}_filter", values)

        self._filters[filter_key] = values

    def _resolve_column_name(
        self, filter_key: str, df: pd.DataFrame
    ) -> Optional[str]:
        """
        Resolve filter key to actual column name.

        Args:
            filter_key: Filter key to resolve
            df: DataFrame with columns

        Returns:
            Column name if found, None otherwise
        """
        # First, check if filter_key is already a column name
        if filter_key in df.columns:
            return filter_key

        # Check in FILTER_CATEGORIES
        for category_config in FILTER_CATEGORIES.values():
            for display_name, column_name in category_config.items():
                if filter_key.lower() == column_name.lower().replace(" ", "_"):
                    return column_name

        # Try common transformations
        potential_names = [
            filter_key,
            filter_key.title(),
            filter_key.replace("_", ""),
            "".join(word.title() for word in filter_key.split("_")),
        ]

        for name in potential_names:
            if name in df.columns:
                return name

        return None
