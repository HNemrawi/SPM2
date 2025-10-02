"""
Common session management patterns.

This module provides shared session operations that were previously duplicated
across different analysis modules, making session handling more consistent.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from . import SessionKeys


class SessionPatterns:
    """
    Common session management patterns used across modules.

    This class consolidates session handling logic that was duplicated across
    different analysis modules.
    """

    @staticmethod
    def get_or_set_default(
        session_key: str,
        default_value: Any,
        session_state: Optional[Dict] = None,
    ) -> Any:
        """
        Get a value from session state or set it to default if not present.

        Args:
            session_key: Key in session state
            default_value: Default value to set if key doesn't exist
            session_state: Optional custom session state dict (defaults to st.session_state)

        Returns:
            Value from session state or the default value
        """
        state = (
            session_state if session_state is not None else st.session_state
        )

        if session_key not in state:
            state[session_key] = default_value

        return state[session_key]

    @staticmethod
    def batch_update_session(updates: Dict[str, Any]) -> None:
        """
        Update multiple session state keys at once to avoid multiple reruns.

        Args:
            updates: Dictionary of key-value pairs to update
        """
        st.session_state.update(updates)

    @staticmethod
    def clear_session_keys(keys: List[str]) -> None:
        """
        Clear multiple session state keys.

        Args:
            keys: List of keys to remove from session state
        """
        for key in keys:
            if key in st.session_state:
                del st.session_state[key]

    @staticmethod
    def get_module_session_keys(module_prefix: str) -> List[str]:
        """
        Get all session keys that start with a module prefix.

        Args:
            module_prefix: Prefix to filter keys by

        Returns:
            List of matching session keys
        """
        return [
            key
            for key in st.session_state.keys()
            if key.startswith(module_prefix)
        ]

    @staticmethod
    def clear_module_session(module_prefix: str) -> None:
        """
        Clear all session state keys for a specific module.

        Args:
            module_prefix: Module prefix to clear (e.g., "spm2_", "inbound_")
        """
        keys_to_clear = SessionPatterns.get_module_session_keys(module_prefix)
        SessionPatterns.clear_session_keys(keys_to_clear)

    @staticmethod
    def save_date_range(
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        key_prefix: str = "date_range",
    ) -> None:
        """
        Save a date range to session state with standard naming.

        Args:
            start_date: Start date
            end_date: End date
            key_prefix: Prefix for the session keys
        """
        st.session_state[f"{key_prefix}_start"] = start_date
        st.session_state[f"{key_prefix}_end"] = end_date
        st.session_state[f"{key_prefix}_updated"] = datetime.now()

    @staticmethod
    def get_date_range(
        key_prefix: str = "date_range",
        default_start: Optional[datetime] = None,
        default_end: Optional[datetime] = None,
    ) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        """
        Get a date range from session state.

        Args:
            key_prefix: Prefix for the session keys
            default_start: Default start date if not in session
            default_end: Default end date if not in session

        Returns:
            Tuple of (start_date, end_date)
        """
        start_key = f"{key_prefix}_start"
        end_key = f"{key_prefix}_end"

        start = st.session_state.get(start_key, default_start)
        end = st.session_state.get(end_key, default_end)

        if start:
            start = pd.to_datetime(start)
        if end:
            end = pd.to_datetime(end)

        return start, end

    @staticmethod
    def save_filter_state(
        filter_name: str, selected_values: List[str], module_prefix: str = ""
    ) -> None:
        """
        Save filter state to session with consistent naming.

        Args:
            filter_name: Name of the filter
            selected_values: List of selected values
            module_prefix: Optional module prefix for namespacing
        """
        key = (
            f"{module_prefix}{filter_name}_filter"
            if module_prefix
            else f"{filter_name}_filter"
        )
        st.session_state[key] = selected_values
        st.session_state[f"{key}_updated"] = datetime.now()

    @staticmethod
    def get_filter_state(
        filter_name: str,
        default: Optional[List[str]] = None,
        module_prefix: str = "",
    ) -> List[str]:
        """
        Get filter state from session.

        Args:
            filter_name: Name of the filter
            default: Default values if not in session
            module_prefix: Optional module prefix for namespacing

        Returns:
            List of selected values
        """
        key = (
            f"{module_prefix}{filter_name}_filter"
            if module_prefix
            else f"{filter_name}_filter"
        )
        return st.session_state.get(key, default or [])

    @staticmethod
    def mark_data_dirty(reason: str = "data_changed") -> None:
        """
        Mark data as needing recalculation.

        Args:
            reason: Reason for marking dirty (for debugging)
        """
        st.session_state["data_dirty"] = True
        st.session_state["data_dirty_reason"] = reason
        st.session_state["data_dirty_timestamp"] = datetime.now()

    @staticmethod
    def is_data_dirty() -> bool:
        """
        Check if data is marked as dirty.

        Returns:
            True if data is dirty, False otherwise
        """
        return st.session_state.get("data_dirty", False)

    @staticmethod
    def clear_data_dirty() -> None:
        """Clear the data dirty flag."""
        st.session_state["data_dirty"] = False
        if "data_dirty_reason" in st.session_state:
            del st.session_state["data_dirty_reason"]
        if "data_dirty_timestamp" in st.session_state:
            del st.session_state["data_dirty_timestamp"]

    @staticmethod
    def has_required_data(required_keys: List[str]) -> bool:
        """
        Check if all required data keys are present in session.

        Args:
            required_keys: List of required session keys

        Returns:
            True if all required keys are present, False otherwise
        """
        return all(key in st.session_state for key in required_keys)

    @staticmethod
    def get_session_summary() -> Dict[str, Any]:
        """
        Get a summary of the current session state.

        Returns:
            Dictionary with session state summary
        """
        total_keys = len(st.session_state.keys())

        # Categorize keys
        data_keys = [
            k
            for k in st.session_state.keys()
            if "df" in k.lower() or "data" in k.lower()
        ]
        filter_keys = [
            k for k in st.session_state.keys() if "filter" in k.lower()
        ]
        module_keys = [
            k
            for k in st.session_state.keys()
            if any(
                mod in k.lower()
                for mod in ["spm2", "inbound", "outbound", "dashboard"]
            )
        ]

        return {
            "total_keys": total_keys,
            "data_keys": len(data_keys),
            "filter_keys": len(filter_keys),
            "module_keys": len(module_keys),
            "has_main_data": SessionKeys.DF in st.session_state,
            "data_loaded": st.session_state.get(
                SessionKeys.DATA_LOADED, False
            ),
            "current_module": st.session_state.get(
                SessionKeys.SELECTED_MODULE, "None"
            ),
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    @staticmethod
    def safe_get_dataframe(
        key: str = SessionKeys.DF,
    ) -> Optional[pd.DataFrame]:
        """
        Safely get a DataFrame from session state.

        Args:
            key: Session key for the DataFrame

        Returns:
            DataFrame if valid, None otherwise
        """
        df = st.session_state.get(key)

        if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
            return df

        return None

    @staticmethod
    def validate_and_clean_session() -> Dict[str, Any]:
        """
        Validate and clean the session state.

        Returns:
            Dictionary with validation results and cleaned keys
        """
        results = {
            "cleaned_keys": [],
            "invalid_dataframes": [],
            "orphaned_keys": [],
            "total_cleaned": 0,
        }

        keys_to_remove = []

        for key, value in st.session_state.items():
            # Check for invalid DataFrames
            if isinstance(value, pd.DataFrame):
                if value.empty:
                    results["invalid_dataframes"].append(key)
                    keys_to_remove.append(key)

            # Check for very old timestamps (older than 24 hours)
            if key.endswith("_timestamp") or key.endswith("_updated"):
                if isinstance(value, datetime):
                    age = (
                        datetime.now() - value
                    ).total_seconds() / 3600  # hours
                    if age > 24:  # Older than 24 hours
                        results["orphaned_keys"].append(key)
                        keys_to_remove.append(key)

        # Remove identified keys
        for key in keys_to_remove:
            del st.session_state[key]
            results["cleaned_keys"].append(key)

        results["total_cleaned"] = len(keys_to_remove)

        return results


class ModuleSessionPattern:
    """
    Base pattern for module-specific session management.

    This provides a template for modules to manage their session state consistently.
    """

    def __init__(self, module_name: str):
        self.module_name = module_name.lower()
        self.key_prefix = f"{self.module_name}_"

    def get_key(self, suffix: str) -> str:
        """Get a prefixed key for this module."""
        return f"{self.key_prefix}{suffix}"

    def set_value(self, key: str, value: Any) -> None:
        """Set a value in session state with module prefix."""
        full_key = self.get_key(key)
        st.session_state[full_key] = value

    def get_value(self, key: str, default: Any = None) -> Any:
        """Get a value from session state with module prefix."""
        full_key = self.get_key(key)
        return st.session_state.get(full_key, default)

    def clear_module_state(self) -> None:
        """Clear all session state for this module."""
        SessionPatterns.clear_module_session(self.key_prefix)

    def get_module_keys(self) -> List[str]:
        """Get all session keys for this module."""
        return SessionPatterns.get_module_session_keys(self.key_prefix)

    def save_config(self, config: Dict[str, Any]) -> None:
        """Save module configuration to session state."""
        for key, value in config.items():
            self.set_value(f"config_{key}", value)

    def get_config(
        self, default_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get module configuration from session state."""
        config = {}

        for key in st.session_state.keys():
            if key.startswith(self.get_key("config_")):
                config_key = key.replace(self.get_key("config_"), "")
                config[config_key] = st.session_state[key]

        if not config and default_config:
            self.save_config(default_config)
            return default_config

        return config


# Create convenience instances
session_patterns = SessionPatterns()
