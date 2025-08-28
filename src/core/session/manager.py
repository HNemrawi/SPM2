"""
Session state management and utility functions
"""

from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import streamlit as st


def reset_session():
    """
    Reset the Streamlit session state and clear all caches comprehensively.

    This function:
    - Clears all session state variables
    - Clears data cache (@st.cache_data)
    - Clears resource cache (@st.cache_resource)
    - Clears singleton cache (legacy)
    - Resets all filter states
    - Clears any temporary data
    """
    # Store keys to delete (to avoid modification during iteration)
    keys_to_delete = list(st.session_state.keys())

    # Delete all items in the session state
    for key in keys_to_delete:
        try:
            del st.session_state[key]
        except KeyError:
            # Key might have been already deleted
            pass

    # Clear all caches
    try:
        # Clear the data cache
        st.cache_data.clear()
    except Exception as e:
        print(f"Warning: Could not clear data cache: {e}")

    try:
        # Clear the resource cache
        st.cache_resource.clear()
    except Exception as e:
        print(f"Warning: Could not clear resource cache: {e}")

    # Clear legacy cache if exists
    try:
        if hasattr(st, "legacy_caching"):
            st.legacy_caching.clear_cache()
    except BaseException:
        pass

    # Clear memo cache if exists
    try:
        if hasattr(st, "memo"):
            st.memo.clear()
    except BaseException:
        pass

    # Clear singleton cache if exists
    try:
        if hasattr(st, "singleton"):
            st.singleton.clear()
    except BaseException:
        pass


def set_analysis_result(analysis_type: str, data: pd.DataFrame) -> None:
    """
    Store analysis results in session state with consistent naming.

    Parameters:
        analysis_type (str): Type of analysis ("spm2", "inbound", "outbound", "general")
        data (pd.DataFrame): Analysis results dataframe
    """
    key = f"{analysis_type}_df"
    st.session_state[key] = data


def get_analysis_result(analysis_type: str) -> Optional[pd.DataFrame]:
    """
    Retrieve analysis results from session state.

    Parameters:
        analysis_type (str): Type of analysis ("spm2", "inbound", "outbound", "general")

    Returns:
        Optional[pd.DataFrame]: Analysis results or None if not found
    """
    key = f"{analysis_type}_df"
    return st.session_state.get(key)


def get_session_info() -> dict:
    """
    Get information about current session state and cache status.

    Returns:
        dict: Information about session state including counts and memory usage
    """
    info = {
        "session_keys": len(st.session_state.keys()),
        "session_items": list(st.session_state.keys()),
        "has_data": "df" in st.session_state,
        "data_size": 0,
    }

    # Check data size if available
    if "df" in st.session_state and hasattr(st.session_state["df"], "shape"):
        df = st.session_state["df"]
        info["data_size"] = df.shape[0] if hasattr(df, "shape") else 0
        info["data_columns"] = df.shape[1] if hasattr(df, "shape") else 0

    return info


def check_data_available() -> Optional[pd.DataFrame]:
    """
    Check if data has been uploaded and is available in session state.

    Returns:
        Optional[pd.DataFrame]: Dataframe if available, None otherwise
    """
    df = st.session_state.get("df")
    if df is None or df.empty:
        st.info("📭 Please upload data in the sidebar first.")
        return None
    return df


def ensure_date_range(
    dates, default_start: str, default_end: str
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Ensure a valid date range is selected or use defaults.

    Parameters:
        dates: Date input value from Streamlit
        default_start (str): Default start date in YYYY-MM-DD format
        default_end (str): Default end date in YYYY-MM-DD format

    Returns:
        Tuple[pd.Timestamp, pd.Timestamp]: Start and end dates as pandas Timestamps
    """
    if not dates or len(dates) != 2:
        # Use defaults
        start_date = pd.to_datetime(default_start)
        end_date = pd.to_datetime(default_end)
    else:
        start_date = pd.to_datetime(dates[0])
        end_date = pd.to_datetime(dates[1])

    return start_date, end_date


# ============================================================================
# UNIFIED STATE MANAGER
# ============================================================================


class StateManager:
    """
    Unified state management for consistent filter and session handling
    across all modules (Dashboard, SPM2, Recidivism).

    This addresses UX consistency issues identified in the audit by providing:
    - Consistent filter persistence across module switches
    - Standardized dirty flag management
    - Unified state validation and cleanup
    - Single source of truth for state keys
    """

    # State key prefixes for different modules
    DASHBOARD_PREFIX = "dashboard"
    SPM2_PREFIX = "spm2"
    RECIDIVISM_INBOUND_PREFIX = "inbound"
    RECIDIVISM_OUTBOUND_PREFIX = "outbound"
    GLOBAL_PREFIX = "global"

    # Common state keys
    FILTERS_KEY = "filters"
    DATE_RANGE_KEY = "date_range"
    DIRTY_FLAG_KEY = "dirty"
    LAST_UPDATED_KEY = "last_updated"
    ANALYSIS_REQUESTED_KEY = "analysis_requested"
    WIDGET_STATE_KEY = "widget_state"

    @classmethod
    def _get_module_key(cls, module: str, key: str) -> str:
        """Generate prefixed key for module-specific state."""
        return f"{module}_{key}"

    @classmethod
    def set_filter_state(cls, module: str, filters: Dict[str, Any]) -> None:
        """
        Set filter state for a specific module.

        Parameters:
            module: Module identifier (dashboard, spm2, inbound, outbound)
            filters: Dictionary of filter selections
        """
        filter_key = cls._get_module_key(module, cls.FILTERS_KEY)
        timestamp_key = cls._get_module_key(module, cls.LAST_UPDATED_KEY)
        dirty_key = cls._get_module_key(module, cls.DIRTY_FLAG_KEY)

        st.session_state[filter_key] = filters
        st.session_state[timestamp_key] = datetime.now().isoformat()
        st.session_state[dirty_key] = True

        # Also set global filters for backward compatibility
        st.session_state[cls.FILTERS_KEY] = filters
        st.session_state["last_filter_change"] = datetime.now().isoformat()

    @classmethod
    def get_filter_state(cls, module: str) -> Dict[str, Any]:
        """
        Get filter state for a specific module.

        Parameters:
            module: Module identifier

        Returns:
            Dictionary of filter selections
        """
        filter_key = cls._get_module_key(module, cls.FILTERS_KEY)
        return st.session_state.get(filter_key, {})

    @classmethod
    def set_date_range(
        cls,
        module: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        prev_start: Optional[pd.Timestamp] = None,
        prev_end: Optional[pd.Timestamp] = None,
    ) -> None:
        """
        Set date range for a specific module.

        Parameters:
            module: Module identifier
            start_date: Current period start
            end_date: Current period end
            prev_start: Previous period start (optional)
            prev_end: Previous period end (optional)
        """
        date_key = cls._get_module_key(module, cls.DATE_RANGE_KEY)
        timestamp_key = cls._get_module_key(module, cls.LAST_UPDATED_KEY)
        dirty_key = cls._get_module_key(module, cls.DIRTY_FLAG_KEY)

        date_range = {
            "start": start_date,
            "end": end_date,
            "prev_start": prev_start,
            "prev_end": prev_end,
        }

        st.session_state[date_key] = date_range
        st.session_state[timestamp_key] = datetime.now().isoformat()
        st.session_state[dirty_key] = True

        # Set global date range for backward compatibility
        st.session_state["t0"] = start_date
        st.session_state["t1"] = end_date
        if prev_start:
            st.session_state["prev_start"] = prev_start
        if prev_end:
            st.session_state["prev_end"] = prev_end

    @classmethod
    def get_date_range(cls, module: str) -> Dict[str, Optional[pd.Timestamp]]:
        """
        Get date range for a specific module.

        Parameters:
            module: Module identifier

        Returns:
            Dictionary with date range information
        """
        date_key = cls._get_module_key(module, cls.DATE_RANGE_KEY)
        return st.session_state.get(
            date_key,
            {"start": None, "end": None, "prev_start": None, "prev_end": None},
        )

    @classmethod
    def set_dirty_flag(cls, module: str, dirty: bool = True) -> None:
        """
        Set dirty flag for a module to indicate changes need processing.

        Parameters:
            module: Module identifier
            dirty: Whether module state is dirty
        """
        dirty_key = cls._get_module_key(module, cls.DIRTY_FLAG_KEY)
        st.session_state[dirty_key] = dirty

        if dirty:
            timestamp_key = cls._get_module_key(module, cls.LAST_UPDATED_KEY)
            st.session_state[timestamp_key] = datetime.now().isoformat()

    @classmethod
    def is_dirty(cls, module: str) -> bool:
        """
        Check if module state is dirty (needs processing).

        Parameters:
            module: Module identifier

        Returns:
            True if module needs reprocessing
        """
        dirty_key = cls._get_module_key(module, cls.DIRTY_FLAG_KEY)
        return st.session_state.get(dirty_key, False)

    @classmethod
    def clear_dirty_flag(cls, module: str) -> None:
        """Clear dirty flag after processing."""
        cls.set_dirty_flag(module, False)

    @classmethod
    def request_analysis(cls, module: str) -> None:
        """
        Request analysis for a module (explicit user trigger).

        Parameters:
            module: Module identifier
        """
        analysis_key = cls._get_module_key(module, cls.ANALYSIS_REQUESTED_KEY)
        st.session_state[analysis_key] = True
        cls.clear_dirty_flag(
            module
        )  # Clear dirty since analysis was requested

    @classmethod
    def is_analysis_requested(cls, module: str) -> bool:
        """
        Check if analysis was explicitly requested for a module.

        Parameters:
            module: Module identifier

        Returns:
            True if analysis should run
        """
        analysis_key = cls._get_module_key(module, cls.ANALYSIS_REQUESTED_KEY)
        return st.session_state.get(analysis_key, False)

    @classmethod
    def clear_analysis_request(cls, module: str) -> None:
        """Clear analysis request flag after processing."""
        analysis_key = cls._get_module_key(module, cls.ANALYSIS_REQUESTED_KEY)
        st.session_state[analysis_key] = False

    @classmethod
    def get_module_state_summary(cls, module: str) -> Dict[str, Any]:
        """
        Get complete state summary for a module.

        Parameters:
            module: Module identifier

        Returns:
            Dictionary with all module state information
        """
        return {
            "module": module,
            "filters": cls.get_filter_state(module),
            "date_range": cls.get_date_range(module),
            "is_dirty": cls.is_dirty(module),
            "analysis_requested": cls.is_analysis_requested(module),
            "last_updated": st.session_state.get(
                cls._get_module_key(module, cls.LAST_UPDATED_KEY), "Never"
            ),
        }

    @classmethod
    def reset_module_state(cls, module: str) -> None:
        """
        Reset all state for a specific module.

        Parameters:
            module: Module identifier
        """
        keys_to_clear = [
            cls._get_module_key(module, cls.FILTERS_KEY),
            cls._get_module_key(module, cls.DATE_RANGE_KEY),
            cls._get_module_key(module, cls.DIRTY_FLAG_KEY),
            cls._get_module_key(module, cls.LAST_UPDATED_KEY),
            cls._get_module_key(module, cls.ANALYSIS_REQUESTED_KEY),
            cls._get_module_key(module, cls.WIDGET_STATE_KEY),
        ]

        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]

    @classmethod
    def reset_all_modules(cls) -> None:
        """Reset state for all modules."""
        modules = [
            cls.DASHBOARD_PREFIX,
            cls.SPM2_PREFIX,
            cls.RECIDIVISM_INBOUND_PREFIX,
            cls.RECIDIVISM_OUTBOUND_PREFIX,
        ]

        for module in modules:
            cls.reset_module_state(module)

    @classmethod
    def validate_state(cls, module: str) -> Tuple[bool, Optional[str]]:
        """
        Validate state consistency for a module.

        Parameters:
            module: Module identifier

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check if filters exist and are valid
            filters = cls.get_filter_state(module)
            if filters and not isinstance(filters, dict):
                return False, f"Invalid filter format for {module}"

            # Check if date ranges are valid
            date_range = cls.get_date_range(module)
            if date_range.get("start") and date_range.get("end"):
                if date_range["start"] > date_range["end"]:
                    return (
                        False,
                        f"Invalid date range for {module}: start after end",
                    )

            return True, None

        except Exception as e:
            return False, f"State validation error for {module}: {str(e)}"

    # ============== WIDGET STATE MANAGEMENT ==============

    @classmethod
    def save_widget_state(
        cls, module: str, widget_key: str, value: Any
    ) -> None:
        """
        Save widget state for a specific module.

        Parameters:
            module: Module identifier
            widget_key: Widget identifier
            value: Widget value to save
        """
        widget_state_key = cls._get_module_key(module, cls.WIDGET_STATE_KEY)

        # Initialize widget state dict if not exists
        if widget_state_key not in st.session_state:
            st.session_state[widget_state_key] = {}

        st.session_state[widget_state_key][widget_key] = value

        # Update module timestamp
        timestamp_key = cls._get_module_key(module, cls.LAST_UPDATED_KEY)
        st.session_state[timestamp_key] = datetime.now().isoformat()

    @classmethod
    def get_widget_state(
        cls, module: str, widget_key: str, default: Any = None
    ) -> Any:
        """
        Get saved widget state for a specific module.

        Parameters:
            module: Module identifier
            widget_key: Widget identifier
            default: Default value if not found

        Returns:
            Saved widget value or default
        """
        widget_state_key = cls._get_module_key(module, cls.WIDGET_STATE_KEY)
        widget_states = st.session_state.get(widget_state_key, {})
        return widget_states.get(widget_key, default)

    @classmethod
    def get_module_widget_key(cls, module: str, base_key: str) -> str:
        """
        Generate a unique widget key for a module.

        Parameters:
            module: Module identifier
            base_key: Base widget key

        Returns:
            Module-prefixed widget key
        """
        return f"{module}_{base_key}_widget"

    @classmethod
    def clear_widget_state(cls, module: str) -> None:
        """
        Clear all widget state for a specific module.

        Parameters:
            module: Module identifier
        """
        widget_state_key = cls._get_module_key(module, cls.WIDGET_STATE_KEY)
        if widget_state_key in st.session_state:
            del st.session_state[widget_state_key]

    @classmethod
    def get_debug_info(cls) -> Dict[str, Any]:
        """Get debug information about all module states."""
        modules = [
            cls.DASHBOARD_PREFIX,
            cls.SPM2_PREFIX,
            cls.RECIDIVISM_INBOUND_PREFIX,
            cls.RECIDIVISM_OUTBOUND_PREFIX,
        ]

        debug_info = {
            "modules": {},
            "global_keys": [],
            "total_session_keys": len(st.session_state.keys()),
        }

        for module in modules:
            debug_info["modules"][module] = cls.get_module_state_summary(
                module
            )
            is_valid, error = cls.validate_state(module)
            debug_info["modules"][module]["is_valid"] = is_valid
            debug_info["modules"][module]["validation_error"] = error

        # List relevant global keys
        global_keys = [
            k
            for k in st.session_state.keys()
            if any(prefix in k for prefix in modules + ["filters", "t0", "t1"])
        ]
        debug_info["global_keys"] = global_keys

        return debug_info
