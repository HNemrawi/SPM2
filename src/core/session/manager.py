"""
Enhanced session state management for HMIS Data Analysis Suite.

Provides centralized session management with validation, logging,
and namespace isolation for improved reliability and maintainability.
"""

import ast
import logging
import re
from datetime import date, datetime
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import streamlit as st

from .keys import SessionKeys, SessionKeyValidator
from .serializer import SessionSerializer

# Setup logging
logger = logging.getLogger(__name__)


class SessionManager:
    """Enhanced session manager for Streamlit state management with validation."""

    def __init__(self):
        """Initialize the session manager."""
        # Use Streamlit's session state directly
        if "initialized" not in st.session_state:
            st.session_state.initialized = True
            # Use centralized keys
            st.session_state[SessionKeys.DATA] = None
            st.session_state[SessionKeys.FILTERS] = {}
            st.session_state[SessionKeys.ANALYSIS_RESULTS] = {}
            st.session_state[SessionKeys.SELECTED_MODULE] = "general"
            logger.debug("SessionManager initialized")

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from session state with validation."""
        if not SessionKeyValidator.validate_key(key):
            logger.warning(f"Invalid key format: {key}")
        value = st.session_state.get(key, default)
        logger.debug(f"Get state: {key} = {type(value).__name__}")
        return value

    def set(self, key: str, value: Any) -> None:
        """Set value in session state with validation."""
        if not SessionKeyValidator.validate_key(key):
            logger.warning(f"Invalid key format: {key}")
        st.session_state[key] = value
        logger.debug(f"Set state: {key} = {type(value).__name__}")

    def delete(self, key: str) -> None:
        """Delete key from session state."""
        if key in st.session_state:
            del st.session_state[key]
            logger.debug(f"Deleted state: {key}")

    def exists(self, key: str) -> bool:
        """Check if key exists in session state."""
        return key in st.session_state

    def set_data(self, data: pd.DataFrame) -> None:
        """Set the main dataset with proper key management."""
        st.session_state[SessionKeys.DATA] = data
        # Also set to DF key for backward compatibility
        st.session_state[SessionKeys.DF] = data
        # Clear cached results when data changes
        st.session_state[SessionKeys.ANALYSIS_RESULTS] = {}
        logger.info(
            f"Data set: shape={data.shape if data is not None else None}"
        )

    def get_data(self) -> Optional[pd.DataFrame]:
        """Get the main dataset from either location."""
        # Check both locations for compatibility
        df = st.session_state.get(SessionKeys.DF)
        data = st.session_state.get(SessionKeys.DATA)
        return df if df is not None else data

    def has_data(self) -> bool:
        """Check if data has been loaded in either location."""
        df = st.session_state.get(SessionKeys.DF)
        data = st.session_state.get(SessionKeys.DATA)
        # Check both locations and ensure not empty
        if df is not None:
            return isinstance(df, pd.DataFrame) and not df.empty
        if data is not None:
            return isinstance(data, pd.DataFrame) and not data.empty
        return False

    def get_filters(self) -> Dict[str, Any]:
        """Get current filters."""
        return st.session_state.get(SessionKeys.FILTERS, {})

    def set_filters(self, filters: Dict[str, Any]) -> None:
        """Set filters and track change."""
        st.session_state[SessionKeys.FILTERS] = filters
        st.session_state[
            SessionKeys.LAST_FILTER_CHANGE
        ] = datetime.now().isoformat()
        logger.debug(f"Filters updated: {len(filters)} filters set")

    def get_date_range(
        self,
    ) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        """Get current date range."""
        start = st.session_state.get(SessionKeys.DATE_START)
        end = st.session_state.get(SessionKeys.DATE_END)
        return start, end

    def set_date_range(self, start: pd.Timestamp, end: pd.Timestamp) -> None:
        """Set date range."""
        st.session_state[SessionKeys.DATE_START] = start
        st.session_state[SessionKeys.DATE_END] = end
        st.session_state[
            SessionKeys.LAST_FILTER_CHANGE
        ] = datetime.now().isoformat()
        logger.debug(f"Date range set: {start} to {end}")

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session state."""
        data = self.get_data()

        # Collect module configurations
        modules = {}
        for prefix in SessionKeys.get_all_module_prefixes():
            module_keys = [
                k for k in st.session_state.keys() if k.startswith(prefix)
            ]
            if module_keys:
                module_name = prefix.rstrip("_")
                modules[module_name] = len(module_keys)

        return {
            "has_data": self.has_data(),
            "data_shape": data.shape if data is not None else (0, 0),
            "data_reference": st.session_state.get(
                SessionKeys.CURRENT_FILE, "None"
            ),
            "selected_module": self.get(
                SessionKeys.SELECTED_MODULE, "Not set"
            ),
            "filters_count": len(self.get_filters()),
            "cached_results": len(
                st.session_state.get(SessionKeys.ANALYSIS_RESULTS, {})
            ),
            "modules": modules,
        }

    def _is_exportable_key(
        self, key: str, current_module_only: bool = False
    ) -> bool:
        """
        Check if a session key should be exported.

        Args:
            key: Session state key to check
            current_module_only: If True, only export current module's keys

        Returns:
            True if the key should be exported
        """
        # Skip internal/technical keys
        skip_keys = {
            "initialized",
            SessionKeys.DF,
            SessionKeys.DATA,
            SessionKeys.ANALYSIS_RESULTS,
            SessionKeys.SHOW_EXPORT_DIALOG,
            SessionKeys.SHOW_IMPORT_DIALOG,
            SessionKeys.DATA_LOADED,
            SessionKeys.DUPLICATE_ANALYSIS,
            SessionKeys.DEDUP_ACTION,
        }

        if key in skip_keys:
            return False

        # Skip internal flags that shouldn't be exported
        internal_patterns = [
            "_dirty",
            "_last_params",
            "_analysis_requested",
            "_initialized",
        ]
        if any(pattern in key for pattern in internal_patterns):
            return False

        # Skip widget keys and internal streamlit keys
        if key.startswith("FormSubmitter:") or key.startswith("_"):
            return False

        # If exporting current module only, filter by module prefix
        if current_module_only:
            current_module = self.get(SessionKeys.SELECTED_MODULE)
            if current_module:
                # Map module names to prefixes
                module_prefix_map = {
                    "System Performance Measure 2": SessionKeys.SPM2_PREFIX,
                    "Inbound Recidivism": SessionKeys.INBOUND_PREFIX,
                    "Outbound Recidivism": SessionKeys.OUTBOUND_PREFIX,
                    "General Dashboard": SessionKeys.DASHBOARD_PREFIX,
                }
                module_prefix = module_prefix_map.get(current_module)
                if module_prefix:
                    # Export only keys for current module or global keys
                    is_current_module = key.startswith(module_prefix)
                    is_global = not any(
                        key.startswith(p)
                        for p in SessionKeys.get_all_module_prefixes()
                    )
                    if not (is_current_module or is_global):
                        return False

        # Export filter-related, module-specific, and config keys
        exportable_patterns = [
            "filter",
            "selected",
            "period",
            "date",
            "range",
            "lookback",
            "return",
            "widget",
            "current_file",
        ]

        # Check if key matches module prefixes or exportable patterns
        for prefix in SessionKeys.get_all_module_prefixes():
            if key.startswith(prefix):
                return True

        return any(pattern in key.lower() for pattern in exportable_patterns)

    def _convert_ph_destinations(self, value: Any) -> set:
        """
        Convert PH destinations value to set format.

        Args:
            value: Value that should represent PH destinations

        Returns:
            Set of destination strings
        """
        if value is None:
            return set()
        elif isinstance(value, str):
            if not value:
                return set()
            try:
                return set(ast.literal_eval(value))
            except Exception:
                logger.warning(
                    f"Failed to parse PH destinations string: {value}"
                )
                return set()
        elif isinstance(value, list):
            return set(value)
        elif isinstance(value, set):
            return value
        else:
            logger.warning(
                f"Unexpected type for PH destinations: {type(value)}"
            )
            return set()

    def _convert_from_json(self, key: str, value: Any) -> Any:
        """
        Recursively convert loaded JSON values to Python objects.

        Args:
            key: The session state key
            value: Value to convert

        Returns:
            Converted value ready for session state
        """
        # Handle date/datetime strings
        if isinstance(value, str):
            # Try ISO date format (YYYY-MM-DD) for date_input widgets
            if re.match(r"^\d{4}-\d{2}-\d{2}$", value):
                try:
                    return pd.to_datetime(value).date()
                except Exception:
                    pass  # Not a valid date string
            # Try ISO datetime format (has T)
            elif "T" in value:
                try:
                    return pd.to_datetime(value)
                except Exception:
                    pass  # Not a datetime string

        # Handle PH destinations specially
        if (
            key == SessionKeys.SELECTED_PH_DESTINATIONS
            or key == "ph_destination_selector_widget"
            or key == "ph_destination_selector"
        ):
            return self._convert_ph_destinations(value)

        # Handle nested dictionaries (like filters)
        if isinstance(value, dict):
            converted_dict = {}
            for k, v in value.items():
                # Check if this dict value is PH destinations
                if k == "selected_ph_destinations":
                    converted_dict[k] = self._convert_ph_destinations(v)
                else:
                    converted_dict[k] = self._convert_from_json(k, v)
            return converted_dict

        # Handle lists (recursively convert items)
        if isinstance(value, list):
            return [self._convert_from_json("", item) for item in value]

        # Return primitives as-is
        return value

    def _convert_for_json(self, value: Any) -> Any:
        """
        Recursively convert Python objects to JSON-serializable formats.

        Args:
            value: Value to convert

        Returns:
            JSON-serializable version of the value
        """
        # Check date BEFORE pd.Timestamp since Timestamp is also a date subclass
        if isinstance(value, date) and not isinstance(value, datetime):
            # Python date object (from date_input widgets)
            return value.isoformat()
        elif isinstance(value, (pd.Timestamp, datetime)):
            return value.isoformat()
        elif isinstance(value, (pd.DataFrame, pd.Series)):
            # Don't serialize dataframes
            return None
        elif isinstance(value, set):
            # Convert sets to lists
            return list(value)
        elif isinstance(value, dict):
            # Recursively convert dictionary values
            return {k: self._convert_for_json(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            # Recursively convert list/tuple items
            return [self._convert_for_json(item) for item in value]
        else:
            # Return as-is for primitives (str, int, float, bool, None)
            return value

    def export_state(
        self,
        include_all: bool = False,
        current_module_only: bool = True,
        session_name: str = "",
        session_description: str = "",
    ) -> Dict[str, Any]:
        """
        Export session state for saving/sharing.

        Args:
            include_all: If True, include data metadata
            current_module_only: If True, export only current module's config
            session_name: Optional name for this session
            session_description: Optional description

        Returns:
            Dictionary containing exportable state
        """
        # Collect all exportable state keys
        state_snapshot = {}
        for key in st.session_state.keys():
            if self._is_exportable_key(key, current_module_only):
                value = st.session_state[key]
                # Skip dataframes
                if isinstance(value, (pd.DataFrame, pd.Series)):
                    continue
                # Recursively convert value to JSON-serializable format
                converted = self._convert_for_json(value)
                if converted is not None:
                    state_snapshot[key] = converted

        # Get current filename if available
        filename = st.session_state.get(SessionKeys.CURRENT_FILE, "Unknown")
        current_module = self.get(SessionKeys.SELECTED_MODULE, "Unknown")

        # Generate default session name if not provided
        if not session_name:
            module_short = current_module.replace(
                "System Performance Measure 2", "SPM2"
            ).replace(" ", "_")
            session_name = (
                f"{module_short}_{datetime.now().strftime('%Y%m%d_%H%M')}"
            )

        export_data = {
            "version": "2.1.0",  # Bump version for new format
            "session_info": {
                "name": session_name,
                "description": session_description,
                "created_at": datetime.now().isoformat(),
                "data_file": filename,
            },
            "module": current_module,
            "state": state_snapshot,
        }

        # Add human-readable configuration summary
        config_summary = self._create_config_summary(
            state_snapshot, current_module
        )
        export_data["configuration_summary"] = config_summary

        # Add module-specific summaries (for multi-module exports)
        if not current_module_only:
            modules_info = {}
            for prefix in SessionKeys.get_all_module_prefixes():
                module_keys = [
                    k for k in state_snapshot if k.startswith(prefix)
                ]
                if module_keys:
                    module_name = prefix.rstrip("_")
                    modules_info[module_name] = len(module_keys)
            export_data["modules"] = modules_info

        if include_all and self.has_data():
            # Include data info but not the actual data (too large)
            data = self.get_data()
            export_data["data_info"] = {
                "shape": data.shape,
                "columns": list(data.columns),
            }

        return export_data

    def _create_config_summary(
        self, state_snapshot: Dict[str, Any], module_name: str
    ) -> Dict[str, Any]:
        """
        Create human-readable configuration summary.

        Args:
            state_snapshot: The state being exported
            module_name: Name of the module

        Returns:
            Dictionary with human-readable summaries
        """
        summary = {}

        # Extract date ranges
        date_keys = [k for k in state_snapshot if "date" in k.lower()]
        if date_keys:
            date_values = []
            for key in date_keys:
                val = state_snapshot[key]
                if isinstance(val, (list, tuple)) and len(val) == 2:
                    try:
                        start = pd.to_datetime(val[0]).strftime("%b %d, %Y")
                        end = pd.to_datetime(val[1]).strftime("%b %d, %Y")
                        date_values.append(f"{start} - {end}")
                    except Exception:
                        pass
            if date_values:
                summary["date_ranges"] = date_values

        # Count filter categories
        filter_categories = {}
        for key in state_snapshot:
            if "filter" in key.lower() or "selected" in key.lower():
                value = state_snapshot[key]
                if isinstance(value, list) and len(value) > 0:
                    # Extract category name from key
                    category = (
                        key.replace("_filter", "")
                        .replace("selected_", "")
                        .replace("_", " ")
                        .title()
                    )

                    # Flatten nested lists if present (handles various list structures)
                    def count_items(lst):
                        """Recursively count items in nested list structures."""
                        count = 0
                        for item in lst:
                            if isinstance(item, (list, tuple)):
                                count += count_items(item)
                            else:
                                count += 1
                        return count

                    item_count = count_items(value)
                    filter_categories[category] = f"{item_count} selected"

        if filter_categories:
            summary["filters"] = filter_categories

        # Module-specific configurations
        if "SPM2" in module_name or "spm2" in module_name.lower():
            lookback = state_snapshot.get("spm2_lookback_period")
            return_period = state_snapshot.get("spm2_return_period")
            if lookback:
                summary["lookback_period"] = f"{lookback} days"
            if return_period:
                summary["return_period"] = f"{return_period} days"

        return summary

    def import_state(
        self, state_data: Dict[str, Any], validate: bool = True
    ) -> list:
        """
        Import session state from saved data.

        Args:
            state_data: Dictionary containing exported state
            validate: Whether to validate before importing

        Returns:
            List of issues encountered (empty if successful)
        """
        issues = []

        try:
            # Validate if requested
            if validate:
                validation_issues = SessionSerializer.validate_import(
                    state_data, None
                )
                if validation_issues:
                    # Filter for actual errors (❌) not info (ℹ️) or warnings (⚠️)
                    errors = [
                        msg for msg in validation_issues if msg.startswith("❌")
                    ]
                    if errors:
                        issues.extend(errors)
                        return issues
                    # Log warnings/info but continue with import
                    for msg in validation_issues:
                        if msg.startswith("⚠️") or msg.startswith("ℹ️"):
                            logger.info(f"Import note: {msg}")

            # Get the state snapshot (handle both old and new formats)
            if "state" in state_data:
                # New format with nested state (v2.0+)
                state_snapshot = state_data["state"]
            else:
                # Old format - backward compatibility (v1.x)
                state_snapshot = {
                    "selected_module": state_data.get("selected_module"),
                    "filters": state_data.get("filters", {}),
                }

            # Handle v2.1+ format with session_info
            if "session_info" in state_data:
                logger.info(
                    f"Importing session: {state_data['session_info'].get('name', 'Unnamed')}"
                )

            # Handle v2.1+ module field (replaces selected_module at root)
            if "module" in state_data:
                target_module = state_data["module"]
            elif "selected_module" in state_data:
                target_module = state_data["selected_module"]
            else:
                target_module = None

            # Import all state keys
            imported_count = 0
            for key, value in state_snapshot.items():
                try:
                    # Recursively convert loaded value to proper Python objects
                    converted_value = self._convert_from_json(key, value)

                    st.session_state[key] = converted_value
                    imported_count += 1
                except Exception as e:
                    issues.append(f"Failed to import key '{key}': {str(e)}")
                    logger.warning(f"Failed to import key {key}: {str(e)}")

            # Import module selection from wherever it is
            if target_module:
                st.session_state[SessionKeys.SELECTED_MODULE] = target_module
                st.session_state[SessionKeys.CURRENT_MODULE] = target_module
                logger.info(
                    f"Session state imported: {imported_count} keys restored for module {target_module}"
                )
            else:
                logger.info(
                    f"Session state imported: {imported_count} keys restored"
                )

        except Exception as e:
            error_msg = f"Error importing state: {str(e)}"
            logger.error(error_msg)
            issues.append(error_msg)

        return issues

    def export_to_json(
        self,
        include_all: bool = False,
        current_module_only: bool = True,
        session_name: str = "",
        session_description: str = "",
    ) -> str:
        """
        Export session state as JSON string.

        Args:
            include_all: If True, include data metadata
            current_module_only: If True, export only current module's config
            session_name: Optional name for this session
            session_description: Optional description

        Returns:
            JSON string of exported state
        """
        import json

        return json.dumps(
            self.export_state(
                include_all,
                current_module_only,
                session_name,
                session_description,
            ),
            indent=2,
            default=str,
        )

    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information about session state."""
        import sys

        data = self.get_data()
        data_size = 0
        if data is not None:
            data_size = sys.getsizeof(data)

        return {
            "store_info": {
                "total_keys": len(st.session_state),
                "memory_usage": {"total_size_bytes": data_size},
            }
        }

    def get_module_state(self, module_type):
        """Get module state (for compatibility)."""
        if module_type == ModuleType.SPM2:
            return get_spm2_state()
        elif module_type == ModuleType.DASHBOARD:
            return get_dashboard_state()
        elif module_type == ModuleType.INBOUND:
            return get_inbound_state()
        elif module_type == ModuleType.OUTBOUND:
            return get_outbound_state()
        return None


# Global instance
_session_manager = None


def get_session_manager() -> SessionManager:
    """Get the global session manager instance."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


def reset_session_manager() -> None:
    """Reset the session manager instance."""
    global _session_manager
    _session_manager = None
    # Clear Streamlit session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.cache_data.clear()
    logger.info("Session manager reset")


def clear_module_state(module_prefix: str) -> None:
    """Clear all state for a specific module."""
    keys_to_delete = [
        k
        for k in st.session_state.keys()
        if SessionKeys.is_module_key(k, module_prefix)
    ]
    for key in keys_to_delete:
        del st.session_state[key]
    logger.debug(
        f"Cleared {len(keys_to_delete)} keys for module {module_prefix}"
    )


def reset_session() -> None:
    """Reset the Streamlit session state and clear all caches."""
    reset_session_manager()


def set_analysis_result(analysis_type: str, data: pd.DataFrame) -> None:
    """Store analysis results in session state."""
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = {}
    st.session_state.analysis_results[analysis_type] = data


def get_analysis_result(analysis_type: str) -> Optional[pd.DataFrame]:
    """Retrieve analysis results from session state."""
    return st.session_state.get("analysis_results", {}).get(analysis_type)


def check_data_available() -> Optional[pd.DataFrame]:
    """Check if data has been uploaded and is available."""
    return st.session_state.get("data")


def get_session_info() -> dict:
    """Get information about current session state and cache status."""
    session_manager = get_session_manager()
    debug_info = session_manager.get_debug_info()

    return {
        "session_keys": debug_info.get("store_info", {}).get("total_keys", 0),
        "session_items": list(st.session_state.keys()),
        "has_data": session_manager.has_data(),
        "data_size": debug_info.get("store_info", {})
        .get("memory_usage", {})
        .get("total_size_bytes", 0),
        "debug_info": debug_info,
    }


def ensure_date_range(
    dates, default_start: str, default_end: str
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Ensure a valid date range is selected or use defaults."""
    if not dates or len(dates) != 2:
        start_date = pd.to_datetime(default_start)
        end_date = pd.to_datetime(default_end)
    else:
        start_date = pd.to_datetime(dates[0])
        end_date = pd.to_datetime(dates[1])

    return start_date, end_date


# Module state functions for backward compatibility
def get_spm2_state():
    """Get SPM2 module state (simplified)."""
    return SimpleModuleState("spm2")


def get_dashboard_state():
    """Get Dashboard module state (simplified)."""
    return SimpleModuleState("dashboard")


def get_inbound_state():
    """Get Inbound module state (simplified)."""
    return SimpleModuleState("inbound")


def get_outbound_state():
    """Get Outbound module state (simplified)."""
    return SimpleModuleState("outbound")


class SimpleModuleState:
    """Simple module state handler for backward compatibility."""

    def __init__(self, module_name: str):
        self.module_name = module_name
        self.key_prefix = f"{module_name}_"
        self.initialize()

    def initialize(self) -> None:
        """Initialize module state."""
        if f"{self.key_prefix}initialized" not in st.session_state:
            st.session_state[f"{self.key_prefix}initialized"] = True
            st.session_state[f"{self.key_prefix}dirty"] = False
            st.session_state[f"{self.key_prefix}last_params"] = {}
            st.session_state[f"{self.key_prefix}initializing"] = False

    def get(self, key: str, default: Any = None) -> Any:
        """Get module-specific value."""
        return st.session_state.get(f"{self.key_prefix}{key}", default)

    def set(self, key: str, value: Any) -> None:
        """Set module-specific value."""
        st.session_state[f"{self.key_prefix}{key}"] = value

    def clear(self) -> None:
        """Clear all module-specific state."""
        keys_to_delete = [
            k for k in st.session_state.keys() if k.startswith(self.key_prefix)
        ]
        for key in keys_to_delete:
            del st.session_state[key]

    def start_initialization(self) -> None:
        """Mark the start of initialization phase to prevent dirty flagging."""
        st.session_state[f"{self.key_prefix}initializing"] = True
        logger.debug(f"{self.module_name}: Started initialization phase")

    def end_initialization(self) -> None:
        """Mark the end of initialization phase and save baseline snapshot."""
        st.session_state[f"{self.key_prefix}initializing"] = False
        # Save snapshot of all parameters as baseline after initialization
        self.save_params_snapshot()
        # Ensure dirty flag is clear after initialization
        self.clear_dirty()
        logger.debug(f"{self.module_name}: Ended initialization phase")

    def is_initializing(self) -> bool:
        """Check if module is in initialization phase."""
        return st.session_state.get(f"{self.key_prefix}initializing", False)

    def is_dirty(self) -> bool:
        """Check if module state has changed."""
        return st.session_state.get(f"{self.key_prefix}dirty", False)

    def mark_dirty(self, debounce: bool = False) -> None:
        """Mark module state as changed (skipped during initialization)."""
        # Don't mark dirty during initialization phase
        if self.is_initializing():
            logger.debug(
                f"{self.module_name}: Skipping mark_dirty during initialization"
            )
            return

        st.session_state[f"{self.key_prefix}dirty"] = True
        logger.debug(f"{self.module_name}: Marked dirty")

    def clear_dirty(self) -> None:
        """Clear dirty flag."""
        st.session_state[f"{self.key_prefix}dirty"] = False

    def check_and_mark_dirty(self, param_key: str, new_value: Any) -> bool:
        """Check if a parameter value has changed and mark dirty if it has."""
        # Skip during initialization - just save the value
        if self.is_initializing():
            if f"{self.key_prefix}last_params" not in st.session_state:
                st.session_state[f"{self.key_prefix}last_params"] = {}
            st.session_state[f"{self.key_prefix}last_params"][
                param_key
            ] = new_value
            logger.debug(
                f"{self.module_name}: Saved param '{param_key}' during initialization"
            )
            return False

        last_params = st.session_state.get(f"{self.key_prefix}last_params", {})

        # If no baseline exists yet, this is first setup - don't mark dirty
        if not last_params:
            # Initialize the params dict but don't mark dirty
            if f"{self.key_prefix}last_params" not in st.session_state:
                st.session_state[f"{self.key_prefix}last_params"] = {}
            st.session_state[f"{self.key_prefix}last_params"][
                param_key
            ] = new_value
            logger.debug(
                f"{self.module_name}: Initialized param '{param_key}' (not marking dirty)"
            )
            return False

        # Compare with last known value
        if param_key in last_params:
            # For lists/multiselects, compare sorted values
            if isinstance(new_value, list) and isinstance(
                last_params[param_key], list
            ):
                changed = sorted(new_value) != sorted(last_params[param_key])
            else:
                changed = new_value != last_params[param_key]
        else:
            # Parameter not in baseline - this is a new widget being added
            # during initialization, so don't mark dirty
            logger.debug(
                f"{self.module_name}: New param '{param_key}' added to baseline (not marking dirty)"
            )
            changed = False

        # Update last known value
        st.session_state[f"{self.key_prefix}last_params"][
            param_key
        ] = new_value

        # Mark dirty only if changed
        if changed:
            self.mark_dirty()
            logger.debug(
                f"{self.module_name}: Parameter '{param_key}' changed, marking dirty"
            )

        return changed

    def save_params_snapshot(self) -> None:
        """Save current parameter values as the baseline for change detection."""
        # This is called after running analysis to reset the baseline
        current_params = {}

        # Also capture any parameters that were explicitly set via check_and_mark_dirty
        if f"{self.key_prefix}last_params" in st.session_state:
            # Start with existing tracked params to ensure we don't lose any
            current_params = st.session_state[
                f"{self.key_prefix}last_params"
            ].copy()

        # Collect all current widget and filter values
        for key in st.session_state.keys():
            # Capture filter keys (e.g., spm2_continuum_filter, spm2_exit_cocs_filter)
            if key.startswith(f"{self.module_name}_") and "_filter" in key:
                # Keep the full key including prefix for consistent comparison
                current_params[key] = st.session_state[key]
            # Capture widget keys (e.g., lookback_days, return_period_days)
            elif key.startswith(self.module_name) and any(
                suffix in key
                for suffix in [
                    "_widget",
                    "lookback",
                    "return_period",
                    "date_range",
                    "reporting",
                    "checkbox",
                    "radio",
                    "unit",
                ]
            ):
                # Keep the full key including prefix for consistent comparison
                current_params[key] = st.session_state[key]

        st.session_state[f"{self.key_prefix}last_params"] = current_params
        logger.debug(
            f"{self.module_name}: Saved params snapshot with {len(current_params)} parameters"
        )

    def request_analysis(self) -> None:
        """Request analysis to be run."""
        st.session_state[f"{self.key_prefix}analysis_requested"] = True
        self.clear_dirty()

    def clear_analysis_request(self) -> None:
        """Clear analysis request flag and save params snapshot."""
        st.session_state[f"{self.key_prefix}analysis_requested"] = False
        # Save current params as baseline after successful analysis
        self.save_params_snapshot()

    def is_analysis_requested(self) -> bool:
        """Check if analysis was requested."""
        return st.session_state.get(
            f"{self.key_prefix}analysis_requested", False
        )

    def get_widget_state(self, widget_key: str, default: Any = None) -> Any:
        """Get saved widget state."""
        return st.session_state.get(
            f"{self.key_prefix}widget_{widget_key}", default
        )

    def set_widget_state(self, widget_key: str, value: Any) -> None:
        """Save widget state."""
        st.session_state[f"{self.key_prefix}widget_{widget_key}"] = value

    # SPM2-specific methods
    def get_analysis_config(self) -> Dict[str, Any]:
        """Get analysis configuration."""
        return {
            "reporting_period": {
                "start": self.get("report_start"),
                "end": self.get("report_end"),
            },
            "lookback": self.get_lookback_period(),
            "return_period": self.get_return_period(),
        }

    def get_lookback_period(self) -> Dict[str, Any]:
        """Get lookback period configuration."""
        return {
            "period": self.get("lookback_period", 730),
            "unit": self.get("lookback_unit", "Days"),
        }

    def set_lookback_period(self, period: int, unit: str) -> None:
        """Set lookback period configuration."""
        self.set("lookback_period", period)
        self.set("lookback_unit", unit)
        self.mark_dirty()

    def get_return_period(self) -> int:
        """Get return period in days."""
        return self.get("return_period", 730)

    def set_return_period(self, days: int) -> None:
        """Set return period in days."""
        self.set("return_period", days)
        self.mark_dirty()

    def set_date_range(self, start_date, end_date) -> None:
        """Set date range for analysis."""
        self.set("report_start", start_date)
        self.set("report_end", end_date)
        self.mark_dirty()


# Simple ModuleType enum for backward compatibility
class ModuleType:
    SPM2 = "spm2"
    DASHBOARD = "dashboard"
    INBOUND = "inbound"
    OUTBOUND = "outbound"
