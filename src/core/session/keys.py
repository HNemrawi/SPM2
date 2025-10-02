"""
Centralized session state key management.

This module defines all session state keys used throughout the application
to prevent conflicts and ensure consistency.
"""


class SessionKeys:
    """Central registry for all session state keys."""

    # ==================== GLOBAL KEYS ====================
    # Core data storage
    DATA = "data"  # Primary data storage location
    DF = "df"  # Legacy compatibility
    CURRENT_FILE = "current_file"
    DATA_LOADED = "data_loaded"

    # Analysis state
    FILTERS = "filters"
    ANALYSIS_RESULTS = "analysis_results"
    SELECTED_MODULE = "selected_module"
    CURRENT_MODULE = "current_module"

    # Duplicate handling
    DUPLICATE_ANALYSIS = "duplicate_analysis"
    DEDUP_ACTION = "dedup_action"

    # UI state
    SHOW_EXPORT_DIALOG = "show_export_dialog"
    SHOW_IMPORT_DIALOG = "show_import_dialog"

    # ==================== DATE MANAGEMENT ====================
    # Current period
    DATE_START = "t0"  # Current window start
    DATE_END = "t1"  # Current window end

    # Previous period (for comparison)
    PREV_START = "prev_start"
    PREV_END = "prev_end"

    # Filter tracking
    LAST_FILTER_CHANGE = "last_filter_change"

    # ==================== FILTER SPECIFIC ====================
    SELECTED_PH_DESTINATIONS = "selected_ph_destinations"
    STATE_FILTER_FORM = "state_filter_form"

    # Filtered data cache
    DF_FILTERED = "df_filt"

    # ==================== MODULE PREFIXES ====================
    DASHBOARD_PREFIX = "dashboard_"
    SPM2_PREFIX = "spm2_"
    INBOUND_PREFIX = "inbound_"
    OUTBOUND_PREFIX = "outbound_"

    # ==================== MODULE SPECIFIC KEYS ====================
    # Dashboard specific
    DASHBOARD_DIRTY = "dashboard_dirty"
    DASHBOARD_ANALYSIS_REQUESTED = "dashboard_analysis_requested"

    # SPM2 specific
    SPM2_LOOKBACK_DAYS = "lookback_days"
    SPM2_LOOKBACK_MONTHS = "lookback_months"
    SPM2_RETURN_PERIOD = "return_period_days"

    # Inbound specific
    INBOUND_DAYS_LOOKBACK = "inbound_days_lookback"

    @classmethod
    def get_module_key(cls, module_prefix: str, key: str) -> str:
        """
        Generate a properly namespaced module key.

        Args:
            module_prefix: Module prefix (e.g., DASHBOARD_PREFIX)
            key: Key name

        Returns:
            Namespaced key string
        """
        return f"{module_prefix}{key}"

    @classmethod
    def is_module_key(cls, key: str, module_prefix: str) -> bool:
        """
        Check if a key belongs to a specific module.

        Args:
            key: Key to check
            module_prefix: Module prefix to check against

        Returns:
            True if key belongs to module
        """
        return key.startswith(module_prefix)

    @classmethod
    def get_all_module_prefixes(cls) -> list:
        """Get all module prefixes."""
        return [
            cls.DASHBOARD_PREFIX,
            cls.SPM2_PREFIX,
            cls.INBOUND_PREFIX,
            cls.OUTBOUND_PREFIX,
        ]

    @classmethod
    def strip_module_prefix(cls, key: str) -> tuple:
        """
        Strip module prefix from a key.

        Args:
            key: Full key with potential prefix

        Returns:
            Tuple of (module_prefix, base_key)
        """
        for prefix in cls.get_all_module_prefixes():
            if key.startswith(prefix):
                return prefix, key[len(prefix) :]
        return "", key


class SessionKeyValidator:
    """Validator for session state keys."""

    @staticmethod
    def validate_key(key: str) -> bool:
        """
        Validate that a key follows naming conventions.

        Args:
            key: Key to validate

        Returns:
            True if valid
        """
        if not key:
            return False

        # Check for spaces or special characters
        if " " in key or "-" in key:
            return False

        # Key should be lowercase with underscores
        if key != key.lower():
            return False

        return True

    @staticmethod
    def suggest_key(bad_key: str) -> str:
        """
        Suggest a valid key based on an invalid one.

        Args:
            bad_key: Invalid key

        Returns:
            Suggested valid key
        """
        # Replace spaces and hyphens with underscores
        suggested = bad_key.replace(" ", "_").replace("-", "_")

        # Convert to lowercase
        suggested = suggested.lower()

        # Remove special characters
        import re

        suggested = re.sub(r"[^a-z0-9_]", "", suggested)

        return suggested
