"""
Enhanced session management for HMIS Data Analysis Suite.
"""

from .keys import SessionKeys, SessionKeyValidator
from .manager import (
    ModuleType,
    SessionManager,
    check_data_available,
    clear_module_state,
    ensure_date_range,
    get_analysis_result,
    get_dashboard_state,
    get_inbound_state,
    get_outbound_state,
    get_session_manager,
    get_spm2_state,
    reset_session,
    reset_session_manager,
    set_analysis_result,
)

__all__ = [
    # Core classes
    "SessionManager",
    "SessionKeys",
    "SessionKeyValidator",
    "ModuleType",
    # Manager functions
    "get_session_manager",
    "reset_session_manager",
    "reset_session",
    "clear_module_state",
    # Data functions
    "check_data_available",
    "set_analysis_result",
    "get_analysis_result",
    "ensure_date_range",
    # Module state functions
    "get_spm2_state",
    "get_dashboard_state",
    "get_inbound_state",
    "get_outbound_state",
]
