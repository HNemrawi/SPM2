"""
Session state management and utility functions
"""

import streamlit as st
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple

def reset_session():
    """
    Reset the Streamlit session state and clear resource and data caches.
    """
    # Delete all items in the session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    # Clear the resource cache
    st.cache_resource.clear()
    
    # Clear the data cache
    st.cache_data.clear()

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

def ensure_date_range(dates, default_start: str, default_end: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
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
