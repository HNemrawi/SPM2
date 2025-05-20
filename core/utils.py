"""
Shared utility functions used across analysis modules
"""

import pandas as pd
import streamlit as st
from typing import List, Optional, Dict, Any, Tuple, Iterable

from config.constants import COLUMNS_TO_REMOVE, COLUMNS_TO_RENAME

def _to_list(data: Optional[Iterable]) -> List:
    """
    Convert *data* to a plain list, gracefully handling NumPy arrays,
    pandas Index objects, generators, etc.

    Parameters
    ----------
    data : Optional[Iterable]
        Anything iterable (or None).

    Returns
    -------
    List
        A concrete Python list (possibly empty).
    """
    if data is None:
        return []

    try:
        return list(data)           # most iterables
    except TypeError:               # scalar passed accidentally
        return []
    
def clean_and_standardize_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize a dataframe for display or export.
    
    - Removes unnecessary columns
    - Renames prefixed columns to standard names
    - Adds basic derived fields if missing
    
    Parameters:
        df (pd.DataFrame): Analysis results dataframe
    
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    result_df = df.copy()
    
    # Remove unnecessary columns
    cols_to_remove = [col for col in COLUMNS_TO_REMOVE if col in result_df.columns]
    if cols_to_remove:
        result_df.drop(columns=cols_to_remove, inplace=True)
    
    # Rename prefixed columns
    cols_to_rename = [col for col in COLUMNS_TO_RENAME if col in result_df.columns]
    mapping = {col: col[len("Exit_"):] for col in cols_to_rename}
    if mapping:
        result_df.rename(columns=mapping, inplace=True)
    
    return result_df


def create_multiselect_filter(                     # noqa: D401, PLR0913
    label: str,
    options: Iterable,
    *,
    default: Optional[Iterable] = None,
    help_text: str = "",
) -> Optional[List[str]]:
    """
    Render a Streamlit **multiselect** with an **“ALL”** convenience choice.

    Returns
    -------
    • ``None``            → User selected **ALL** (or list was empty).  
    • ``[]`` (empty list) → User deselected everything.  
    • ``list[str]``       → Specific items the user picked.
    """
    option_list: List[str] = _to_list(options)

    # ── EARLY-OUT ──────────────────────────────────────────────────────────
    if len(option_list) == 0:                       # ← no ambiguous truth check
        st.caption(f"ℹ️  No values available for *{label}* filter.")
        return None

    # Build the choices shown to the user
    all_option = ["ALL"]
    choices: List[str] = all_option + sorted(option_list)

    # Sanitize default selections
    default_list: List[str] = (
        [d for d in _to_list(default) if d in choices] if default else all_option
    )

    try:
        selection: List[str] = st.multiselect(
            label=label,
            options=choices,
            default=default_list,
            help=help_text,
        )
    except Exception as exc:                        # noqa: BLE001
        return None

    # Normalise output
    if not selection or "ALL" in selection:
        return None
    return selection


def filter_dataframe(
    df: pd.DataFrame,
    column: str,
    allowed_values: Optional[List[str]]
) -> pd.DataFrame:
    """
    Filter a DataFrame by allowable values in a column.
    
    Parameters:
        df (pd.DataFrame): DataFrame to filter
        column (str): Column to check
        allowed_values (Optional[List[str]]): Values to keep, or None to keep all
    
    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    if not allowed_values or column not in df.columns:
        return df
    
    return df[df[column].isin(allowed_values)]


def derive_age_categories(df: pd.DataFrame, dob_col: str, ref_date_col: str) -> pd.DataFrame:
    """
    Add age range categories to a dataframe based on date of birth.
    
    Parameters:
        df (pd.DataFrame): DataFrame to enhance
        dob_col (str): Name of date of birth column
        ref_date_col (str): Name of reference date column (e.g., exit date)
    
    Returns:
        pd.DataFrame: DataFrame with added AgeRange column
    """
    result = df.copy()
    
    if dob_col in result.columns and ref_date_col in result.columns:
        age_days = (result[ref_date_col] - result[dob_col]).dt.days
        age_years = age_days / 365.25
        
        def age_bucket(age):
            if pd.isna(age):
                return "Unknown"
            
            for bound, label in zip(
                [18, 25, 35, 45, 55, 65],
                ["0–17", "18–24", "25–34", "35–44", "45–54", "55–64"]
            ):
                if age < bound:
                    return label
            return "65+"
        
        result["AgeRange"] = age_years.apply(age_bucket)
    else:
        result["AgeRange"] = "Unknown"
    
    return result


def check_date_range_validity(
    analysis_start: pd.Timestamp,
    analysis_end: pd.Timestamp,
    data_start: pd.Timestamp,
    data_end: pd.Timestamp
) -> bool:
    """
    Check if the analysis date range is valid within the data's date range.
    
    Parameters:
        analysis_start (pd.Timestamp): Start of analysis period
        analysis_end (pd.Timestamp): End of analysis period
        data_start (pd.Timestamp): Start of data period
        data_end (pd.Timestamp): End of data period
    
    Returns:
        bool: True if valid, False otherwise
    """
    if analysis_start < data_start or analysis_end > data_end:
        st.warning(
            f"WARNING: Your selected analysis range ({analysis_start:%Y-%m-%d} to {analysis_end:%Y-%m-%d}) "
            f"is outside the data's available range ({data_start:%Y-%m-%d} to {data_end:%Y-%m-%d}). "
            "This may result in missing data. Please adjust your Reporting Period or Lookback period accordingly."
        )
        return False
    return True
