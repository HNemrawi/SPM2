"""
Shared utility functions used across analysis modules
"""

import pandas as pd
import streamlit as st
from typing import List, Optional, Dict, Any, Tuple, Iterable, Union, TypeVar, Protocol
from functools import lru_cache
from datetime import datetime
import numpy as np

from config.constants import COLUMNS_TO_REMOVE, COLUMNS_TO_RENAME

# Type definitions
T = TypeVar('T')
DateLike = Union[pd.Timestamp, datetime, str]

class DataFrameFilter(Protocol):
   """Protocol for dataframe filter functions"""
   def __call__(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame: ...

# ==================== DATA CONVERSION UTILITIES ====================

def _to_list(data: Optional[Iterable[T]]) -> List[T]:
   """
   Convert any iterable to a list, handling edge cases gracefully.
   
   Parameters:
       data: Any iterable, None, or scalar value
   
   Returns:
       List containing the items, empty list if None or invalid
   """
   if data is None:
       return []
   
   if isinstance(data, (str, bytes)):
       return [data]
   
   if hasattr(data, '__iter__'):
       try:
           return list(data)
       except (TypeError, ValueError):
           return []
   
   return []

# ==================== DATAFRAME OPERATIONS ====================

def clean_and_standardize_df(
   df: pd.DataFrame,
   inplace: bool = False,
   additional_columns_to_remove: Optional[List[str]] = None,
   additional_rename_mapping: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
   """
   Clean and standardize a dataframe for display or export.
   
   Parameters:
       df: Analysis results dataframe
       inplace: Whether to modify the original dataframe
       additional_columns_to_remove: Extra columns to remove beyond defaults
       additional_rename_mapping: Extra column rename mappings
   
   Returns:
       Cleaned dataframe
   """
   result_df = df if inplace else df.copy()
   
   # Combine default and additional columns to remove
   cols_to_remove = set(COLUMNS_TO_REMOVE)
   if additional_columns_to_remove:
       cols_to_remove.update(additional_columns_to_remove)
   
   # Remove columns that exist
   existing_cols_to_remove = [col for col in cols_to_remove if col in result_df.columns]
   if existing_cols_to_remove:
       result_df.drop(columns=existing_cols_to_remove, inplace=True)
   
   # Build rename mapping
   rename_mapping = {}
   
   # Default prefix removal
   for col in result_df.columns:
       if col in COLUMNS_TO_RENAME:
           rename_mapping[col] = col[len("Exit_"):]
   
   # Additional rename mappings
   if additional_rename_mapping:
       rename_mapping.update(additional_rename_mapping)
   
   # Apply renaming
   if rename_mapping:
       result_df.rename(columns=rename_mapping, inplace=True)
   
   # Sort columns alphabetically for consistency
   result_df = result_df.reindex(sorted(result_df.columns), axis=1)
   
   return result_df

# ==================== FILTER COMPONENTS ====================

def create_multiselect_filter(
   label: str,
   options: Iterable[str],
   *,
   default: Optional[Iterable[str]] = None,
   help_text: str = "",
   key: Optional[str] = None,
   allow_empty: bool = True,
   sort_options: bool = True,
   placeholder: str = "Select values..."
) -> Optional[List[str]]:
   """
   Create an enhanced multiselect filter with ALL option.
   
   Parameters:
       label: Filter label
       options: Available options
       default: Default selections
       help_text: Help tooltip text
       key: Unique widget key
       allow_empty: Whether to allow empty selection
       sort_options: Whether to sort options alphabetically
       placeholder: Placeholder text when empty
   
   Returns:
       Selected values or None if ALL selected
   """
   option_list = _to_list(options)
   
   if not option_list:
       st.caption(f"ℹ️ No values available for *{label}* filter.")
       return None
   
   # Prepare options
   unique_options = list(dict.fromkeys(option_list))  # Preserve order, remove duplicates
   if sort_options:
       unique_options = sorted(unique_options)
   
   all_option = ["ALL"]
   choices = all_option + unique_options
   
   # Determine defaults
   if default is None:
       default_list = all_option
   else:
       default_list = [d for d in _to_list(default) if d in choices]
       if not default_list:
           default_list = all_option
   
   # Create widget
   try:
       selection = st.multiselect(
           label=label,
           options=choices,
           default=default_list,
           help=help_text,
           key=key,
           placeholder=placeholder
       )
   except Exception:
       st.error(f"Error creating filter for {label}")
       return None
   
   # Process selection
   if not selection:
       return [] if allow_empty else None
   
   if "ALL" in selection:
       return None
   
   return selection

# ==================== DATAFRAME FILTERING ====================

@lru_cache(maxsize=128)
def _create_filter_mask(
   df_hash: str,
   column: str,
   allowed_values: Tuple[str, ...]
) -> pd.Series:
   """Cached filter mask creation for performance"""
   # This is a placeholder - actual implementation would need the df
   pass

def filter_dataframe(
   df: pd.DataFrame,
   column: str,
   allowed_values: Optional[List[str]],
   case_sensitive: bool = True,
   na_action: str = 'exclude'
) -> pd.DataFrame:
   """
   Enhanced dataframe filtering with additional options.
   
   Parameters:
       df: DataFrame to filter
       column: Column to check
       allowed_values: Values to keep, or None to keep all
       case_sensitive: Whether to use case-sensitive matching
       na_action: How to handle NaN values ('exclude', 'include', 'only')
   
   Returns:
       Filtered DataFrame
   """
   if allowed_values is None or column not in df.columns:
       return df
   
   if not allowed_values and na_action != 'only':
       return df.iloc[0:0]  # Return empty df with same structure
   
   # Create filter mask
   if case_sensitive:
       mask = df[column].isin(allowed_values)
   else:
       lower_values = [str(v).lower() for v in allowed_values]
       mask = df[column].astype(str).str.lower().isin(lower_values)
   
   # Handle NaN values
   if na_action == 'include':
       mask |= df[column].isna()
   elif na_action == 'only':
       return df[df[column].isna()]
   
   return df[mask]

def apply_multiple_filters(
   df: pd.DataFrame,
   filters: Dict[str, Optional[List[str]]],
   operator: str = 'AND'
) -> pd.DataFrame:
   """
   Apply multiple filters to a dataframe.
   
   Parameters:
       df: DataFrame to filter
       filters: Dictionary of column: allowed_values pairs
       operator: How to combine filters ('AND' or 'OR')
   
   Returns:
       Filtered DataFrame
   """
   if not filters:
       return df
   
   masks = []
   for column, values in filters.items():
       if values is not None and column in df.columns:
           masks.append(df[column].isin(values))
   
   if not masks:
       return df
   
   if operator == 'AND':
       combined_mask = pd.concat(masks, axis=1).all(axis=1)
   else:
       combined_mask = pd.concat(masks, axis=1).any(axis=1)
   
   return df[combined_mask]

# ==================== AGE CATEGORIZATION ====================

# Age boundaries and labels
AGE_BOUNDARIES = [18, 25, 35, 45, 55, 65]
AGE_LABELS = ["0-17", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]

def categorize_age(age: Optional[float]) -> str:
   """Categorize a single age value"""
   if pd.isna(age) or age < 0:
       return "Unknown"
   
   for boundary, label in zip(AGE_BOUNDARIES, AGE_LABELS):
       if age < boundary:
           return label
   
   return AGE_LABELS[-1]

def derive_age_categories(
   df: pd.DataFrame,
   dob_col: str,
   ref_date_col: str,
   age_col_name: str = "AgeRange",
   include_age_years: bool = False
) -> pd.DataFrame:
   """
   Enhanced age categorization with additional features.
   
   Parameters:
       df: DataFrame to enhance
       dob_col: Name of date of birth column
       ref_date_col: Name of reference date column
       age_col_name: Name for the age category column
       include_age_years: Whether to include numeric age column
   
   Returns:
       DataFrame with age categories added
   """
   result = df.copy()
   
   if dob_col not in result.columns or ref_date_col not in result.columns:
       result[age_col_name] = "Unknown"
       if include_age_years:
           result["AgeYears"] = np.nan
       return result
   
   # Calculate age in years
   try:
       age_delta = pd.to_datetime(result[ref_date_col]) - pd.to_datetime(result[dob_col])
       age_years = age_delta.dt.total_seconds() / (365.25 * 24 * 60 * 60)
       
       if include_age_years:
           result["AgeYears"] = age_years.round(1)
       
       # Vectorized categorization
       result[age_col_name] = pd.cut(
           age_years,
           bins=[-np.inf] + AGE_BOUNDARIES + [np.inf],
           labels=["Unknown"] + AGE_LABELS,
           right=False
       ).fillna("Unknown")
       
   except Exception:
       result[age_col_name] = "Unknown"
       if include_age_years:
           result["AgeYears"] = np.nan
   
   return result

# ==================== DATE VALIDATION ====================

def check_date_range_validity(
   analysis_start: DateLike,
   analysis_end: DateLike,
   data_start: DateLike,
   data_end: DateLike,
   warn: bool = True,
   error_on_invalid: bool = False
) -> Tuple[bool, Optional[str]]:
   """
   Enhanced date range validation with detailed feedback.
   
   Parameters:
       analysis_start: Start of analysis period
       analysis_end: End of analysis period
       data_start: Start of data period
       data_end: End of data period
       warn: Whether to show warning in UI
       error_on_invalid: Whether to raise exception on invalid range
   
   Returns:
       Tuple of (is_valid, error_message)
   """
   # Convert to timestamps
   analysis_start = pd.to_datetime(analysis_start)
   analysis_end = pd.to_datetime(analysis_end)
   data_start = pd.to_datetime(data_start)
   data_end = pd.to_datetime(data_end)
   
   # Check various validity conditions
   issues = []
   
   if analysis_start > analysis_end:
       issues.append("Analysis start date is after end date")
   
   if analysis_start < data_start:
       days_before = (data_start - analysis_start).days
       issues.append(f"Analysis starts {days_before} days before available data")
   
   if analysis_end > data_end:
       days_after = (analysis_end - data_end).days
       issues.append(f"Analysis ends {days_after} days after available data")
   
   # Build message
   if issues:
       message = (
           f"Date range issues detected:\n"
           f"• " + "\n• ".join(issues) + "\n\n"
           f"Analysis range: {analysis_start:%Y-%m-%d} to {analysis_end:%Y-%m-%d}\n"
           f"Data range: {data_start:%Y-%m-%d} to {data_end:%Y-%m-%d}"
       )
       
       if warn:
           st.warning(message)
       
       if error_on_invalid:
           raise ValueError(message)
       
       return False, message
   
   return True, None

def suggest_valid_date_range(
   requested_start: DateLike,
   requested_end: DateLike,
   data_start: DateLike,
   data_end: DateLike
) -> Tuple[pd.Timestamp, pd.Timestamp]:
   """
   Suggest a valid date range based on requested and available dates.
   
   Returns:
       Tuple of (suggested_start, suggested_end)
   """
   requested_start = pd.to_datetime(requested_start)
   requested_end = pd.to_datetime(requested_end)
   data_start = pd.to_datetime(data_start)
   data_end = pd.to_datetime(data_end)
   
   # Constrain to available data
   suggested_start = max(requested_start, data_start)
   suggested_end = min(requested_end, data_end)
   
   # Ensure valid range
   if suggested_start > suggested_end:
       suggested_start = suggested_end - pd.Timedelta(days=30)
   
   return suggested_start, suggested_end

# ==================== UTILITY FUNCTIONS ====================

def format_percentage(
   value: float,
   decimals: int = 1,
   include_sign: bool = False
) -> str:
   """Format a number as percentage with consistent styling"""
   if pd.isna(value):
       return "—"
   
   formatted = f"{value:.{decimals}f}%"
   if include_sign and value > 0:
       formatted = f"+{formatted}"
   
   return formatted

def format_count(
   value: Union[int, float],
   decimals: int = 0
) -> str:
   """Format a count with thousands separator"""
   if pd.isna(value):
       return "—"
   
   if isinstance(value, float) and value.is_integer():
       value = int(value)
   
   if isinstance(value, int):
       return f"{value:,}"
   
   return f"{value:,.{decimals}f}"

# ==================== EXPORT PUBLIC API ====================

__all__ = [
   '_to_list',
   'clean_and_standardize_df',
   'create_multiselect_filter',
   'filter_dataframe',
   'apply_multiple_filters',
   'derive_age_categories',
   'check_date_range_validity',
   'suggest_valid_date_range',
   'format_percentage',
   'format_count',
   'categorize_age',
   'AGE_BOUNDARIES',
   'AGE_LABELS'
]