"""
Permanent Housing Destinations Customization
"""

import pandas as pd
import streamlit as st


def should_apply_custom_destinations() -> bool:
    """
    Check if custom destinations should be applied based on current
    analysis module.
    """
    # Only apply in General Analysis module
    selected_module = st.session_state.get("selected_module", "")
    return "General Comprehensive Dashboard" in selected_module


def apply_custom_ph_destinations(
    df: pd.DataFrame, force: bool = False
) -> pd.DataFrame:
    """
    Apply custom PH destination selections if user has configured them.
    This modifies ExitDestinationCat based on user selections.
    This function is idempotent - can be called multiple times safely.

    Parameters:
        df: DataFrame to process
        force: If True, apply regardless of module (for explicit
               General Analysis calls)
    """
    # Check if we should apply custom destinations
    if not force and not should_apply_custom_destinations():
        return df

    if (
        "ExitDestination" not in df.columns
        or "ExitDestinationCat" not in df.columns
    ):
        return df

    # Check if we need to do anything
    if "selected_ph_destinations" not in st.session_state:
        # No custom selections, return as-is if no custom flag
        if "_custom_ph_destinations" not in df.columns:
            return df
        elif not df["_custom_ph_destinations"].any():
            return df

    # Only create copy if we need to modify
    df = df.copy()

    # If we have an original category backup, restore it first
    if "ExitDestinationCat_Original" in df.columns:
        df["ExitDestinationCat"] = df["ExitDestinationCat_Original"].copy()
    else:
        # First time - backup the original categories
        df["ExitDestinationCat_Original"] = df["ExitDestinationCat"].copy()

    # Check if user has custom selections in session state
    if "selected_ph_destinations" in st.session_state:
        selected_destinations = st.session_state.get(
            "selected_ph_destinations", set()
        )

        # Validate selections exist in current data
        available_destinations = set(df["ExitDestination"].dropna().unique())
        selected_destinations = selected_destinations.intersection(
            available_destinations
        )

        # Update session state with validated selections
        if selected_destinations != st.session_state.get(
            "selected_ph_destinations", set()
        ):
            st.session_state["selected_ph_destinations"] = (
                selected_destinations
            )

        # Get original PH destinations for comparison
        ph_mask = (
            df["ExitDestinationCat_Original"] == "Permanent Housing Situations"
        )
        original_ph_destinations = set(
            df.loc[ph_mask, "ExitDestination"].dropna().unique()
        )

        # Only modify if selections differ from original
        if selected_destinations != original_ph_destinations:
            # Mark destinations that should be PH based on user selection
            mask_should_be_ph = df["ExitDestination"].isin(
                selected_destinations
            )

            # Mark destinations that were originally PH but user excluded
            mask_was_ph_not_selected = (
                df["ExitDestinationCat_Original"]
                == "Permanent Housing Situations"
            ) & (~df["ExitDestination"].isin(selected_destinations))

            # Update categories
            df.loc[mask_should_be_ph, "ExitDestinationCat"] = (
                "Permanent Housing Situations"
            )
            df.loc[mask_was_ph_not_selected, "ExitDestinationCat"] = "Other"

            # Add a flag to indicate custom destinations are in use
            df["_custom_ph_destinations"] = True
        else:
            # Using original destinations
            df["_custom_ph_destinations"] = False
    else:
        # No custom selections, ensure flag is False
        if "_custom_ph_destinations" in df.columns:
            df["_custom_ph_destinations"] = False

    return df
