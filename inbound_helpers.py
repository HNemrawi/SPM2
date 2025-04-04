"""
Inbound Recidivism Helpers
--------------------------
All supporting functions for inbound recidivism logic: classification, grouping, plotting.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

def bucket_return_period(days):
    """
    Bucket the number of days into standard return period categories.
    """
    if pd.isna(days):
        return "New"
    if days <= 180:
        return "Return < 6 Months"
    elif days <= 365:
        return "Return 6–12 Months"
    elif days <= 730:
        return "Return 12–24 Months"
    else:
        return "Return > 24 Months"

@st.cache_data(show_spinner=False)
def run_return_analysis(
    df: pd.DataFrame,
    report_start: pd.Timestamp,
    report_end: pd.Timestamp,
    months_lookback: int,
    allowed_cocs,
    allowed_localcocs,
    allowed_programs,
    allowed_agencies,
    entry_project_types,
    exit_project_types
) -> pd.DataFrame:
    """
    Perform inbound recidivism analysis with exit enrollment merge and lookback threshold.
    
    Steps:
      1. **New Entry Filtering (Keep All Columns):**
         Filter the dataset based on selected CoC codes, local CoC codes, program names, agencies,
         and entry project types. Then select new entries with ProjectStart within the reporting period.
         For clients with multiple entries, only the earliest is kept.
      
      2. **Exit Event Lookup & Merge (Within Lookback Period):**
         For each new entry, look up exit events (filtered by exit project types) from the complete
         client history where ProjectExit occurs before the entry date and within the lookback period.
         Then:
           - If there is an exit with "ExitDestinationCat" equal to "Permanent Housing Situations",
             choose the most recent one.
           - Otherwise, choose the most recent exit event.
         The full exit enrollment record (including ClientID and EnrollmentID) is returned with a new
         column "ReturnCategory" that marks the entry as:
             - "Returning From Housing (X Months Lookback)" or
             - "Returning (X Months Lookback)"
         If no exit record is found within the lookback period, the exit columns are set to None and the
         category is "New".
         
         **Note:** After merging, all entry columns are prefixed with "Enter_" and exit columns retain the "Exit_" prefix.
         Additionally, the days difference between entry and exit is computed.
    
    Returns:
        pd.DataFrame: One row per client containing all entry enrollment info (prefixed with "Enter_")
                      and corresponding exit enrollment info (prefixed with "Exit_"), along with computed metrics.
    """
    try:
        # --- New Entry Selection (Keep All Columns) ---
        raw_df = df.copy()
        if allowed_cocs is not None and len(allowed_cocs) > 0 and "ProgramSetupCoC" in raw_df.columns:
            raw_df = raw_df[raw_df["ProgramSetupCoC"].isin(allowed_cocs)]
        if allowed_localcocs is not None and len(allowed_localcocs) > 0 and "LocalCoCCode" in raw_df.columns:
            raw_df = raw_df[raw_df["LocalCoCCode"].isin(allowed_localcocs)]
        if allowed_programs is not None and len(allowed_programs) > 0 and "ProgramName" in raw_df.columns:
            raw_df = raw_df[raw_df["ProgramName"].isin(allowed_programs)]
        if allowed_agencies is not None and len(allowed_agencies) > 0 and "AgencyName" in raw_df.columns:
            raw_df = raw_df[raw_df["AgencyName"].isin(allowed_agencies)]
        if entry_project_types is not None and len(entry_project_types) > 0 and "ProjectTypeCode" in raw_df.columns:
            raw_df = raw_df[raw_df["ProjectTypeCode"].isin(entry_project_types)]
        
        # Filter for entries with valid ProjectStart within the reporting period.
        entry_df = raw_df.dropna(subset=["ProjectStart"])
        entry_df = entry_df[(entry_df["ProjectStart"] >= report_start) & (entry_df["ProjectStart"] <= report_end)]
        # For clients with multiple entries, keep the earliest entry.
        entry_df = entry_df.sort_values(by="ProjectStart", ascending=True).drop_duplicates(subset="ClientID", keep="first")
        entry_df = entry_df.copy()
        
        # --- Exit Event Filtering ---
        exit_df = df.copy()
        if exit_project_types is not None and len(exit_project_types) > 0 and "ProjectTypeCode" in exit_df.columns:
            exit_df = exit_df[exit_df["ProjectTypeCode"].isin(exit_project_types)]
        exit_df = exit_df.dropna(subset=["ProjectExit"])
        
        # Define the lookback threshold in days.
        threshold_days = months_lookback * 30.4167
        
        # --- Define function to fetch the relevant exit record for each entry ---
        def get_exit_record(row):
            # Use the original entry row (without prefixing yet)
            client_id = row["ClientID"]
            entry_date = row["ProjectStart"]
            # Filter exit records for this client that occurred before the entry date.
            sub = exit_df[(exit_df["ClientID"] == client_id) & (exit_df["ProjectExit"] < entry_date)]
            # Further restrict to exits within the lookback period.
            sub = sub[sub["ProjectExit"].apply(lambda x: (entry_date - x).days <= threshold_days)]
            if sub.empty:
                # No valid exit within lookback: return a Series with exit columns as None.
                exit_info = pd.Series({f"Exit_{col}": None for col in exit_df.columns})
                exit_info["ReturnCategory"] = "New"
                return exit_info
            # Try to find a stable exit (Permanent Housing Situations) within lookback.
            stable = sub[sub["ExitDestinationCat"] == "Permanent Housing Situations"]
            if not stable.empty:
                best_row = stable.loc[stable["ProjectExit"].idxmax()]
                category = f"Returning From Housing ({months_lookback} Months Lookback)"
            else:
                best_row = sub.loc[sub["ProjectExit"].idxmax()]
                category = f"Returning ({months_lookback} Months Lookback)"
            exit_info = best_row.add_prefix("Exit_")
            exit_info["ReturnCategory"] = category
            return exit_info
        
        # Apply exit lookup for each entry record.
        exit_records = entry_df.apply(get_exit_record, axis=1)
        
        # Rename all entry columns with "Enter_" prefix.
        entry_df_renamed = entry_df.copy().add_prefix("Enter_")
        
        # Merge exit info with renamed entry data.
        merged_df = pd.concat([entry_df_renamed.reset_index(drop=True), exit_records.reset_index(drop=True)], axis=1)
        
        # Compute days_since_last_exit as the difference between Enter_ProjectStart and Exit_ProjectExit.
        merged_df["days_since_last_exit"] = merged_df.apply(
            lambda row: (row["Enter_ProjectStart"] - row["Exit_ProjectExit"]).days 
                        if pd.notnull(row["Exit_ProjectExit"]) else None, axis=1
        )
        
        return merged_df

    except Exception as e:
        st.error(f"Error during Return Analysis: {e}")
        return pd.DataFrame()


def display_return_metrics_cards(final_df: pd.DataFrame):
    from styling import style_metric_cards

    total = len(final_df)

    # "New" is straightforward
    new_count = (final_df["ReturnCategory"] == "New").sum()

    # Specifically count rows labeled as "Returning From Housing (X Months Lookback)"
    returning_housing_count = final_df["ReturnCategory"].str.contains("Returning From Housing").sum()

    # Count rows that say "Returning (X Months Lookback)" but NOT "Returning From Housing"
    returning_only_mask = (
        final_df["ReturnCategory"].str.contains("Returning") &
        ~final_df["ReturnCategory"].str.contains("From Housing")
    )
    returning_count = returning_only_mask.sum()

    # Percentages
    pct_new = (new_count / total * 100) if total else 0
    pct_returning = (returning_count / total * 100) if total else 0
    pct_returning_housing = (returning_housing_count / total * 100) if total else 0

    # Apply any styling
    style_metric_cards()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Entries", f"{total:,}")
    col2.metric("New (%)", f"{new_count:,} ({pct_new:.1f}%)")
    col3.metric("Returning (%)", f"{returning_count:,} ({pct_returning:.1f}%)")
    col4.metric("Returning From Housing (%)", f"{returning_housing_count:,} ({pct_returning_housing:.1f}%)")

@st.cache_data(show_spinner=False)
def return_breakdown_analysis(df: pd.DataFrame, group_cols: list):
    """
    Group inbound recidivism data by selected columns and compute summary metrics.
    
    Returns:
        pd.DataFrame: Aggregated results sorted by total entries.
    """
    grouped = df.groupby(group_cols)
    rows = []
    for group_vals, subdf in grouped:
        group_vals = group_vals if isinstance(group_vals, tuple) else (group_vals,)
        row_data = dict(zip(group_cols, group_vals))
        total = len(subdf)
        new_count = (subdf["ReturnCategory"] == "New").sum()
        ret = subdf["ReturnCategory"].str.contains("Returning").sum()
        ret_housing = subdf["ReturnCategory"].str.contains("Returning From Housing").sum()

        valid_time = subdf.dropna(subset=["days_since_last_exit"])
        median_days = valid_time["days_since_last_exit"].median() if not valid_time.empty else 0
        avg_days = valid_time["days_since_last_exit"].mean() if not valid_time.empty else 0

        row_data["Total Entries"] = total
        row_data["New (%)"] = f"{new_count} ({(new_count/total*100 if total else 0):.1f}%)"
        row_data["Returning (%)"] = f"{ret} ({(ret/total*100 if total else 0):.1f}%)"
        row_data["Returning From Housing (%)"] = f"{ret_housing} ({(ret_housing/total*100 if total else 0):.1f}%)"
        row_data["Median Days"] = f"{median_days:.1f}"
        row_data["Average Days"] = f"{avg_days:.1f}"
        rows.append(row_data)
    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values("Total Entries", ascending=False)
    return out_df

def create_flow_pivot_ra(final_df: pd.DataFrame, source_col: str, target_col: str) -> pd.DataFrame:
    """
    Create a pivot table for inbound recidivism flow analysis (Exit → Entry).
    
    Parameters:
        final_df (DataFrame): DataFrame with inbound recidivism data.
        source_col (str): Column representing the exit category.
        target_col (str): Column representing the entry category.
    
    Returns:
        pd.DataFrame: Crosstab pivot table.
    """
    df_copy = final_df.copy()
    df_copy[target_col] = df_copy[target_col].fillna("No Data").astype(str)
    pivot = pd.crosstab(df_copy[source_col], df_copy[target_col], margins=False, dropna=False)
    return pivot

def plot_flow_sankey_ra(pivot_df: pd.DataFrame, title: str = "Exit → Entry Sankey") -> go.Figure:
    """
    Build a Sankey diagram for inbound recidivism (Exit → Entry).
    
    Parameters:
        pivot_df (DataFrame): Flow pivot table.
        title (str): Diagram title.
    
    Returns:
        go.Figure: Plotly figure.
    """
    df = pivot_df.copy()
    exit_cats = df.index.tolist()
    entry_cats = df.columns.tolist()
    nodes = exit_cats + entry_cats
    sources, targets, values = [], [], []
    for i, ecat in enumerate(exit_cats):
        for j, rcat in enumerate(entry_cats):
            val = df.loc[ecat, rcat]
            if val > 0:
                sources.append(i)
                targets.append(len(exit_cats) + j)
                values.append(val)
    sankey = go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="#FFFFFF", width=0.5),
            label=nodes,
            color="#1f77b4"
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color="rgba(255,127,14,0.6)"
        )
    )
    fig = go.Figure(data=[sankey])
    fig.update_layout(
        title_text=title,
        font=dict(family="Open Sans", color="#FFFFFF"),
        hovermode="x",
        template="plotly_dark"
    )
    return fig

def plot_time_to_entry_box(final_df: pd.DataFrame) -> go.Figure:
    """
    Create a box plot for the Time-to-Entry distribution (days between exit and new entry).
    
    Parameters:
        final_df (DataFrame): Inbound recidivism DataFrame.
    
    Returns:
        go.Figure: Plotly box plot figure.
    """
    returned_df = final_df[final_df["ReturnCategory"] != "New"].dropna(subset=["days_since_last_exit"])
    if returned_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No Return Entries Found",
            xaxis_title="Time to Entry (Days)",
            template="plotly_dark"
        )
        return fig

    x = returned_df["days_since_last_exit"].dropna()
    median_val = x.median()
    avg_val = x.mean()
    fig = go.Figure()
    fig.add_trace(go.Box(
        x=x,
        name="Time to Entry",
        boxmean='sd'
    ))
    fig.update_layout(
        title="Time to Entry Distribution (Box Plot)",
        template="plotly_dark",
        xaxis_title="Days to Entry",
        showlegend=False,
        shapes=[
            dict(
                type="line",
                xref="x",
                x0=median_val, x1=median_val,
                yref="paper", y0=0, y1=1,
                line=dict(color="yellow", width=2, dash="dot"),
            ),
            dict(
                type="line",
                xref="x",
                x0=avg_val, x1=avg_val,
                yref="paper", y0=0, y1=1,
                line=dict(color="orange", width=2, dash="dash"),
            ),
        ]
    )
    return fig

