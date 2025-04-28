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
    days_lookback: int,
    allowed_cocs: list[str] | None,
    allowed_localcocs: list[str] | None,
    allowed_programs: list[str] | None,
    allowed_agencies: list[str] | None,
    entry_project_types: list[str] | None,
    allowed_cocs_exit: list[str] | None,
    allowed_localcocs_exit: list[str] | None,
    allowed_programs_exit: list[str] | None,
    allowed_agencies_exit: list[str] | None,
    exit_project_types: list[str] | None
) -> pd.DataFrame:
    """
    Perform inbound recidivism analysis with entry and exit filtering and a lookback threshold.

    Steps:
    1. Filter entries by CoC, LocalCoC, Agency, Program, and Project-Type; keep first entry per client 
       within report_start → report_end.
    2. Filter exits by CoC, LocalCoC, Agency, Program, and Project-Type; drop missing exits.
    3. For each entry, find the most recent exit within days_lookback days before the entry:
       - Prefer exits where ExitDestinationCat == "Permanent Housing Situations".
       - Label as "Returning From Housing (...)" or "Returning (...)".
       - If none, label as "New" and set exit columns to None.
    4. Prefix entry columns with "Enter_" and exit columns with "Exit_".
    5. Compute days_since_last_exit.

    Returns
    -------
    pd.DataFrame
        One row per entry, with Enter_*/Exit_* fields, ReturnCategory, and days_since_last_exit.
    """
    try:
        # --- Entry-side Filtering & Selection ---
        raw = df.copy()
        if allowed_cocs and "ProgramSetupCoC" in raw.columns:
            raw = raw[raw["ProgramSetupCoC"].isin(allowed_cocs)]
        if allowed_localcocs and "LocalCoCCode" in raw.columns:
            raw = raw[raw["LocalCoCCode"].isin(allowed_localcocs)]
        if allowed_programs and "ProgramName" in raw.columns:
            raw = raw[raw["ProgramName"].isin(allowed_programs)]
        if allowed_agencies and "AgencyName" in raw.columns:
            raw = raw[raw["AgencyName"].isin(allowed_agencies)]
        if entry_project_types and "ProjectTypeCode" in raw.columns:
            raw = raw[raw["ProjectTypeCode"].isin(entry_project_types)]

        entries = raw.dropna(subset=["ProjectStart"])
        entries = entries[
            (entries["ProjectStart"] >= report_start) &
            (entries["ProjectStart"] <= report_end)
        ]
        entries = (
            entries
            .sort_values("ProjectStart")
            .drop_duplicates("ClientID", keep="first")
            .copy()
        )

        # --- Exit-side Filtering ---
        exits = df.copy()
        if allowed_cocs_exit and "ProgramSetupCoC" in exits.columns:
            exits = exits[exits["ProgramSetupCoC"].isin(allowed_cocs_exit)]
        if allowed_localcocs_exit and "LocalCoCCode" in exits.columns:
            exits = exits[exits["LocalCoCCode"].isin(allowed_localcocs_exit)]
        if allowed_programs_exit and "ProgramName" in exits.columns:
            exits = exits[exits["ProgramName"].isin(allowed_programs_exit)]
        if allowed_agencies_exit and "AgencyName" in exits.columns:
            exits = exits[exits["AgencyName"].isin(allowed_agencies_exit)]
        if exit_project_types and "ProjectTypeCode" in exits.columns:
            exits = exits[exits["ProjectTypeCode"].isin(exit_project_types)]
        exits = exits.dropna(subset=["ProjectExit"])

        threshold = days_lookback

        def get_exit_record(row: pd.Series) -> pd.Series:
            client = row["ClientID"]
            start = row["ProjectStart"]
            subset = exits[
                (exits["ClientID"] == client) &
                (exits["ProjectExit"] < start)
            ]
            subset = subset[subset["ProjectExit"].apply(lambda x: (start - x).days <= threshold)]
            if subset.empty:
                none_s = pd.Series({f"Exit_{c}": None for c in exits.columns})
                none_s["ReturnCategory"] = "New"
                return none_s

            stable = subset[subset["ExitDestinationCat"] == "Permanent Housing Situations"]
            if not stable.empty:
                choice = stable.loc[stable["ProjectExit"].idxmax()]
                cat = f"Returning From Housing ({days_lookback} Days Lookback)"
            else:
                choice = subset.loc[subset["ProjectExit"].idxmax()]
                cat = f"Returning ({days_lookback} Days Lookback)"

            info = choice.add_prefix("Exit_")
            info["ReturnCategory"] = cat
            return info

        exit_info = entries.apply(get_exit_record, axis=1).reset_index(drop=True)
        entry_info = entries.add_prefix("Enter_").reset_index(drop=True)

        merged = pd.concat([entry_info, exit_info], axis=1)
        merged["days_since_last_exit"] = merged.apply(
            lambda r: (r["Enter_ProjectStart"] - r["Exit_ProjectExit"]).days
            if pd.notna(r["Exit_ProjectExit"]) else None,
            axis=1
        )

        return merged

    except Exception as e:
        st.error(f"Error during Return Analysis: {e}")
        return pd.DataFrame()


def display_return_metrics_cards(final_df: pd.DataFrame):
    from styling import style_metric_cards

    total = len(final_df)

    # "New" is straightforward
    new_count = (final_df["ReturnCategory"] == "New").sum()

    # Specifically count rows labeled as "Returning From Housing (X Days Lookback)"
    returning_housing_count = final_df["ReturnCategory"].str.contains("Returning From Housing").sum()

    # Count rows that say "Returning (X Days Lookback)" but NOT "Returning From Housing"
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
        ret = (subdf["ReturnCategory"].str.contains("Returning") & ~subdf["ReturnCategory"].str.contains("From Housing")).sum()
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
    Build a Sankey diagram for inbound recidivism (Exit → Entry) with detailed labels.

    Parameters:
        pivot_df (DataFrame): Flow pivot table (rows: Exit categories, columns: Entry categories).
        title (str): Diagram title.

    Returns:
        go.Figure: Plotly Sankey figure.
    """
    import plotly.graph_objects as go

    df = pivot_df.copy()
    exit_cats = df.index.tolist()     # Left side nodes (Exits)
    entry_cats = df.columns.tolist()  # Right side nodes (Entries)

    nodes = exit_cats + entry_cats
    n_exit = len(exit_cats)
    
    # Node types for hover clarity
    node_types = ["Exits"] * n_exit + ["Entries"] * len(entry_cats)

    sources, targets, values = [], [], []
    for i, exit_cat in enumerate(exit_cats):
        for j, entry_cat in enumerate(entry_cats):
            val = df.loc[exit_cat, entry_cat]
            if val > 0:
                sources.append(i)
                targets.append(n_exit + j)
                values.append(val)

    sankey = go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="#FFFFFF", width=0.5),
            label=nodes,
            color="#1f77b4",
            customdata=node_types,
            hovertemplate='%{label}<br>%{customdata}: %{value}<extra></extra>'
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color="rgba(255,127,14,0.6)",
            hovertemplate='From: %{source.label}<br>To: %{target.label}<br>Count: %{value}<extra></extra>'
        ),
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

