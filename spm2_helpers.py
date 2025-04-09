"""
Helper Functions & Metrics (SPM2 & Visualizations)
--------------------------------------------------
Core logic for SPM2 analysis, summary metrics, grouping, and plotting.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# -------------------------------------------------------------------------
# Internal Helper for SPM2
# -------------------------------------------------------------------------
def _find_earliest_return(client_df, earliest_exit, cutoff, report_end, exit_enrollment_id):
    """
    Identify the earliest valid return enrollment after an exit enrollment.
    
    For PH projects, ensure at least a 14-day gap and consider exclusion windows.
    
    Parameters:
        client_df (DataFrame): Client's enrollment records.
        earliest_exit (Timestamp): The exit date.
        cutoff (Timestamp): Maximum ProjectStart date to consider.
        report_end (Timestamp): End date of the report.
        exit_enrollment_id (numeric): EnrollmentID to exclude from scanning.
    
    Returns:
        tuple: (row, "PH" or "Non-PH") for a valid return enrollment, or None if not found.
    """
    if client_df.empty or "ProjectStart" not in client_df.columns:
        return None

    ph_projects = {
        "PH – Housing Only",
        "PH – Housing with Services (no disability required for entry)",
        "PH – Permanent Supportive Housing (disability required for entry)",
        "PH – Rapid Re-Housing"
    }
    
    client_df = client_df.dropna(subset=["ProjectStart"]).copy().sort_values(["ProjectStart", "EnrollmentID"])
    exclusion_windows = []

    def add_or_extend_window(new_window, windows):
        new_start, new_end = new_window
        updated_windows = []
        for window in windows:
            win_start, win_end = window
            if new_start <= win_end and new_end >= win_start:
                new_start = min(new_start, win_start)
                new_end = max(new_end, win_end)
            else:
                updated_windows.append(window)
        updated_windows.append((new_start, new_end))
        updated_windows.sort(key=lambda x: x[0])
        return updated_windows

    for row in client_df.itertuples(index=False):
        if getattr(row, "EnrollmentID", None) is not None and row.EnrollmentID == exit_enrollment_id:
            continue
        if row.ProjectStart < earliest_exit:
            if row.ProjectTypeCode in ph_projects:
                if pd.notnull(row.ProjectExit) and row.ProjectExit > earliest_exit:
                    new_window = (
                        earliest_exit + pd.Timedelta(days=1),
                        min(row.ProjectExit + pd.Timedelta(days=14), report_end)
                    )
                    exclusion_windows = add_or_extend_window(new_window, exclusion_windows)
            continue
        if row.ProjectStart > cutoff:
            break
        if row.ProjectTypeCode not in ph_projects:
            return (row, "Non-PH")
        else:
            gap = (row.ProjectStart - earliest_exit).days
            new_window = (
                row.ProjectStart + pd.Timedelta(days=1),
                min((row.ProjectExit + pd.Timedelta(days=14)) if pd.notnull(row.ProjectExit) else report_end, report_end)
            )
            if gap <= 14:
                exclusion_windows = add_or_extend_window(new_window, exclusion_windows)
                continue
            else:
                if any(win_start <= row.ProjectStart <= win_end for win_start, win_end in exclusion_windows):
                    exclusion_windows = add_or_extend_window(new_window, exclusion_windows)
                    continue
                return (row, "PH")
    return None


# -------------------------------------------------------------------------
# Main SPM2 Logic
# -------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def run_spm2(
    df: pd.DataFrame,
    report_start: pd.Timestamp,
    report_end: pd.Timestamp,
    months_lookback: int = 24,
    exit_cocs: list = None,
    exit_localcocs: list = None,
    exit_agencies: list = None,
    exit_programs: list = None,
    return_cocs: list = None,
    return_localcocs: list = None,
    return_agencies: list = None,
    return_programs: list = None,
    allowed_continuum: list = None,
    allowed_exit_dest_cats: list = None,
    exiting_projects: list = None,
    return_projects: list = None,
    return_period: int = 730
) -> pd.DataFrame:
    """
    Main logic for SPM2 analysis.
    
    1. Filter and subset data for exit and return based on user-selected filters.
    2. Identify valid exit enrollments within the derived lookback window.
    3. Scan for the earliest valid return enrollment.
    4. Classify returns by the gap in days and calculate additional metrics.
    
    Returns:
        pd.DataFrame: Merged exit and return enrollment records with computed metrics.
    """
    required_cols = ["ProjectStart", "ProjectExit", "ProjectTypeCode"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Error during SPM2 processing: Missing required columns: {missing_cols}")
        return pd.DataFrame()

    if not exiting_projects:
        st.error("Error during SPM2 processing: No project types selected for exit identification.")
        return pd.DataFrame()
    if not return_projects:
        st.error("Error during SPM2 processing: No project types selected for return identification.")
        return pd.DataFrame()

    try:
        # Subset for EXIT filtering.
        df_exit_base = df.copy()
        if exit_cocs and "ProgramSetupCoC" in df_exit_base.columns:
            df_exit_base = df_exit_base[df_exit_base["ProgramSetupCoC"].isin(exit_cocs)]
        if exit_localcocs and "LocalCoCCode" in df_exit_base.columns:
            df_exit_base = df_exit_base[df_exit_base["LocalCoCCode"].isin(exit_localcocs)]
        if exit_agencies and "AgencyName" in df_exit_base.columns:
            df_exit_base = df_exit_base[df_exit_base["AgencyName"].isin(exit_agencies)]
        if exit_programs and "ProgramName" in df_exit_base.columns:
            df_exit_base = df_exit_base[df_exit_base["ProgramName"].isin(exit_programs)]

        # Subset for RETURN filtering.
        df_return_base = df.copy()
        if return_cocs and "ProgramSetupCoC" in df_return_base.columns:
            df_return_base = df_return_base[df_return_base["ProgramSetupCoC"].isin(return_cocs)]
        if return_localcocs and "LocalCoCCode" in df_return_base.columns:
            df_return_base = df_return_base[df_return_base["LocalCoCCode"].isin(return_localcocs)]
        if return_agencies and "AgencyName" in df_return_base.columns:
            df_return_base = df_return_base[df_return_base["AgencyName"].isin(return_agencies)]
        if return_programs and "ProgramName" in df_return_base.columns:
            df_return_base = df_return_base[df_return_base["ProgramName"].isin(return_programs)]

        if allowed_continuum and "Programs Continuum Project" in df.columns:
            df_exit_base = df_exit_base[df_exit_base["Programs Continuum Project"].isin(allowed_continuum)]
            df_return_base = df_return_base[df_return_base["Programs Continuum Project"].isin(allowed_continuum)]

        if allowed_exit_dest_cats is None and "ExitDestinationCat" in df_exit_base.columns:
            allowed_exit_dest_cats = df_exit_base["ExitDestinationCat"].dropna().unique().tolist()

        # Determine exit analysis window.
        exit_min = report_start - pd.DateOffset(months=months_lookback)
        exit_max = report_end - pd.DateOffset(months=months_lookback)

        # Identify valid exit enrollments.
        df_exits = df_exit_base.dropna(subset=["ProjectExit"]).copy()
        df_exits = df_exits[df_exits["ProjectTypeCode"].isin(exiting_projects)]
        df_exits = df_exits[(df_exits["ProjectExit"] >= exit_min) & (df_exits["ProjectExit"] <= exit_max)]

        if allowed_exit_dest_cats and "ExitDestinationCat" in df_exits.columns:
            df_exits = df_exits[df_exits["ExitDestinationCat"].isin(allowed_exit_dest_cats)]

        df_exits = df_exits.sort_values(["ClientID", "ProjectExit", "EnrollmentID"])
        df_exits = df_exits.drop_duplicates(subset=["ClientID"], keep="first")

        # Prepare data for scanning returns.
        required_return_cols = df_return_base.columns.tolist()
        df_for_scan = df_return_base[required_return_cols].copy()
        df_for_scan = df_for_scan[df_for_scan["ProjectTypeCode"].isin(return_projects)]
        grouped_returns = df_for_scan.groupby("ClientID")

        exit_rows = []
        for cid, group_ex in df_exits.groupby("ClientID"):
            group_ex = group_ex.sort_values(["ProjectExit", "EnrollmentID"])
            group_ret = grouped_returns.get_group(cid) if cid in grouped_returns.groups else pd.DataFrame()
            for row in group_ex.itertuples(index=False):
                row_dict = {f"Exit_{c}": getattr(row, c, None) for c in group_ex.columns}
                earliest_exit_date = getattr(row, "ProjectExit", pd.NaT)
                if pd.isna(earliest_exit_date):
                    row_dict["DaysToReturn"] = pd.NA
                    row_dict["ReturnCategory"] = "No Return"
                    row_dict["ReturnedToHomelessness"] = False
                    exit_rows.append(row_dict)
                    continue

                cutoff_date = min(earliest_exit_date + pd.Timedelta(days=return_period), report_end)
                found_result = _find_earliest_return(
                    group_ret,
                    earliest_exit_date,
                    cutoff_date,
                    report_end,
                    exit_enrollment_id=row.EnrollmentID
                )
                if found_result is not None:
                    found, _ = found_result
                    for cc in df_for_scan.columns:
                        row_dict[f"Return_{cc}"] = getattr(found, cc, None)
                    row_dict["DaysToReturn"] = (found.ProjectStart - earliest_exit_date).days
                else:
                    for cc in df_for_scan.columns:
                        row_dict[f"Return_{cc}"] = None
                    row_dict["DaysToReturn"] = pd.NA

                if pd.isna(row_dict["DaysToReturn"]):
                    row_dict["ReturnCategory"] = "No Return"
                else:
                    days = row_dict["DaysToReturn"]
                    if days <= 180:
                        row_dict["ReturnCategory"] = "Return < 6 Months"
                    elif days <= 365:
                        row_dict["ReturnCategory"] = "Return 6–12 Months"
                    elif days <= 730:
                        row_dict["ReturnCategory"] = "Return 12–24 Months"
                    else:
                        row_dict["ReturnCategory"] = "Return > 24 Months"

                row_dict["ReturnedToHomelessness"] = row_dict["ReturnCategory"] in [
                    "Return < 6 Months",
                    "Return 6–12 Months",
                    "Return 12–24 Months",
                    "Return > 24 Months"
                ]
                exit_rows.append(row_dict)

        final_df = pd.DataFrame(exit_rows)

        if "Exit_ExitDestinationCat" in final_df.columns:
            final_df["PH_Exit"] = final_df["Exit_ExitDestinationCat"].apply(
                lambda x: x == "Permanent Housing Situations"
            )
        else:
            final_df["PH_Exit"] = False

        if "Exit_DOB" in final_df.columns and "Exit_ProjectExit" in final_df.columns:
            age_days = (final_df["Exit_ProjectExit"] - final_df["Exit_DOB"]).dt.days
            age_years = age_days / 365.25

            def age_range(age):
                if pd.isna(age):
                    return "Unknown"
                if age < 18:
                    return "0 to 17"
                elif age < 25:
                    return "18 to 24"
                elif age < 35:
                    return "25 to 34"
                elif age < 45:
                    return "35 to 44"
                elif age < 55:
                    return "45 to 54"
                elif age < 65:
                    return "55 to 64"
                return "65 or Above"

            final_df["AgeAtExitRange"] = age_years.apply(age_range)
        else:
            final_df["AgeAtExitRange"] = "Unknown"

        return final_df

    except Exception as e:
        st.error(f"Error during SPM2 processing: {e}")
        return pd.DataFrame()


# -------------------------------------------------------------------------
# Compute Summary Metrics for SPM2
# -------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def compute_summary_metrics(final_df: pd.DataFrame, return_period: int = 730) -> dict:
    """
    Compute key summary metrics for SPM2 analysis.
    
    Returns:
        dict: A dictionary with counts, percentages, and day statistics.
    """
    total_exited = len(final_df)
    ph_exits = final_df["PH_Exit"].sum() if "PH_Exit" in final_df.columns else 0

    cat_counts = final_df["ReturnCategory"].value_counts(dropna=False)
    r6 = cat_counts.get("Return < 6 Months", 0)
    r6_12 = cat_counts.get("Return 6–12 Months", 0)
    r12_24 = cat_counts.get("Return 12–24 Months", 0)
    r_gt_24 = cat_counts.get("Return > 24 Months", 0)

    total_return = r6 + r6_12 + r12_24 + r_gt_24
    pct_return = (total_return / total_exited * 100) if total_exited else 0
    pct_r6 = (r6 / total_exited * 100) if total_exited else 0
    pct_r6_12 = (r6_12 / total_exited * 100) if total_exited else 0
    pct_r12_24 = (r12_24 / total_exited * 100) if total_exited else 0
    pct_r_gt_24 = (r_gt_24 / total_exited * 100) if total_exited else 0
    pct_ph_exits = (ph_exits / total_exited * 100) if total_exited else 0

    returned_df = final_df[
        (final_df["ReturnedToHomelessness"]) &
        (final_df["DaysToReturn"].notna()) &
        (final_df["DaysToReturn"] <= return_period)
    ]
    if not returned_df.empty:
        median_days = returned_df["DaysToReturn"].median()
        avg_days = returned_df["DaysToReturn"].mean()
        p25 = returned_df["DaysToReturn"].quantile(0.25)
        p75 = returned_df["DaysToReturn"].quantile(0.75)
        max_days = returned_df["DaysToReturn"].max()
    else:
        median_days = avg_days = p25 = p75 = max_days = 0

    return {
        "Total Exits": total_exited,
        "PH Exits": ph_exits,
        "% PH Exits": pct_ph_exits,
        "Return < 6 Months": r6,
        "% Return < 6M": pct_r6,
        "Return 6–12 Months": r6_12,
        "% Return 6–12M": pct_r6_12,
        "Return 12–24 Months": r12_24,
        "% Return 12–24M": pct_r12_24,
        "Return > 24 Months": r_gt_24,
        "% Return > 24M": pct_r_gt_24,
        "Total Return": total_return,
        "% Return": pct_return,
        "Median Days (<=period)": median_days,
        "Average Days (<=period)": avg_days,
        "DaysToReturn 25th Pctl": p25,
        "DaysToReturn 75th Pctl": p75,
        "DaysToReturn Max": max_days,
    }


# -------------------------------------------------------------------------
# Breakdown Analysis
# -------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def breakdown_by_columns(final_df: pd.DataFrame, columns: list, return_period: int = 730) -> pd.DataFrame:
    """
    Group the SPM2 DataFrame by the specified columns and compute summary metrics.
    
    Returns:
        pd.DataFrame: Aggregated metrics with a combined "count (%)" format.
    """
    grouped = final_df.groupby(columns)
    rows = []
    for group_vals, subdf in grouped:
        group_vals = group_vals if isinstance(group_vals, tuple) else (group_vals,)
        row_data = {col: val for col, val in zip(columns, group_vals)}
        m = compute_summary_metrics(subdf, return_period)

        row_data["Total Exits"] = m["Total Exits"]
        row_data["Total Return"] = m["Total Return"]
        row_data["% Return"] = f"{m['% Return']:.1f}%"

        pairs = [
            ("PH Exits", "% PH Exits"), 
            ("Return < 6 Months", "% Return < 6M"), 
            ("Return 6–12 Months", "% Return 6–12M"),
            ("Return 12–24 Months", "% Return 12–24M"),
            ("Return > 24 Months", "% Return > 24M"),
        ]
        for count_col, pct_col in pairs:
            if count_col in m and pct_col in m:
                row_data[count_col] = f"{m[count_col]} ({m[pct_col]:.1f}%)"

        row_data["Median Days"] = f"{m['Median Days (<=period)']:.1f}"
        row_data["Average Days"] = f"{m['Average Days (<=period)']:.1f}"

        rows.append(row_data)

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values(by="Total Exits", key=lambda x: x.astype(int), ascending=False)
    return out_df


def get_top_flows_from_pivot(pivot_df: pd.DataFrame, top_n=10) -> pd.DataFrame:
    """
    Generate a table of the top flows from a crosstab pivot table.
    
    Parameters:
        pivot_df (DataFrame): Crosstab pivot table.
        top_n (int): Number of top flows to return.
    
    Returns:
        pd.DataFrame: Top flows with source, target, count, and percent.
    """
    df = pivot_df.copy()
    total = df.values.sum()
    rows = []
    row_labels = df.index.tolist()
    col_labels = df.columns.tolist()
    for rlab in row_labels:
        for clab in col_labels:
            val = df.loc[rlab, clab]
            rows.append({
                "Source": rlab,
                "Target": clab,
                "Count": val,
                "Percent": (val / total * 100) if total else 0
            })
    flows_df = pd.DataFrame(rows)
    flows_df = flows_df[flows_df["Count"] > 0].sort_values("Count", ascending=False)
    return flows_df.head(top_n)


@st.cache_data(show_spinner=False)
def create_flow_pivot(final_df: pd.DataFrame, exit_col: str, return_col: str) -> pd.DataFrame:
    """
    Create a crosstab pivot table for flow analysis (Exit → Return).
    
    Parameters:
        final_df (DataFrame): Final merged SPM2 DataFrame.
        exit_col (str): Column name for exit category.
        return_col (str): Column name for return category.
    
    Returns:
        pd.DataFrame: Crosstab pivot without margins.
    """
    df_copy = final_df.copy()
    df_copy[return_col] = df_copy[return_col].fillna("No Return").astype(str)
    pivot = pd.crosstab(df_copy[exit_col], df_copy[return_col], margins=False)
    return pivot


def plot_flow_sankey(pivot_df: pd.DataFrame, title: str = "Exit → Return Sankey Diagram") -> go.Figure:
    """
    Build a Sankey diagram to visualize the flow from exit to return categories.
    
    Parameters:
        pivot_df (DataFrame): Flow pivot table.
        title (str): Diagram title.
    
    Returns:
        go.Figure: Plotly figure of the Sankey diagram.
    """
    df = pivot_df.copy()
    exit_cats = df.index.tolist()
    return_cats = df.columns.tolist()
    nodes = exit_cats + return_cats
    sources, targets, values = [], [], []
    for i, ecat in enumerate(exit_cats):
        for j, rcat in enumerate(return_cats):
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
            color="#1f77b4",
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color="rgba(255,127,14,0.6)",
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


def plot_days_to_return_box(final_df: pd.DataFrame, return_period: int = 730) -> go.Figure:
    """
    Create a box plot for the distribution of Days-to-Return.
    
    Parameters:
        final_df (DataFrame): Merged SPM2 DataFrame.
        return_period (int): Maximum days to consider.
    
    Returns:
        go.Figure: Plotly box plot figure.
    """
    returned_df = final_df[
        (final_df["ReturnedToHomelessness"]) &
        (final_df["DaysToReturn"].notna()) &
        (final_df["DaysToReturn"] <= return_period)
    ]
    import plotly.graph_objects as go

    if returned_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No Returns Found",
            xaxis_title="Days to Return",
            yaxis_title="",
            template="plotly_dark"
        )
        return fig

    x = returned_df["DaysToReturn"].dropna()
    median_val = x.median()
    avg_val = x.mean()
    fig = go.Figure()
    fig.add_trace(go.Box(
        x=x,
        name="Days to Return",
        boxmean='sd'
    ))
    fig.update_layout(
        title="Days to Return Distribution (Box Plot)",
        template="plotly_dark",
        xaxis_title="Days to Return",
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


def display_spm_metrics(metrics: dict, show_total_exits: bool = True):
    """
    Display key SPM2 metrics in a card-style layout.
    
    Parameters:
        metrics (dict): Dictionary of computed metrics.
        show_total_exits (bool): Whether to display total exit count.
    """
    import streamlit as st
    from styling import style_metric_cards

    style_metric_cards()
    col1, col2, col3, col4, col5 = st.columns(5)
    if show_total_exits:
        col1.metric("Total Exits", f"{metrics['Total Exits']:,}")
    col1.metric("PH Exits", f"{metrics['PH Exits']:,} ({metrics['% PH Exits']:.1f}%)")
    col2.metric("Return <6M", f"{metrics['Return < 6 Months']:,} ({metrics['% Return < 6M']:.1f}%)")
    col3.metric("Return 6–12M", f"{metrics['Return 6–12 Months']:,} ({metrics['% Return 6–12M']:.1f}%)")
    col4.metric("Return 12–24M", f"{metrics['Return 12–24 Months']:,} ({metrics['% Return 12–24M']:.1f}%)")
    col5.metric("Return >24M", f"{metrics['Return > 24 Months']:,} ({metrics['% Return > 24M']:.1f}%)")
    
    colA, colB, colC = st.columns(3)
    colA.metric("Total Return", f"{metrics['Total Return']:,} ({metrics['% Return']:.1f}%)")
    colB.metric("Median Days", f"{metrics['Median Days (<=period)']:.1f}")
    colC.metric("Avg Days", f"{metrics['Average Days (<=period)']:.1f}")
    st.markdown(
        f"**25th/75th:** {metrics['DaysToReturn 25th Pctl']:.1f}/{metrics['DaysToReturn 75th Pctl']:.1f} | **Max Days:** {metrics['DaysToReturn Max']:.0f}"
    )
