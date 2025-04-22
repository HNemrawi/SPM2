"""
Outbound Recidivism – Helper Functions & Metrics
================================================
Core logic for the Outbound Recidivism analysis page:

    • Filtering exits and candidate returns
    • Identifying “Return” vs. “Return to Homelessness”
    • Summary‑metric calculators
    • Break‑down, flow, and visualisation helpers

Everything is **Streamlit‑cacheable** and PEP‑8‑compliant.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# -----------------------------------------------------------------------------#
# Constants                                                                     #
# -----------------------------------------------------------------------------#
PH_PROJECTS: Set[str] = {
    "PH – Housing Only",
    "PH – Housing with Services (no disability required for entry)",
    "PH – Permanent Supportive Housing (disability required for entry)",
    "PH – Rapid Re-Housing",
}

NON_HOMELESS_PROJECTS: Set[str] = {
    "Coordinated Entry",
    "Day Shelter",
    "Homelessness Prevention",
    "Other",
    "Services Only",
}

# -----------------------------------------------------------------------------#
# Internal Helpers                                                              #
# -----------------------------------------------------------------------------#
def _find_earliest_return_any(
    client_df: pd.DataFrame,
    exit_date: pd.Timestamp,
    exit_enrollment_id: int,
) -> Optional[Tuple[pd.Series, int]]:
    """
    Return the first enrollment (any project type) that begins strictly
    after *exit_date* (skipping the reference enrollment).
    """
    try:
        df_next = (
            client_df.copy()
            .dropna(subset=["ProjectStart"])
            .loc[lambda d: d["ProjectStart"] > exit_date]
            .sort_values(["ProjectStart", "EnrollmentID"])
        )
        for row in df_next.itertuples(index=False):
            if row.EnrollmentID == exit_enrollment_id:
                continue
            gap_days: int = (row.ProjectStart - exit_date).days
            return (pd.Series(row._asdict()), gap_days)
        return None
    except Exception as exc:
        st.error(f"Error in _find_earliest_return_any: {exc}")
        return None


def _find_earliest_return_homeless(
    client_df: pd.DataFrame,
    exit_date: pd.Timestamp,
    exit_enrollment_id: int,
) -> Optional[Tuple[pd.Series, bool, int]]:
    """
    HUD‑compliant “Return to Homelessness” finder.
    """
    try:
        df_next = (
            client_df.copy()
            .dropna(subset=["ProjectStart"])
            .loc[lambda d: ~d["ProjectTypeCode"].isin(NON_HOMELESS_PROJECTS)]
            .loc[lambda d: d["ProjectStart"] > exit_date]
            .sort_values(["ProjectStart", "EnrollmentID"])
        )
        if df_next.empty:
            return None

        exclusion_windows: List[Tuple[pd.Timestamp, pd.Timestamp]] = []

        def _merge_window(
            new_win: Tuple[pd.Timestamp, pd.Timestamp],
            windows: List[Tuple[pd.Timestamp, pd.Timestamp]],
        ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
            new_start, new_end = new_win
            merged: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
            for win_start, win_end in windows:
                if new_start <= win_end and new_end >= win_start:
                    new_start = min(new_start, win_start)
                    new_end = max(new_end, win_end)
                else:
                    merged.append((win_start, win_end))
            merged.append((new_start, new_end))
            merged.sort(key=lambda w: w[0])
            return merged

        for row in df_next.itertuples(index=False):
            if row.EnrollmentID == exit_enrollment_id:
                continue
            gap_days: int = (row.ProjectStart - exit_date).days
            is_ph: bool = row.ProjectTypeCode in PH_PROJECTS

            if is_ph:
                if gap_days <= 14:
                    if pd.notna(row.ProjectExit):
                        exclusion_windows = _merge_window(
                            (
                                row.ProjectStart + pd.Timedelta(days=1),
                                row.ProjectExit + pd.Timedelta(days=14),
                            ),
                            exclusion_windows,
                        )
                    continue

                inside_window = any(
                    start <= row.ProjectStart <= end for start, end in exclusion_windows
                )
                if inside_window:
                    if pd.notna(row.ProjectExit):
                        exclusion_windows = _merge_window(
                            (
                                row.ProjectStart + pd.Timedelta(days=1),
                                row.ProjectExit + pd.Timedelta(days=14),
                            ),
                            exclusion_windows,
                        )
                    continue

                return (pd.Series(row._asdict()), True, gap_days)

            # Non‑PH
            return (pd.Series(row._asdict()), True, gap_days)

        return None
    except Exception as exc:
        st.error(f"Error in _find_earliest_return_homeless: {exc}")
        return None


# -----------------------------------------------------------------------------#
# Main ETL                                                                    #
# -----------------------------------------------------------------------------#
@st.cache_data(show_spinner=False)
def run_outbound_recidivism(
    df: pd.DataFrame,
    report_start: pd.Timestamp,
    report_end: pd.Timestamp,
    *,
    exit_cocs: Optional[List[str]] = None,
    exit_localcocs: Optional[List[str]] = None,
    exit_agencies: Optional[List[str]] = None,
    exit_programs: Optional[List[str]] = None,
    return_cocs: Optional[List[str]] = None,
    return_localcocs: Optional[List[str]] = None,
    return_agencies: Optional[List[str]] = None,
    return_programs: Optional[List[str]] = None,
    allowed_continuum: Optional[List[str]] = None,
    allowed_exit_dest_cats: Optional[List[str]] = None,
    exiting_projects: Optional[List[str]] = None,
    return_projects: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    End‑to‑end ETL for Outbound Recidivism.
    """
    try:
        req_cols: Set[str] = {
            "ClientID",
            "ProjectStart",
            "ProjectExit",
            "ProjectTypeCode",
            "EnrollmentID",
        }
        missing = sorted(req_cols.difference(df.columns))
        if missing:
            st.error(f"Missing required columns: {missing}")
            return pd.DataFrame()

        exiting_projects = exiting_projects or sorted(df["ProjectTypeCode"].dropna().unique())
        return_projects = return_projects or exiting_projects
        if not exiting_projects:
            st.error("No project types selected for Exits.")
            return pd.DataFrame()

        def _opt_filter(frame: pd.DataFrame, col: str, allowed: Optional[Iterable[str]]) -> pd.DataFrame:
            return frame if not allowed or col not in frame.columns else frame[frame[col].isin(allowed)]

        # EXIT filter
        df_exit = df.copy()
        df_exit = _opt_filter(df_exit, "ProgramSetupCoC", exit_cocs)
        df_exit = _opt_filter(df_exit, "LocalCoCCode", exit_localcocs)
        df_exit = _opt_filter(df_exit, "AgencyName", exit_agencies)
        df_exit = _opt_filter(df_exit, "ProgramName", exit_programs)
        df_exit = _opt_filter(df_exit, "ProgramsContinuumProject", allowed_continuum)
        df_exit = _opt_filter(df_exit, "ExitDestinationCat", allowed_exit_dest_cats)

        df_exit = (
            df_exit.dropna(subset=["ProjectExit"])
            .loc[lambda d: d["ProjectTypeCode"].isin(exiting_projects)]
            .loc[lambda d: (d["ProjectExit"] >= report_start) & (d["ProjectExit"] <= report_end)]
        )
        if df_exit.empty:
            return pd.DataFrame()

        df_exit = (
            df_exit.sort_values(["ClientID", "ProjectExit", "EnrollmentID"])
            .groupby("ClientID", as_index=False)
            .tail(1)
        )

        # RETURN filter
        df_return = df.copy()
        df_return = _opt_filter(df_return, "ProgramSetupCoC", return_cocs)
        df_return = _opt_filter(df_return, "LocalCoCCode", return_localcocs)
        df_return = _opt_filter(df_return, "AgencyName", return_agencies)
        df_return = _opt_filter(df_return, "ProgramName", return_programs)
        df_return = _opt_filter(df_return, "ProgramsContinuumProject", allowed_continuum)
        df_return = df_return.loc[df_return["ProjectTypeCode"].isin(return_projects)]

        grouped_return = df_return.groupby("ClientID")

        rows: List[Dict[str, Any]] = []
        for exit_row in df_exit.itertuples(index=False):
            cid = exit_row.ClientID
            exit_date = exit_row.ProjectExit
            exit_enroll_id = exit_row.EnrollmentID

            row_dict = {f"Exit_{col}": getattr(exit_row, col, pd.NA) for col in df_exit.columns}
            for col in df_return.columns:
                row_dict[f"Return_{col}"] = pd.NA

            row_dict.update({
                "HasReturn": False,
                "ReturnToHomelessness": False,
                "DaysToReturnEnrollment": pd.NA,
            })

            if cid not in grouped_return.groups:
                rows.append(row_dict)
                continue

            client_enrolls = grouped_return.get_group(cid)

            # Any Return
            result_any = _find_earliest_return_any(client_enrolls, exit_date, exit_enroll_id)
            if result_any:
                next_row_any, gap_any = result_any
                row_dict["HasReturn"] = True
                row_dict["DaysToReturnEnrollment"] = gap_any
                for col in df_return.columns:
                    row_dict[f"Return_{col}"] = next_row_any.get(col, pd.NA)

            # Return to Homelessness
            result_rth = _find_earliest_return_homeless(client_enrolls, exit_date, exit_enroll_id)
            if result_rth:
                _, is_return, _ = result_rth
                row_dict["ReturnToHomelessness"] = is_return

            rows.append(row_dict)

        final_df = pd.DataFrame(rows)
        final_df["PH_Exit"] = final_df["Exit_ExitDestinationCat"].eq("Permanent Housing Situations")

        # Age buckets
        if {"Exit_DOB", "Exit_ProjectExit"}.issubset(final_df.columns):
            age_years = (final_df["Exit_ProjectExit"] - final_df["Exit_DOB"]).dt.days / 365.25
            def _bucket(age):
                if pd.isna(age):
                    return "Unknown"
                for bound, label in zip(
                    [18, 25, 35, 45, 55, 65],
                    ["0–17", "18–24", "25–34", "35–44", "45–54", "55–64"],
                ):
                    if age < bound:
                        return label
                return "65+"
            final_df["AgeAtExitRange"] = age_years.apply(_bucket)
        else:
            final_df["AgeAtExitRange"] = "Unknown"

        return final_df

    except Exception as exc:
        st.error(f"Error in run_outbound_recidivism: {exc}")
        return pd.DataFrame()

# -----------------------------------------------------------------------------#
# Metric Calculators & Display Helpers                                        #
# -----------------------------------------------------------------------------#
@st.cache_data(show_spinner=False)
def compute_summary_metrics(final_df: pd.DataFrame) -> Dict[str, Any]:
    if final_df.empty:
        return {
            "Number of Relevant Exits": 0,
            "Total Exits to PH": 0,
            "Return": 0,
            "Return %": 0.0,
            "Return to Homelessness": 0,
            "% Return to Homelessness": 0.0,
            "Median Days": 0,
            "Average Days": 0,
            "Max Days": 0,
        }
    total_exits = len(final_df)
    total_ph_exits = final_df["PH_Exit"].sum()
    return_count = final_df["HasReturn"].sum()
    return_pct = return_count / total_exits * 100
    rth_count = final_df.loc[final_df["PH_Exit"], "ReturnToHomelessness"].sum()
    rth_pct = (rth_count / total_ph_exits * 100) if total_ph_exits else 0.0
    rtn_days = final_df["DaysToReturnEnrollment"].dropna()
    return {
        "Number of Relevant Exits": total_exits,
        "Total Exits to PH": int(total_ph_exits),
        "Return": int(return_count),
        "Return %": float(return_pct),
        "Return to Homelessness": int(rth_count),
        "% Return to Homelessness": float(rth_pct),
        "Median Days": float(rtn_days.median() if not rtn_days.empty else 0),
        "Average Days": float(rtn_days.mean() if not rtn_days.empty else 0),
        "Max Days": float(rtn_days.max() if not rtn_days.empty else 0),
    }

def display_spm_metrics(metrics: Dict[str, Any]) -> None:
    try:
        from styling import style_metric_cards
        style_metric_cards()
    except Exception:
        pass
    col1, col2, col3 = st.columns(3)
    col1.metric("Number of Relevant Exits", f"{metrics['Number of Relevant Exits']:,}")
    col2.metric("Exits to PH", f"{metrics['Total Exits to PH']:,}")
    col3.metric("Return", f"{metrics['Return']:,}")
    col4, col5, col6 = st.columns(3)
    col4.metric("% Return", f"{metrics['Return %']:.1f}%")
    col5.metric("Return → Homeless (PH)", f"{metrics['Return to Homelessness']:,}")
    col6.metric("% Return → Homeless (PH)", f"{metrics['% Return to Homelessness']:.1f}%")
    col7, col8, col9 = st.columns(3)
    col7.metric("Median Days", f"{metrics['Median Days']:.1f}")
    col8.metric("Average Days", f"{metrics['Average Days']:.1f}")
    col9.metric("Max Days", f"{metrics['Max Days']:.0f}")

def display_spm_metrics_ph(metrics: Dict[str, Any]) -> None:
    col1, col2, col3 = st.columns(3)
    col1.metric("PH Exits", f"{metrics['Number of Relevant Exits']:,}")
    col2.metric("Return", f"{metrics['Return']:,}")
    col3.metric("% Return", f"{metrics['Return %']:.1f}%")
    col4, col5, col6 = st.columns(3)
    col4.metric("Return → Homeless", f"{metrics['Return to Homelessness']:,}")
    col5.metric("% Return → Homeless", f"{metrics['% Return to Homelessness']:.1f}%")
    col6.metric("Max Days", f"{metrics['Max Days']:.0f}")
    col7, col8, _ = st.columns(3)
    col7.metric("Median Days", f"{metrics['Median Days']:.1f}")
    col8.metric("Average Days", f"{metrics['Average Days']:.1f}")

def display_spm_metrics_non_ph(metrics: Dict[str, Any]) -> None:
    col1, col2, col3 = st.columns(3)
    col1.metric("Non‑PH Exits", f"{metrics['Number of Relevant Exits']:,}")
    col2.metric("Return", f"{metrics['Return']:,}")
    col3.metric("% Return", f"{metrics['Return %']:.1f}%")
    col4, col5, col6 = st.columns(3)
    col4.metric("Median Days", f"{metrics['Median Days']:.1f}")
    col5.metric("Average Days", f"{metrics['Average Days']:.1f}")
    col6.metric("Max Days", f"{metrics['Max Days']:.0f}")

# -----------------------------------------------------------------------------#
# Breakdown & Top‑flows                                                       #
# -----------------------------------------------------------------------------#
@st.cache_data(show_spinner=False)
def breakdown_by_columns(final_df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    if final_df.empty or not columns:
        return pd.DataFrame()
    records: List[Dict[str, Any]] = []
    for keys, grp in final_df.groupby(columns):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(columns, keys))
        m = compute_summary_metrics(grp)
        row.update({
            "Number of Relevant Exits": m["Number of Relevant Exits"],
            "Exits to PH": m["Total Exits to PH"],
            "Return": f"{m['Return']} ({m['Return %']:.1f}%)",
            "Return → Homeless": f"{m['Return to Homelessness']} ({m['% Return to Homelessness']:.1f}%)",
            "Median Days": f"{m['Median Days']:.1f}",
            "Average Days": f"{m['Average Days']:.1f}",
            "Max Days": f"{m['Max Days']:.0f}",
        })
        records.append(row)
    return pd.DataFrame(records).sort_values("Number of Relevant Exits", ascending=False)

def get_top_flows_from_pivot(pivot_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Flatten pivot to top‑*n* flows by count, **excluding** any flows
    where either the source or target is "No Return".
    """
    total = pivot_df.values.sum()
    flows = [
        {
            "Source": src,
            "Target": tgt,
            "Count": int(val),
            "Percent": (val / total * 100) if total else 0,
        }
        for src, row in pivot_df.iterrows()
        for tgt, val in row.items()
        if val > 0 and src != "No Return" and tgt != "No Return"
    ]
    return (
        pd.DataFrame(flows)
          .sort_values("Count", ascending=False)
          .head(top_n)
    )


@st.cache_data(show_spinner=False)
def create_flow_pivot(
    final_df: pd.DataFrame,
    exit_col: str,
    return_col: str,
) -> pd.DataFrame:
    """
    Build a crosstab pivot table for flow analysis using the exact column names provided,
    **including** “No Return” as a category.
    """
    df_copy = final_df.copy()
    df_copy[return_col] = df_copy[return_col].fillna("No Return").astype(str)
    if exit_col not in df_copy.columns:
        return pd.DataFrame()
    return pd.crosstab(
        df_copy[exit_col],
        df_copy[return_col],
        margins=False
    )

def plot_flow_sankey(
    pivot_df: pd.DataFrame,
    title: str = "Exit → Return Sankey Diagram"
) -> go.Figure:
    """
    Build a Sankey diagram to visualize the flow from exit to return categories.
    """
    # If no flows are present
    if pivot_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title_text="No flows available",
            template="plotly_dark"
        )
        return fig

    df = pivot_df.copy()
    exit_cats = df.index.tolist()
    return_cats = df.columns.tolist()
    nodes = exit_cats + return_cats
    n_exit = len(exit_cats)
    node_types = ["Exit"] * n_exit + ["Return"] * len(return_cats)

    sources, targets, values = [], [], []
    for i, ecat in enumerate(exit_cats):
        for j, rcat in enumerate(return_cats):
            count = df.loc[ecat, rcat]
            if count > 0:
                sources.append(i)
                targets.append(n_exit + j)
                values.append(int(count))

    sankey = go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="#FFFFFF", width=0.5),
            label=nodes,
            color="#1f77b4",
            customdata=node_types,
            hovertemplate="%{label}<br>%{customdata}: %{value}<extra></extra>"
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color="rgba(255,127,14,0.6)",
            hovertemplate="From: %{source.label}<br>To: %{target.label}<br>= %{value}<extra></extra>"
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

def plot_days_to_return_box(final_df: pd.DataFrame) -> go.Figure:
    """
    Horizontal box‑plot of *DaysToReturnEnrollment*.
    """
    data = final_df["DaysToReturnEnrollment"].dropna()
    if data.empty:
        return go.Figure(
            layout={
                "title": {"text": "No Return Enrollments Found"},
                "template": "plotly_dark",
            }
        )

    median_val = float(data.median())
    avg_val = float(data.mean())

    fig = go.Figure()
    fig.add_trace(
        go.Box(
            x=data.astype(float),
            orientation="h",
            name="Days to Return",
            boxmean="sd",
        )
    )
    fig.update_layout(
        title="Distribution of Days to Return Enrollment",
        template="plotly_dark",
        xaxis_title="Days",
        shapes=[
            dict(type="line", x0=median_val, x1=median_val, yref="paper", y0=0, y1=1, line=dict(dash="dot", width=2)),
            dict(type="line", x0=avg_val, x1=avg_val, yref="paper", y0=0, y1=1, line=dict(dash="dash", width=2)),
        ],
        showlegend=False,
    )
    return fig
