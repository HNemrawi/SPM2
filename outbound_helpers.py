import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime

# ----------------------------------------------------------------------------------
# SPM Outbound Recidivism Helpers Module
# ----------------------------------------------------------------------------------

PH_PROJECTS = {
    "PH – Housing Only",
    "PH – Housing with Services (no disability required for entry)",
    "PH – Permanent Supportive Housing (disability required for entry)",
    "PH – Rapid Re-Housing"
}

NON_HOMELESS_PROJECTS = {
    "Coordinated Entry",
    "Day Shelter",
    "Homelessness Prevention",
    "Other",
    "Services Only"
}


def _find_earliest_return_any(client_df: pd.DataFrame,
                              exit_date: pd.Timestamp,
                              exit_enrollment_id: int):
    """
    Find the earliest next enrollment (any project type) strictly after exit_date.
    Returns (row, gap_days) if found, else None.
    """
    try:
        df_next = client_df.copy().dropna(subset=["ProjectStart"])
        df_next = df_next[df_next["ProjectStart"] > exit_date]
        if df_next.empty:
            return None

        df_next = df_next.sort_values(["ProjectStart", "EnrollmentID"])
        for row in df_next.itertuples(index=False):
            if row.EnrollmentID == exit_enrollment_id:
                continue
            gap_days = (row.ProjectStart - exit_date).days
            return (row, gap_days)
        return None
    except Exception as e:
        st.error(f"Error in _find_earliest_return_any: {e}")
        return None


def _find_earliest_return_homeless(client_df: pd.DataFrame,
                                   exit_date: pd.Timestamp,
                                   exit_enrollment_id: int):
    """
    Find the earliest next enrollment that qualifies as a "return to homelessness":
      1. Exclude 'NON_HOMELESS_PROJECTS' (CE, HP, Services Only, etc.).
      2. Among the remaining, find the earliest with ProjectStart > exit_date.
      3. If next project is in PH and gap <= 14 days => Not a literal return
         Else => Return to homelessness = True
    Returns (row, is_return, gap_days) if found, else None.
    """
    try:
        df_next = client_df.copy().dropna(subset=["ProjectStart"])
        df_next = df_next[~df_next["ProjectTypeCode"].isin(NON_HOMELESS_PROJECTS)]
        df_next = df_next[df_next["ProjectStart"] > exit_date]
        if df_next.empty:
            return None

        df_next = df_next.sort_values(["ProjectStart", "EnrollmentID"])
        for row in df_next.itertuples(index=False):
            if row.EnrollmentID == exit_enrollment_id:
                continue
            gap_days = (row.ProjectStart - exit_date).days
            if row.ProjectTypeCode in PH_PROJECTS:
                # If PH and gap <= 14 => not a return to literal homelessness
                if gap_days <= 14:
                    return (row, False, gap_days)
                else:
                    return (row, True, gap_days)
            else:
                # Non-PH (but not in NON_HOMELESS_PROJECTS) => immediate return
                return (row, True, gap_days)
        return None
    except Exception as e:
        st.error(f"Error in _find_earliest_return_homeless: {e}")
        return None


@st.cache_data(show_spinner=False)
def run_outbound_recidivism(
    df: pd.DataFrame,
    report_start: pd.Timestamp,
    report_end: pd.Timestamp,
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
    return_projects: list = None
) -> pd.DataFrame:
    """
    Processes outbound recidivism with two layers of logic:
      1) "Return": earliest next enrollment for any project type (ignoring PH rule).
      2) "Return to Homelessness": excludes 'NON_HOMELESS_PROJECTS' and applies the 14-day PH rule.
    """
    try:
        req_cols = ["ClientID", "ProjectStart", "ProjectExit", "ProjectTypeCode", "EnrollmentID"]
        missing = [c for c in req_cols if c not in df.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
            return pd.DataFrame()

        # If not specified, assume all codes in the dataset
        if return_projects is None:
            return_projects = df["ProjectTypeCode"].dropna().unique().tolist()
        if exiting_projects is None:
            exiting_projects = df["ProjectTypeCode"].dropna().unique().tolist()

        if not exiting_projects:
            st.error("No project types selected for Exits.")
            return pd.DataFrame()
        if not return_projects:
            st.error("No project types selected for Returns.")
            return pd.DataFrame()

        # ---------------- Filter exit data ----------------
        df_exit = df.copy()
        if exit_cocs and "ProgramSetupCoC" in df_exit.columns:
            df_exit = df_exit[df_exit["ProgramSetupCoC"].isin(exit_cocs)]
        if exit_localcocs and "LocalCoCCode" in df_exit.columns:
            df_exit = df_exit[df_exit["LocalCoCCode"].isin(exit_localcocs)]
        if exit_agencies and "AgencyName" in df_exit.columns:
            df_exit = df_exit[df_exit["AgencyName"].isin(exit_agencies)]
        if exit_programs and "ProgramName" in df_exit.columns:
            df_exit = df_exit[df_exit["ProgramName"].isin(exit_programs)]
        if allowed_continuum and "Programs Continuum Project" in df_exit.columns:
            df_exit = df_exit[df_exit["Programs Continuum Project"].isin(allowed_continuum)]
        if allowed_exit_dest_cats and "ExitDestinationCat" in df_exit.columns:
            df_exit = df_exit[df_exit["ExitDestinationCat"].isin(allowed_exit_dest_cats)]

        df_exit = df_exit.dropna(subset=["ProjectExit"])
        df_exit = df_exit[df_exit["ProjectTypeCode"].isin(exiting_projects)]
        df_exit = df_exit[(df_exit["ProjectExit"] >= report_start) & (df_exit["ProjectExit"] <= report_end)]
        if df_exit.empty:
            return pd.DataFrame()

        # Keep each client's LAST exit in that date window
        df_exit = (
            df_exit.sort_values("ProjectExit")
                   .groupby("ClientID", as_index=False)
                   .tail(1)
                   .sort_values("ClientID")
        )

        # ---------------- Filter return data ----------------
        df_return = df.copy()
        if return_cocs and "ProgramSetupCoC" in df_return.columns:
            df_return = df_return[df_return["ProgramSetupCoC"].isin(return_cocs)]
        if return_localcocs and "LocalCoCCode" in df_return.columns:
            df_return = df_return[df_return["LocalCoCCode"].isin(return_localcocs)]
        if return_agencies and "AgencyName" in df_return.columns:
            df_return = df_return[df_return["AgencyName"].isin(return_agencies)]
        if return_programs and "ProgramName" in df_return.columns:
            df_return = df_return[df_return["ProgramName"].isin(return_programs)]
        if allowed_continuum and "Programs Continuum Project" in df_return.columns:
            df_return = df_return[df_return["Programs Continuum Project"].isin(allowed_continuum)]

        df_return = df_return[df_return["ProjectTypeCode"].isin(return_projects)]
        grouped_return = df_return.groupby("ClientID")

        # ---------------- Build final output ----------------
        all_rows = []
        for row in df_exit.itertuples(index=False):
            cid = row.ClientID
            exit_date = row.ProjectExit
            exit_enrollment_id = row.EnrollmentID

            out_dict = {f"Exit_{col}": getattr(row, col, None) for col in df_exit.columns}
            out_dict["DaysToReturnEnrollment"] = pd.NA
            out_dict["ReturnToHomelessness"] = False
            out_dict["HasReturn"] = False

            # Pre-assign placeholders for Return_ columns
            for ccol in df_return.columns:
                out_dict[f"Return_{ccol}"] = pd.NA

            if cid not in grouped_return.groups:
                # No possible next enrollment
                all_rows.append(out_dict)
                continue

            # 1) Earliest next enrollment of any type
            found_any = _find_earliest_return_any(
                grouped_return.get_group(cid),
                exit_date=exit_date,
                exit_enrollment_id=exit_enrollment_id
            )
            if found_any:
                next_enroll_any, gap_days_any = found_any
                out_dict["DaysToReturnEnrollment"] = gap_days_any
                out_dict["HasReturn"] = True
                for ccol in df_return.columns:
                    out_dict[f"Return_{ccol}"] = getattr(next_enroll_any, ccol, None)

            # 2) Earliest next enrollment for homeless logic
            found_hmls = _find_earliest_return_homeless(
                grouped_return.get_group(cid),
                exit_date=exit_date,
                exit_enrollment_id=exit_enrollment_id
            )
            if found_hmls:
                _, is_rth, _ = found_hmls
                if is_rth:
                    out_dict["ReturnToHomelessness"] = True

            all_rows.append(out_dict)

        final_df = pd.DataFrame(all_rows)

        # Flag PH_Exit if "ExitDestinationCat" is "Permanent Housing Situations"
        if "Exit_ExitDestinationCat" in final_df.columns:
            final_df["PH_Exit"] = final_df["Exit_ExitDestinationCat"].eq("Permanent Housing Situations")
        else:
            final_df["PH_Exit"] = False

        # Optional: AgeAtExitRange
        if "Exit_DOB" in final_df.columns and "Exit_ProjectExit" in final_df.columns:
            age_days = (final_df["Exit_ProjectExit"] - final_df["Exit_DOB"]).dt.days
            age_years = age_days / 365.25

            def age_range(a):
                if pd.isna(a):
                    return "Unknown"
                if a < 18:
                    return "0 to 17"
                elif a < 25:
                    return "18 to 24"
                elif a < 35:
                    return "25 to 34"
                elif a < 45:
                    return "35 to 44"
                elif a < 55:
                    return "45 to 54"
                elif a < 65:
                    return "55 to 64"
                return "65 or Above"

            final_df["AgeAtExitRange"] = age_years.apply(age_range)
        else:
            final_df["AgeAtExitRange"] = "Unknown"

        return final_df
    except Exception as e:
        st.error(f"Error in run_outbound_recidivism: {e}")
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def compute_summary_metrics(final_df: pd.DataFrame) -> dict:
    """
    Compute summary metrics with correct Return to Homelessness logic:
      - "Return to Homelessness" is only relevant if PH_Exit == True.
    """
    # Return a default dictionary if empty
    if final_df.empty:
        return {
            "Total Exits": 0,
            "Total Exits to PH": 0,
            "Return": 0,
            "Return %": 0.0,
            "Return to Homelessness": 0,
            "% Return to Homelessness": 0.0,
            "Median Days": 0,
            "Average Days": 0,
            "Max Days": 0
        }

    # 1) Overall counts
    total_exits = len(final_df)

    # 2) Identify how many are PH exits
    if "PH_Exit" not in final_df.columns:
        final_df["PH_Exit"] = False
    ph_subset = final_df[final_df["PH_Exit"] == True]
    total_ph_exits = len(ph_subset)

    # 3) "Return" metrics (any next enrollment)
    return_count = final_df["HasReturn"].sum() if "HasReturn" in final_df.columns else 0
    return_pct = (return_count / total_exits * 100) if total_exits > 0 else 0.0

    # 4) "Return to Homelessness" among PH exits only
    if "ReturnToHomelessness" in ph_subset.columns and not ph_subset.empty:
        rth_count = ph_subset["ReturnToHomelessness"].sum()
    else:
        rth_count = 0

    if total_ph_exits > 0:
        rth_pct = (rth_count / total_ph_exits) * 100
    else:
        rth_pct = 0.0

    # 5) Days to Return stats (any next enrollment)
    valid_days = final_df.dropna(subset=["DaysToReturnEnrollment"])
    if valid_days.empty:
        median_days = 0
        avg_days = 0
        max_days = 0
    else:
        median_days = valid_days["DaysToReturnEnrollment"].median()
        avg_days = valid_days["DaysToReturnEnrollment"].mean()
        max_days = valid_days["DaysToReturnEnrollment"].max()

    return {
        "Total Exits": total_exits,
        "Total Exits to PH": total_ph_exits,
        "Return": return_count,
        "Return %": return_pct,
        "Return to Homelessness": rth_count,
        "% Return to Homelessness": rth_pct,
        "Median Days": median_days,
        "Average Days": avg_days,
        "Max Days": max_days
    }


def display_spm_metrics(metrics: dict):
    """
    Displays key metrics in a 3x3 grid for an overall summary.
    Note: "Return to Homelessness" and "% Return to Homelessness" 
    now reflect only PH exits.
    """
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Exits", f"{metrics['Total Exits']}")
    col2.metric("Total Exits to PH", f"{metrics['Total Exits to PH']}")
    col3.metric("Return", f"{metrics['Return']}")

    col4, col5, col6 = st.columns(3)
    col4.metric("% Return", f"{metrics['Return %']:.1f}%")
    col5.metric("Return to Homelessness (PH)", f"{metrics['Return to Homelessness']}")
    col6.metric("% Return to Homelessness (PH)", f"{metrics['% Return to Homelessness']:.1f}%")

    col7, col8, col9 = st.columns(3)
    col7.metric("Median Days to Return", f"{metrics['Median Days']:.1f}")
    col8.metric("Avg Days to Return", f"{metrics['Average Days']:.1f}")
    col9.metric("Max Days to Return", f"{metrics['Max Days']:.0f}")


@st.cache_data(show_spinner=False)
def breakdown_by_columns(final_df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Group by specified columns, compute metrics within each group.
    """
    if final_df.empty or not columns:
        return pd.DataFrame()

    grouped = final_df.groupby(columns)
    rows = []
    for vals, subset_df in grouped:
        gvals = vals if isinstance(vals, tuple) else (vals,)
        row_data = dict(zip(columns, gvals))

        m = compute_summary_metrics(subset_df)
        row_data["Total Exits"] = m["Total Exits"]
        row_data["Total Exits to PH"] = m["Total Exits to PH"]
        row_data["Return"] = f"{m['Return']} ({m['Return %']:.1f}%)"
        row_data["Return to Homelessness"] = f"{m['Return to Homelessness']} ({m['% Return to Homelessness']:.1f}%)"
        row_data["Median Days"] = f"{m['Median Days']:.1f}"
        row_data["Average Days"] = f"{m['Average Days']:.1f}"
        row_data["Max Days"] = f"{m['Max Days']:.0f}"
        rows.append(row_data)

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values("Total Exits", ascending=False)
    return out_df


@st.cache_data(show_spinner=False)
def create_flow_pivot(final_df: pd.DataFrame, exit_col: str, return_col: str) -> pd.DataFrame:
    """
    Crosstab where rows = Exit_{exit_col}, columns = Return_{return_col}.
    Only includes rows with Return_EnrollmentID not NA (i.e., they actually returned).
    """
    df_ok = final_df.dropna(subset=["Return_EnrollmentID"]).copy()
    if df_ok.empty:
        return pd.DataFrame()

    pivot = pd.crosstab(
        df_ok[f"Exit_{exit_col}"],
        df_ok[f"Return_{return_col}"]
    )
    return pivot


def get_top_flows_from_pivot(pivot_df: pd.DataFrame, top_n=10) -> pd.DataFrame:
    """
    Returns top N flows (source -> target) by count from a crosstab pivot.
    """
    total = pivot_df.values.sum()
    flows = []
    for src in pivot_df.index:
        for tgt in pivot_df.columns:
            val = pivot_df.loc[src, tgt]
            if val > 0:
                pct = (val / total * 100) if total > 0 else 0
                flows.append({"Source": src, "Target": tgt, "Count": val, "Percent": pct})
    flows_df = pd.DataFrame(flows).sort_values("Count", ascending=False)
    return flows_df.head(top_n)


def plot_flow_sankey(pivot_df: pd.DataFrame, title: str) -> go.Figure:
    """
    Creates a Sankey diagram from the pivot (rows=exit dimension, cols=return dimension).
    """
    node_labels = list(pivot_df.index) + list(pivot_df.columns)
    idx_map = {}
    i = 0
    for label in pivot_df.index:
        idx_map[label] = i
        i += 1
    for label in pivot_df.columns:
        idx_map[label] = i
        i += 1

    sources, targets, values = [], [], []
    for s in pivot_df.index:
        for t in pivot_df.columns:
            v = pivot_df.loc[s, t]
            if v > 0:
                sources.append(idx_map[s])
                targets.append(idx_map[t])
                values.append(v)

    sankey = go.Sankey(
        node=dict(label=node_labels, pad=15, thickness=20),
        link=dict(source=sources, target=targets, value=values)
    )
    fig = go.Figure(data=[sankey])
    fig.update_layout(title_text=title, template="plotly_dark")
    return fig


def plot_days_to_return_box(final_df: pd.DataFrame) -> go.Figure:
    """
    Box plot for distribution of DaysToReturnEnrollment.
    """
    valid_df = final_df.dropna(subset=["DaysToReturnEnrollment"])
    fig = go.Figure()
    if valid_df.empty:
        fig.update_layout(title="No Return Enrollments Found", template="plotly_dark")
        return fig

    xvals = valid_df["DaysToReturnEnrollment"].astype(float)
    median_val = xvals.median()
    avg_val = xvals.mean()

    fig.add_trace(go.Box(x=xvals, boxmean='sd', name="Days to Return Enrollment"))
    fig.update_layout(
        title="Distribution of Days to Return Enrollment",
        template="plotly_dark",
        xaxis_title="DaysToReturnEnrollment",
        shapes=[
            dict(
                type="line",
                xref="x", x0=median_val, x1=median_val,
                yref="paper", y0=0, y1=1,
                line=dict(dash="dot", width=2)
            ),
            dict(
                type="line",
                xref="x", x0=avg_val, x1=avg_val,
                yref="paper", y0=0, y1=1,
                line=dict(dash="dash", width=2)
            )
        ]
    )
    return fig


# --------------------------------------------------------------------------
# Specialized display for PH vs. Non-PH subsets
# --------------------------------------------------------------------------

def display_spm_metrics_ph(metrics: dict):
    """
    Displays only the relevant metrics for PH exits:
      - "PH Exits" instead of "Total Exits"
      - No separate 'Total Exits to PH' (redundant)
      - Return, % Return
      - Return to Homelessness (PH), % Return to Homelessness (PH)
      - Days to Return stats
    """
    col1, col2, col3 = st.columns(3)
    col1.metric("PH Exits", f"{metrics['Total Exits']}")
    col2.metric("Return", f"{metrics['Return']}")
    col3.metric("% Return", f"{metrics['Return %']:.1f}%")

    col4, col5, col6 = st.columns(3)
    col4.metric("Return to Homelessness (PH)", f"{metrics['Return to Homelessness']}")
    col5.metric("% Return to Homelessness (PH)", f"{metrics['% Return to Homelessness']:.1f}%")
    col6.metric("Max Days to Return", f"{metrics['Max Days']:.0f}")

    col7, col8, _ = st.columns(3)
    col7.metric("Median Days to Return", f"{metrics['Median Days']:.1f}")
    col8.metric("Avg Days to Return", f"{metrics['Average Days']:.1f}")


def display_spm_metrics_non_ph(metrics: dict):
    """
    Displays only the relevant metrics for Non-PH exits:
      - "Non-PH Exits"
      - Return, % Return
      - No "Return to Homelessness (PH)" since it doesn't apply
      - Days to Return stats
    """
    col1, col2, col3 = st.columns(3)
    col1.metric("Non-PH Exits", f"{metrics['Total Exits']}")
    col2.metric("Return", f"{metrics['Return']}")
    col3.metric("% Return", f"{metrics['Return %']:.1f}%")

    col4, col5, col6 = st.columns(3)
    col4.metric("Median Days to Return", f"{metrics['Median Days']:.1f}")
    col5.metric("Avg Days to Return", f"{metrics['Average Days']:.1f}")
    col6.metric("Max Days to Return", f"{metrics['Max Days']:.0f}")
