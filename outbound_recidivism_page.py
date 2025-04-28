"""
Outbound Recidivism Streamlit Page
==================================
End‑to‑end UI that mirrors the layout, visual hierarchy, and UX polish
of the SPM 2 page.  Place this file alongside `outbound_helpers.py`.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

import pandas as pd
import streamlit as st

from outbound_helpers import (
    breakdown_by_columns,
    compute_summary_metrics,
    create_flow_pivot,
    display_spm_metrics,
    display_spm_metrics_non_ph,
    display_spm_metrics_ph,
    get_top_flows_from_pivot,
    plot_days_to_return_box,
    plot_flow_sankey,
    run_outbound_recidivism,
)

# -----------------------------------------------------------------------------#
# Sidebar Utilities                                                            #
# -----------------------------------------------------------------------------#
def create_sidebar_multiselect(
    label: str,
    options: List[str],
    default: Optional[List[str]],
    help_text: str,
) -> Optional[List[str]]:
    """
    Multiselect wrapper with an “ALL” option that returns *None* if ALL selected.
    """
    selection = st.multiselect(label, ["ALL"] + options, default=default or ["ALL"], help=help_text)
    if "ALL" in selection or not selection:
        return None
    return selection

# -----------------------------------------------------------------------------#
# Main Page Function                                                           #
# -----------------------------------------------------------------------------#
def outbound_recidivism_page() -> None:
    """Render the Outbound Recidivism page."""
    st.header("📈 Outbound Recidivism Analysis")

    # ----------------- About ------------------------------------------------- #
    with st.expander("📘  About Outbound Recidivism Analysis", expanded=False):
        st.markdown(
            """
            ### How This Analysis Works

            1. **Which Exits Are Included?**  
               You configure filters (CoC, Agency, Program, Project Type,
               Destination Category, etc.) and a reporting date range.  
               Only last exit enrollment per client that match those filters,
               with exit dates in the selected window, are analyzed.

            2. **Return Definitions**  
               - **Return** = first enrollment after exit (any project).  
               - **Return to Homelessness** = first enrollment that is **not**
                 Coordinated Entry/Day Shelter/HP/Services **and** is either:
                 - a non‑PH project, or  
                 - a PH project beginning > 14 days after the exit
                   (excluding any PH transition windows same as SPM2 logic).  
               *Note:* The “Return to Homelessness” rate is calculated only for
               clients who exited to Permanent Housing Situations.

            3. **Key Output Metrics**  
               - **Number of Relevant Exits**: count of all exits in the filtered dataset.  
               - **Total Exits to PH**: count of exits whose
                 `ExitDestinationCat` = “Permanent Housing Situations.”  
               - **Return** & **% Return**: number and share of exits with
                 any subsequent enrollment.  
               - **Return to Homelessness (PH)** & **% Return to Homelessness (PH)**:
                 among PH exits, the count and share of true homelessness returns.  
               - **Median / Average / Max Days to Return**: days between exit and
                 start of that next enrollment.

            4. **PH vs. Non‑PH Comparison**  
               Side‑by‑side metrics for those who exited to PH versus those who
               exited elsewhere. PH‑specific metrics (e.g. Return to Homelessness)
               are only shown for the PH subset.
            """,
            unsafe_allow_html=True,
        )

    # ----------------- Data Check ------------------------------------------- #
    if "df" not in st.session_state or st.session_state["df"].empty:
        st.info("📭 No data uploaded!  Please upload HMIS export on the main page.")
        st.stop()

    df: pd.DataFrame = st.session_state["df"]

    # ----------------- Sidebar Configuration -------------------------------- #
    st.sidebar.header("⚙️ Outbound Parameters")

    # --- Reporting period
    with st.sidebar.expander("📅 Reporting Period", expanded=True):
        start_d, end_d = st.date_input(
            "Exit date range",
            value=[datetime(2025, 1, 1), datetime(2025, 1, 31)],
            help="Clients must have an exit date inside this window.",
        )
        report_start, report_end = pd.to_datetime(start_d), pd.to_datetime(end_d)

    # --- Exit filters
    with st.sidebar.expander("🚪 Exit Filters", expanded=True):
        exit_cocs = create_sidebar_multiselect(
            "CoC Codes",
            sorted(df["ProgramSetupCoC"].dropna().unique()) if "ProgramSetupCoC" in df.columns else [],
            default=None,
            help_text="Filter exits by CoC code",
        )
        exit_localcocs = create_sidebar_multiselect(
            "Local CoC",
            sorted(df["LocalCoCCode"].dropna().unique()) if "LocalCoCCode" in df.columns else [],
            default=None,
            help_text="Filter exits by local CoC",
        )
        exit_agencies = create_sidebar_multiselect(
            "Agencies",
            sorted(df["AgencyName"].dropna().unique()) if "AgencyName" in df.columns else [],
            default=None,
            help_text="Filter exits by agency",
        )
        exit_programs = create_sidebar_multiselect(
            "Programs",
            sorted(df["ProgramName"].dropna().unique()) if "ProgramName" in df.columns else [],
            default=None,
            help_text="Filter exits by program",
        )
        exiting_projects = st.multiselect(
            "Project Types (Exit)",
            sorted(df["ProjectTypeCode"].dropna().unique()) if "ProjectTypeCode" in df.columns else [],
            default=[
                "Street Outreach",
                "Emergency Shelter – Entry Exit",
                "Emergency Shelter – Night-by-Night",
                "Transitional Housing",
                "Safe Haven",
                "PH – Housing Only",
                "PH – Housing with Services (no disability required for entry)",
                "PH – Permanent Supportive Housing (disability required for entry)",
                "PH – Rapid Re-Housing"
            ],
            help="Project types treated as exits",
        )
        allowed_exit_dest_cats = st.multiselect(
            "Exit Destination Categories",
            sorted(df["ExitDestinationCat"].dropna().unique()) if "ExitDestinationCat" in df.columns else [],
            default=["Permanent Housing Situations"],
            help="Limit exits to these destination categories",
        )

    # --- Return filters
    with st.sidebar.expander("↩️ Return Filters", expanded=True):
        return_cocs = create_sidebar_multiselect(
            "CoC Codes",
            sorted(df["ProgramSetupCoC"].dropna().unique()) if "ProgramSetupCoC" in df.columns else [],
            default=None,
            help_text="Filter next enrollments by CoC code",
        )
        return_localcocs = create_sidebar_multiselect(
            "Local CoC",
            sorted(df["LocalCoCCode"].dropna().unique()) if "LocalCoCCode" in df.columns else [],
            default=None,
            help_text="Filter next enrollments by local CoC",
        )
        return_agencies = create_sidebar_multiselect(
            "Agencies",
            sorted(df["AgencyName"].dropna().unique()) if "AgencyName" in df.columns else [],
            default=None,
            help_text="Filter next enrollments by agency",
        )
        return_programs = create_sidebar_multiselect(
            "Programs",
            sorted(df["ProgramName"].dropna().unique()) if "ProgramName" in df.columns else [],
            default=None,
            help_text="Filter next enrollments by program",
        )
        return_projects = st.multiselect(
            "Project Types (Return)",
            sorted(df["ProjectTypeCode"].dropna().unique()) if "ProjectTypeCode" in df.columns else [],
            default=[
                "Street Outreach",
                "Emergency Shelter – Entry Exit",
                "Emergency Shelter – Night-by-Night",
                "Transitional Housing",
                "Safe Haven",
                "PH – Housing Only",
                "PH – Housing with Services (no disability required for entry)",
                "PH – Permanent Supportive Housing (disability required for entry)",
                "PH – Rapid Re-Housing"
            ],
            help="Project types treated as candidate returns",
        )

    # --- Continuum
    with st.sidebar.expander("⚡ Continuum", expanded=False):
        cont_opts = (
            sorted(df["ProgramsContinuumProject"].dropna().unique())
            if "ProgramsContinuumProject" in df.columns
            else []
        )
        chosen_continuum = create_sidebar_multiselect(
            "Programs Continuum Project",
            cont_opts,
            default=None,
            help_text="Optional continuum filter",
        )

    st.divider()

    # ----------------- Run Analysis Button ---------------------------------- #
    if st.button("▶️ Run Analysis", type="primary", use_container_width=True):
        with st.status("🔄 Running Outbound Recidivism…", expanded=True) as status:
            try:
                outbound_df = run_outbound_recidivism(
                    df,
                    report_start,
                    report_end,
                    exit_cocs=exit_cocs,
                    exit_localcocs=exit_localcocs,
                    exit_agencies=exit_agencies,
                    exit_programs=exit_programs,
                    return_cocs=return_cocs,
                    return_localcocs=return_localcocs,
                    return_agencies=return_agencies,
                    return_programs=return_programs,
                    allowed_continuum=chosen_continuum,
                    allowed_exit_dest_cats=allowed_exit_dest_cats,
                    exiting_projects=exiting_projects,
                    return_projects=return_projects,
                )
                cols_to_remove = [
                    "Return_UniqueIdentifier",
                    "Return_ClientID",
                    "Return_RaceEthnicity",
                    "Return_Gender",
                    "Return_DOB",
                    "Return_VeteranStatus",
                    "Exit_ReportingPeriodStartDate",
                    "Exit_ReportingPeriodEndDate",
                    "Return_ReportingPeriodStartDate",
                    "Return_ReportingPeriodEndDate"
                ]
                outbound_df.drop(columns=cols_to_remove, inplace=True, errors='ignore')
                cols_to_rename = [
                    "Exit_UniqueIdentifier", 
                    "Exit_ClientID",
                    "Exit_RaceEthnicity",
                    "Exit_Gender",
                    "Exit_DOB",
                    "Exit_VeteranStatus"
                ]
                mapping = {col: col[len("Exit_"):] for col in cols_to_rename if col in outbound_df.columns}
                outbound_df.rename(columns=mapping, inplace=True)

                st.session_state["outbound_df"] = outbound_df
                status.update(label="✅ Done!", state="complete")
                st.toast("Analysis complete 🎉", icon="🎉")
            except Exception as exc:
                status.update(label=f"🚨 Error: {exc}", state="error")

    # ----------------- Results Section -------------------------------------- #
    if "outbound_df" not in st.session_state or st.session_state["outbound_df"].empty:
        return

    out_df = st.session_state["outbound_df"]

    # --- Metrics
    st.divider()
    st.markdown("### 📊 Outbound Analysis Summary")
    display_spm_metrics(compute_summary_metrics(out_df))

    # --- Days‑to‑Return box
    st.divider()
    st.markdown("### ⏳ Days to Return Distribution")
    st.plotly_chart(plot_days_to_return_box(out_df), use_container_width=True)

    # --- Breakdown
    st.divider()
    st.markdown("### 📈 Breakdown")
    breakdown_columns = [
        "RaceEthnicity",
        "Gender",
        "VeteranStatus",
        "Exit_HasIncome",
        "Exit_HasDisability",
        "Exit_HouseholdType",
        "Exit_CHStartHousehold",
        "Exit_LocalCoCCode",
        "Exit_PriorLivingCat",
        "Exit_ProgramSetupCoC",
        "Exit_ProjectTypeCode",
        "Exit_AgencyName",
        "Exit_ProgramName",
        "Exit_ExitDestinationCat",
        "Exit_ExitDestination",
        "Exit_CustomProgramType",
        "Exit_AgeTieratEntry",
        "Return_HasIncome",
        "Return_HasDisability",
        "Return_HouseholdType",
        "Return_CHStartHousehold",
        "Return_LocalCoCCode",
        "Return_PriorLivingCat",
        "Return_ProgramSetupCoC",
        "Return_ProjectTypeCode",
        "Return_AgencyName",
        "Return_ProgramName",
        "Return_ExitDestinationCat",
        "Return_ExitDestination",
        "AgeAtExitRange",
    ]
    cols_to_group = st.multiselect(
        "Group by columns",
        options=[col for col in breakdown_columns if col in out_df.columns],
        default=["Exit_ProjectTypeCode"] if "Exit_ProjectTypeCode" in out_df.columns else [],
        help="Select up to 3 columns",
    )
    if cols_to_group:
        bdf = breakdown_by_columns(out_df, cols_to_group[:3])
        st.dataframe(
            bdf.style.format(thousands=",").background_gradient(cmap="Blues"),
            use_container_width=True,
        )

    # --- Flow Analysis (with drill-in) ---
    st.divider()
    st.markdown("### 🌊 Client Flow Analysis")

    exit_columns = [
        "Exit_HasIncome",
        "Exit_HasDisability",
        "Exit_HouseholdType",
        "Exit_CHStartHousehold",
        "Exit_LocalCoCCode",
        "Exit_PriorLivingCat",
        "Exit_ProgramSetupCoC",
        "Exit_ProjectTypeCode",
        "Exit_AgencyName",
        "Exit_ProgramName",
        "Exit_ExitDestinationCat",
        "Exit_ExitDestination",
    ]
    return_columns = [
        "Return_HasIncome",
        "Return_HasDisability",
        "Return_HouseholdType",
        "Return_CHStartHousehold",
        "Return_LocalCoCCode",
        "Return_PriorLivingCat",
        "Return_ProgramSetupCoC",
        "Return_ProjectTypeCode",
        "Return_AgencyName",
        "Return_ProgramName",
        "Return_ExitDestinationCat",
        "Return_ExitDestination",
    ]
    exit_dims = [c for c in exit_columns   if c in out_df.columns]
    ret_dims  = [c for c in return_columns if c in out_df.columns]

    if exit_dims and ret_dims:
        # Dimension selectors
        flow_cols = st.columns(2)
        with flow_cols[0]:
            ex_choice = st.selectbox(
                "Exit Dimension: Rows",
                exit_dims,
                index=exit_dims.index("Exit_ProjectTypeCode") if "Exit_ProjectTypeCode" in exit_dims else 0,
            )
        with flow_cols[1]:
            ret_choice = st.selectbox(
                "Return Dimension: Columns",
                ret_dims,
                index=ret_dims.index("Return_ProjectTypeCode") if "Return_ProjectTypeCode" in ret_dims else 0,
            )

        # Build the full pivot
        pivot_c = create_flow_pivot(out_df, ex_choice, ret_choice)
        if "No Return" in pivot_c.columns:
            cols_order = [c for c in pivot_c.columns if c != "No Return"] + ["No Return"]
            pivot_c = pivot_c[cols_order]

        # Drill-in controls
        colL, colR = st.columns(2)
        focus_exit   = colL.selectbox(
            "🔍 Focus Exit Dimension",
            ["All"] + pivot_c.index.tolist(),
            help="Show only this exit in the flow",
        )
        focus_return = colR.selectbox(
            "🔍 Focus Return Dimension",
            ["All"] + pivot_c.columns.tolist(),
            help="Show only this return in the flow",
        )

        # Subset pivot_c in place
        if focus_exit != "All":
            pivot_c = pivot_c.loc[[focus_exit]]
        if focus_return != "All":
            pivot_c = pivot_c[[focus_return]]

        # 1) Flow Matrix Details
        with st.expander("🔍 Flow Matrix Details", expanded=True):
            if pivot_c.empty:
                st.info("📭 No return enrollments to build flow.")
            else:
                cols_to_color = [c for c in pivot_c.columns if c != "No Return"]
                st.dataframe(
                    pivot_c
                        .style
                        .background_gradient(cmap="Blues", subset=cols_to_color, axis=1)
                        .format(precision=0),
                    use_container_width=True,
                )

        # 2) Top Client Pathways
        st.markdown("#### 🔝 Top Client Pathways")
        top_n = st.slider(
            "Number of Flows",
            min_value=5,
            max_value=25,
            value=10,
            step=1,
            help="Top N pathways to display",
        )
        top_flows_df = get_top_flows_from_pivot(pivot_c, top_n=top_n)
        if not top_flows_df.empty:
            if "Percent" in top_flows_df.columns:
                top_flows_df["Percent"] = top_flows_df["Percent"].astype(float)
                styled = top_flows_df.style.format({'Percent': '{:.1f}%'}).background_gradient(
                    cmap="Blues", subset=["Count", "Percent"]
                )
                st.dataframe(styled, use_container_width=True)
            else:
                st.dataframe(top_flows_df.style.background_gradient(cmap="Blues"), use_container_width=True)
        else:
            st.info("No significant pathways detected")

        # 3) Client Flow Network
        st.markdown("#### 🌐 Client Flow Network")
        sankey_fig = plot_flow_sankey(pivot_c, f"{ex_choice} → {ret_choice}")
        st.plotly_chart(sankey_fig, use_container_width=True)

    else:
        st.info("📭 Insufficient data for flow analysis")


    # --- PH vs Non‑PH comparison
    st.divider()
    st.markdown("### 🏠 PH vs. Non‑PH Exit Comparison")
    if st.checkbox("Show comparison", value=False):
        ph_df = out_df[out_df["PH_Exit"]]
        nonph_df = out_df[~out_df["PH_Exit"]]
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("PH Exits")
            if not ph_df.empty:
                display_spm_metrics_ph(compute_summary_metrics(ph_df))
            else:
                st.info("No PH exits found.")
        with c2:
            st.subheader("Non‑PH Exits")
            if not nonph_df.empty:
                display_spm_metrics_non_ph(compute_summary_metrics(nonph_df))
            else:
                st.info("No Non‑PH exits found.")

    # --- Download
    st.divider()
    st.download_button(
        "⬇️ Download results (CSV)",
        data=out_df.to_csv(index=False),
        file_name="OutboundRecidivismResults.csv",
        mime="text/csv",
        use_container_width=True,
    )

if __name__ == "__main__":
    st.set_page_config(page_title="Outbound Recidivism", layout="wide")
    outbound_recidivism_page()
