"""
Outbound Recidivism Page
========================
A dedicated page for analyzing outbound recidivism:
 - Clients who exit during the reporting period.
 - Earliest next enrollment after each exit.
 - 14-day logic for "Return to Homelessness."
Mirrors the layout of the SPM2 page.
"""

import streamlit as st
import pandas as pd
from datetime import datetime

# Import our new helper functions
from outbound_helpers import (
    run_outbound_recidivism,
    compute_summary_metrics,
    display_spm_metrics,
    breakdown_by_columns,
    plot_days_to_return_box,
    create_flow_pivot,
    get_top_flows_from_pivot,
    plot_flow_sankey
)

# ----------------------------------------------------------------------------------
# Outbound Recidivism Analysis Page
# ----------------------------------------------------------------------------------

def create_sidebar_multiselect(label: str, options: list, default: list, help_text: str):
    """
    Sidebar helper for multiselect with an "ALL" option.
    """
    selection = st.multiselect(
        label=label,
        options=["ALL"] + options,
        default=default,
        help=help_text
    )
    if "ALL" in selection:
        return None
    if not selection:
        st.error(f"Please select at least one option for {label}, or choose ALL.")
        return None
    return selection

def outbound_recidivism_page():
    """
    Renders the Outbound Recidivism Analysis page using the updated logic and enhanced error handling.
    """
    st.header("📊 Outbound Recidivism Analysis")

    with st.expander("📘 Methodology", expanded=False):
        st.markdown("""
        **Outbound Recidivism** analysis works in two steps:

        **1. Return (Next Enrollment):**  
        - For each client's *last exit* within the reporting period, we search for the earliest next enrollment (i.e. the next enrollment that starts after the exit date).  
        - If found, we record the number of days between the exit and the next enrollment as **DaysToReturnEnrollment**. This simply tells us which clients have a documented next enrollment.

        **2. Return to Homelessness:**  
        - This calculation applies extra rules to determine if that next enrollment reflects an actual return to homelessness.
        - **If the next enrollment is in a non-permanent housing (non-PH) project:** The client is flagged immediately as having returned to homelessness.
        - **If the next enrollment is in a PH (permanent housing) project:** We check the gap in days:
            - **14 days or less:** The client is likely transitioning seamlessly within housing services, so it is **not** considered a return to homelessness.
            - **More than 14 days:** The gap is long enough that the client is flagged as a return to homelessness.

        In short, “Return” means the client has another enrollment after exiting, while “Return to Homelessness” differentiates between a smooth transfer within housing services and a genuine relapse into homelessness.
        """)


    if "df" not in st.session_state or st.session_state["df"].empty:
        st.info("No data loaded. Please upload data on the main page sidebar.")
        return

    df = st.session_state["df"]

    # Sidebar: Configuration
    st.sidebar.header("⚙️ Outbound Recidivism Parameters")

    with st.sidebar.expander("📅 Reporting Period", expanded=True):
        try:
            date_range = st.date_input(
                "Reporting Period (for Exits)",
                [datetime(2023, 10, 1), datetime(2024, 9, 30)],
                help="Clients must have an exit date within this range."
            )
            if len(date_range) != 2:
                st.error("⚠️ Please select both a start and end date.")
                st.stop()

            report_start, report_end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        except Exception as e:
            st.error(f"📅 Date Error: {str(e)}")
            st.stop()


    with st.sidebar.expander("🚪 Exit Filters", expanded=True):
        exit_cocs = create_sidebar_multiselect(
            "CoC Codes - Exit",
            sorted(df["ProgramSetupCoC"].dropna().unique()) if "ProgramSetupCoC" in df.columns else [],
            default=["ALL"],
            help_text="Filter exit enrollments by CoC code"
        )
        exit_localcocs = create_sidebar_multiselect(
            "Local CoC - Exit",
            sorted(df["LocalCoCCode"].dropna().unique()) if "LocalCoCCode" in df.columns else [],
            default=["ALL"],
            help_text="Filter exit enrollments by local CoC"
        )
        exit_agencies = create_sidebar_multiselect(
            "Agencies - Exit",
            sorted(df["AgencyName"].dropna().unique()) if "AgencyName" in df.columns else [],
            default=["ALL"],
            help_text="Filter exit enrollments by agency"
        )
        exit_programs = create_sidebar_multiselect(
            "Programs - Exit",
            sorted(df["ProgramName"].dropna().unique()) if "ProgramName" in df.columns else [],
            default=["ALL"],
            help_text="Filter exit enrollments by program"
        )
        default_exit_types = [
            "Street Outreach",
            "Emergency Shelter – Entry Exit",
            "Emergency Shelter – Night-by-Night",
            "Transitional Housing",
            "Safe Haven",
            "PH – Housing Only",
            "PH – Housing with Services (no disability required for entry)",
            "PH – Permanent Supportive Housing (disability required for entry)",
            "PH – Rapid Re-Housing"
        ]
        exiting_projects = st.multiselect(
            "Exit Project Types",
            sorted(df["ProjectTypeCode"].dropna().unique()) if "ProjectTypeCode" in df.columns else [],
            default=default_exit_types,
            help="Project types considered as exits"
        )
        allowed_exit_dest_cats = st.multiselect(
            "Exit Destination Categories",
            sorted(df["ExitDestinationCat"].dropna().unique()) if "ExitDestinationCat" in df.columns else [],
            default=["Permanent Housing Situations"],
            help="Filter exits by exit destination category"
        )

    with st.sidebar.expander("↩️ Return Filters", expanded=True):
        return_cocs = create_sidebar_multiselect(
            "CoC Codes - Return",
            sorted(df["ProgramSetupCoC"].dropna().unique()) if "ProgramSetupCoC" in df.columns else [],
            default=["ALL"],
            help_text="Filter next enrollments by CoC code"
        )
        return_localcocs = create_sidebar_multiselect(
            "Local CoC - Return",
            sorted(df["LocalCoCCode"].dropna().unique()) if "LocalCoCCode" in df.columns else [],
            default=["ALL"],
            help_text="Filter next enrollments by local CoC code"
        )
        return_agencies = create_sidebar_multiselect(
            "Agencies - Return",
            sorted(df["AgencyName"].dropna().unique()) if "AgencyName" in df.columns else [],
            default=["ALL"],
            help_text="Filter next enrollments by agency"
        )
        return_programs = create_sidebar_multiselect(
            "Programs - Return",
            sorted(df["ProgramName"].dropna().unique()) if "ProgramName" in df.columns else [],
            default=["ALL"],
            help_text="Filter next enrollments by program"
        )
        default_return_types = [
            "Street Outreach",
            "Emergency Shelter – Entry Exit",
            "Emergency Shelter – Night-by-Night",
            "Safe Haven",
            "Transitional Housing",
            "PH – Housing Only",
            "PH – Housing with Services (no disability required for entry)",
            "PH – Permanent Supportive Housing (disability required for entry)",
            "PH – Rapid Re-Housing"
        ]
        return_projects = st.multiselect(
            "Return Project Types",
            sorted(df["ProjectTypeCode"].dropna().unique()) if "ProjectTypeCode" in df.columns else [],
            default=default_return_types,
            help="Project types considered for next enrollment"
        )

    with st.sidebar.expander("⚡ Continuum Filter", expanded=False):
        if "Programs Continuum Project" in df.columns:
            continuum_list = sorted(df["Programs Continuum Project"].dropna().unique())
            chosen_continuum = st.multiselect(
                "Programs Continuum Project",
                ["ALL"] + continuum_list,
                default=["ALL"],
                help="Filter for continuum projects"
            )
            if "ALL" in chosen_continuum:
                chosen_continuum = None
        else:
            chosen_continuum = None

    st.divider()

    if st.button("▶️ Run Outbound Recidivism Analysis", type="primary", use_container_width=True):
        try:
            with st.spinner("Processing..."):
                result_df = run_outbound_recidivism(
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
                    return_projects=return_projects
                )
                st.session_state["outbound_df"] = result_df
                st.success("Outbound Recidivism Analysis Complete!")
        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")

    if "outbound_df" in st.session_state and not st.session_state["outbound_df"].empty:
        final_df_c = st.session_state["outbound_df"]

        st.divider()
        st.markdown("### 📊 Key Metrics")
        metrics = compute_summary_metrics(final_df_c)
        display_spm_metrics(metrics)

        st.divider()
        st.markdown("### ⏳ Days to Return Distribution")
        fig_box = plot_days_to_return_box(final_df_c)
        st.plotly_chart(fig_box, use_container_width=True)

        st.divider()
        st.markdown("### 📈 Breakdown by Columns")
        bcols = st.multiselect(
            "Group By",
            final_df_c.columns.tolist(),
            default=["Exit_ProjectTypeCode"] if "Exit_ProjectTypeCode" in final_df_c.columns else []
        )
        if bcols:
            bdf = breakdown_by_columns(final_df_c, bcols)
            st.dataframe(bdf, use_container_width=True)

        st.divider()
        st.markdown("### 🌊 Flow Analysis")
        exit_candidates = [c.replace("Exit_","") for c in final_df_c.columns if c.startswith("Exit_")]
        return_candidates = [c.replace("Return_","") for c in final_df_c.columns if c.startswith("Return_")]
        if exit_candidates and return_candidates:
            colA, colB = st.columns(2)
            with colA:
                ex_dim = st.selectbox(
                    "Exit Dimension",
                    exit_candidates,
                    index=exit_candidates.index("ProjectTypeCode") if "ProjectTypeCode" in exit_candidates else 0
                )
            with colB:
                ret_dim = st.selectbox(
                    "Return Dimension",
                    return_candidates,
                    index=return_candidates.index("ProjectTypeCode") if "ProjectTypeCode" in return_candidates else 0
                )

            pivot_df = create_flow_pivot(final_df_c, ex_dim, ret_dim)
            if pivot_df.empty:
                st.info("No next enrollments found to build flow pivot.")
            else:
                st.markdown("#### Flow Matrix")
                st.dataframe(
                    pivot_df.style.background_gradient(cmap="Blues").format(precision=0),
                    use_container_width=True
                )

                top_n = st.slider("Top N Flows", 5, 25, 10)
                top_flows = get_top_flows_from_pivot(pivot_df, top_n=top_n)
                if not top_flows.empty:
                    st.markdown("#### Top Flows")
                    st.dataframe(
                        top_flows.style.format({"Percent": "{:.1f}%"}).background_gradient(cmap="Blues"),
                        use_container_width=True
                    )
                sankey_fig = plot_flow_sankey(pivot_df, f"{ex_dim} → {ret_dim}")
                st.plotly_chart(sankey_fig, use_container_width=True)

        st.divider()
        st.markdown("### PH vs. Non-PH Exits Comparison")
        comp_check = st.checkbox("Compare PH vs. Non-PH Exits", value=False)
        if comp_check:
            ph_df = final_df_c[final_df_c["PH_Exit"] == True]
            nonph_df = final_df_c[final_df_c["PH_Exit"] == False]
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("PH Exits")
                if not ph_df.empty:
                    ph_metrics = compute_summary_metrics(ph_df)
                    display_spm_metrics(ph_metrics)
                else:
                    st.info("No PH Exits found.")
            with c2:
                st.subheader("Non-PH Exits")
                if not nonph_df.empty:
                    nonph_metrics = compute_summary_metrics(nonph_df)
                    display_spm_metrics(nonph_metrics)
                else:
                    st.info("No Non-PH Exits found.")

        st.divider()
        st.markdown("### 📤 Export Results")
        st.download_button(
            "Download Outbound Recidivism Results (CSV)",
            data=final_df_c.to_csv(index=False),
            file_name="OutboundRecidivismResults.csv",
            mime="text/csv",
            use_container_width=True
        )

if __name__ == "__main__":
    outbound_recidivism_page()