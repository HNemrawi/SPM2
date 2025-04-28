"""
Inbound Recidivism Analysis Page (Enhanced UI/UX)
------------------------------------------------
This module renders a Streamlit page for analyzing inbound recidivism,
focusing on clients returning to homelessness programs after previous exits.
It offers dynamic date filtering, multi-dimensional filters, visual analytics,
and data export options.
"""

import streamlit as st
import pandas as pd
from datetime import datetime

# Import helper functions for analysis and visualization
from inbound_helpers import (
    run_return_analysis,
    display_return_metrics_cards,
    plot_time_to_entry_box,
    return_breakdown_analysis,
    create_flow_pivot_ra,
    plot_flow_sankey_ra
)
from spm2_helpers import get_top_flows_from_pivot


def inbound_recidivism_page():
    """
    Render the Inbound Recidivism Analysis page.
    
    Users can configure filters, run the analysis, view visualizations (including
    a detailed flow matrix), and export results.
    """
    st.header("📈 Inbound Recidivism Analysis")
    
    # About section with overview and methodology
    with st.expander("📘 About Inbound Recidivism Analysis", expanded=False):
        st.markdown("""
            **Inbound Recidivism Analysis Overview**  
            *Evaluate client returns to homelessness programs after prior exits.*
            
            ### Key Features:
            - **Customizable Date Range:** Uses a default entry period with a 24-month (730-day) lookback.
            - **Flexible Filters:** Adjust by CoC codes, local CoC, agencies, programs, and project types.
            - **Return Classification:**
            - 🆕 **New Clients:** No exit found within the lookback.
            - 🔄 **Returning Clients:** Prior exits detected.
            - 🏠 **Returning from Housing:** Clients with exits to permanent housing.
            - **Interactive Visuals:** Charts, flow matrix, and network diagrams.
            
            ### Methodology Highlights:
            1. **Entry Identification:** Select first new entries within the chosen date range.
            2. **Exit Lookup:** Scan client history for exits within the Days lookback.
            3. **Classification Logic:** Categorize based on previous exits.
        """)
        st.divider()
        st.markdown("""
            **Interpretation Guide:**
            - **Metrics:** Counts and percentages for New, Returning, and Returning from Housing.
            - **Timing Analysis:** Box plots display days-to-entry.
            - **Flow Analysis:** Visual mapping of exit-to-entry pathways.
        """)


    # Verify that data has been uploaded
    try:
        df = st.session_state.get("df")
        if df is None or df.empty:
            st.info("📭 Please upload data in the sidebar first.")
            return
    except Exception as e:
        st.error(f"🚨 Data Error: {str(e)}")
        return

    # Sidebar: Analysis parameters and filters
    st.sidebar.header("⚙️ Analysis Parameters", divider="gray")
    
    # Date range selection
    with st.sidebar.expander("🗓️ Entry Date & Lookback", expanded=True):
        try:
            date_range = st.date_input(
                "Entry Date Range",
                [datetime(2025, 1, 1), datetime(2025, 1, 31)],
                help="Analysis period for new entries"
            )
            if len(date_range) != 2:
                st.error("⚠️ Please select both dates")
                return
            report_start, report_end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        except Exception as e:
            st.error(f"📅 Date Error: {str(e)}")
            return

        days_lookback = st.number_input(
            "🔍 Days Lookback",
            min_value=1,
            value=730,
            help="Number of days prior to entry to consider exits"
        )
        analysis_start = report_start - pd.Timedelta(days=days_lookback)
        analysis_end = report_end
        
        data_reporting_start = pd.to_datetime(df["ReportingPeriodStartDate"].iloc[0])
        data_reporting_end = pd.to_datetime(df["ReportingPeriodEndDate"].iloc[0])
        

        if analysis_start < data_reporting_start or analysis_end > data_reporting_end:
            st.warning(
                f"WARNING: Your selected analysis range ({analysis_start:%Y-%m-%d} to {analysis_end:%Y-%m-%d}) "
                f"is outside the data's available range ({data_reporting_start:%Y-%m-%d} to {data_reporting_end:%Y-%m-%d}). "
                "This may result in missing data. Please adjust your Reporting Period or Lookback period accordingly."
            )


    with st.sidebar.expander("📍 Entry Filters", expanded=True):
        allowed_cocs = allowed_localcocs = allowed_agencies = allowed_programs = entry_project_types = None

        if "ProgramSetupCoC" in df.columns:
            all_cocs = sorted(df["ProgramSetupCoC"].dropna().unique().tolist())
            co_selection = st.multiselect(
                "CoC Codes - Entry",
                ["ALL"] + all_cocs,
                default=["ALL"]
            )
            if co_selection and "ALL" not in co_selection:
                allowed_cocs = co_selection

        if "LocalCoCCode" in df.columns:
            all_localcocs = sorted(df["LocalCoCCode"].dropna().unique().tolist())
            lc_selection = st.multiselect(
                "Local CoC Codes - Entry",
                ["ALL"] + all_localcocs,
                default=["ALL"]
            )
            if lc_selection and "ALL" not in lc_selection:
                allowed_localcocs = lc_selection

        if "AgencyName" in df.columns:
            all_agencies = sorted(df["AgencyName"].dropna().unique().tolist())
            ag_selection = st.multiselect(
                "Agencies - Entry",
                ["ALL"] + all_agencies,
                default=["ALL"]
            )
            if ag_selection and "ALL" not in ag_selection:
                allowed_agencies = ag_selection

        if "ProgramName" in df.columns:
            all_programs = sorted(df["ProgramName"].dropna().unique().tolist())
            pr_selection = st.multiselect(
                "Programs - Entry",
                ["ALL"] + all_programs,
                default=["ALL"]
            )
            if pr_selection and "ALL" not in pr_selection:
                allowed_programs = pr_selection

        # Entry Project Types filter
        if "ProjectTypeCode" in df.columns:
            all_proj_types = sorted(df["ProjectTypeCode"].dropna().unique().tolist())
            entry_sel = st.multiselect(
                "Project Types - Entry",
                ["ALL"] + all_proj_types,
                default=["ALL"]
            )
            if entry_sel and "ALL" not in entry_sel:
                entry_project_types = entry_sel


    with st.sidebar.expander("🚪 Exit Filters", expanded=True):
        allowed_cocs_exit = allowed_localcocs_exit = allowed_agencies_exit = allowed_programs_exit = exit_project_types = None

        # CoC Codes
        if "ProgramSetupCoC" in df.columns:
            all_cocs = sorted(df["ProgramSetupCoC"].dropna().unique().tolist())
            co_exit = st.multiselect(
                "CoC Codes - Exit",
                ["ALL"] + all_cocs,
                default=["ALL"]
            )
            if co_exit and "ALL" not in co_exit:
                allowed_cocs_exit = co_exit

        # Local CoC Codes
        if "LocalCoCCode" in df.columns:
            all_localcocs = sorted(df["LocalCoCCode"].dropna().unique().tolist())
            lc_exit = st.multiselect(
                "Local CoC Codes - Exit",
                ["ALL"] + all_localcocs,
                default=["ALL"]
            )
            if lc_exit and "ALL" not in lc_exit:
                allowed_localcocs_exit = lc_exit

        # Agencies
        if "AgencyName" in df.columns:
            all_agencies = sorted(df["AgencyName"].dropna().unique().tolist())
            ag_exit = st.multiselect(
                "Agencies - Exit",
                ["ALL"] + all_agencies,
                default=["ALL"]
            )
            if ag_exit and "ALL" not in ag_exit:
                allowed_agencies_exit = ag_exit

        # Programs
        if "ProgramName" in df.columns:
            all_programs = sorted(df["ProgramName"].dropna().unique().tolist())
            pr_exit = st.multiselect(
                "Programs - Exit",
                ["ALL"] + all_programs,
                default=["ALL"]
            )
            if pr_exit and "ALL" not in pr_exit:
                allowed_programs_exit = pr_exit

        # Project Types
        if "ProjectTypeCode" in df.columns:
            all_proj_types = sorted(df["ProjectTypeCode"].dropna().unique().tolist())
            exit_sel = st.multiselect(
                "Project Types - Exit",
                ["ALL"] + all_proj_types,
                default=["ALL"]
            )
            if exit_sel and "ALL" not in exit_sel:
                exit_project_types = exit_sel


    # Run the analysis when the button is clicked
    st.divider()
    if st.button("▶️ Run Inbound Analysis", type="primary", use_container_width=True):
        try:
            with st.status("🔍 Processing...", expanded=True) as status:
                merged_df = run_return_analysis(
                    df,
                    report_start=report_start,
                    report_end=report_end,
                    days_lookback=days_lookback,
                    allowed_cocs=allowed_cocs,
                    allowed_localcocs=allowed_localcocs,
                    allowed_programs=allowed_programs,
                    allowed_agencies=allowed_agencies,
                    entry_project_types=entry_project_types,
                    allowed_cocs_exit=allowed_cocs_exit,
                    allowed_localcocs_exit=allowed_localcocs_exit,
                    allowed_programs_exit=allowed_programs_exit,
                    allowed_agencies_exit=allowed_agencies_exit,
                    exit_project_types=exit_project_types
                )

                cols_to_remove = [
                    "Exit_UniqueIdentifier",	
                    "Exit_ClientID",
                    "Exit_RaceEthnicity",	
                    "Exit_Gender",	
                    "Exit_DOB",	
                    "Exit_VeteranStatus",
                    "Enter_ReportingPeriodStartDate",
                    "Enter_ReportingPeriodEndDate",
                    "Exit_ReportingPeriodStartDate",
                    "Exit_ReportingPeriodEndDate"
                ]
                merged_df.drop(columns=cols_to_remove, inplace=True, errors='ignore')

                cols_to_rename = [
                    "Enter_UniqueIdentifier", 
                    "Enter_ClientID",
                    "Enter_RaceEthnicity",
                    "Enter_Gender",
                    "Enter_DOB",
                    "Enter_VeteranStatus"
                ]
                mapping = {col: col[len("Enter_"):] for col in cols_to_rename if col in merged_df.columns}
                merged_df.rename(columns=mapping, inplace=True)
                
                st.session_state["return_df"] = merged_df
                status.update(label="✅ Analysis Complete!", state="complete")
            st.toast("Analysis completed successfully!", icon="🎉")
        except Exception as e:
            st.error(f"🚨 Analysis Error: {str(e)}")

    # Display analysis results if available
    if "return_df" in st.session_state and not st.session_state["return_df"].empty:
        final_ret_df = st.session_state["return_df"]
        
        st.divider()
        st.markdown("### 📊 Inbound Analysis Summary")
        display_return_metrics_cards(final_ret_df)
        
        st.divider()
        st.markdown("### ⏳ Days to Return Distribution")
        try:
            st.plotly_chart(plot_time_to_entry_box(final_ret_df), use_container_width=True)
        except Exception as e:
            st.error(f"📉 Visualization Error: {str(e)}")

        st.divider()
        st.markdown("### 📈 Breakdown")
        breakdown_columns = [
            "RaceEthnicity",
            "Gender",
            "VeteranStatus",
            "Enter_HasIncome",
            "Enter_HasDisability",
            "Enter_HouseholdType",
            "Enter_CHStartHousehold",
            "Enter_LocalCoCCode",
            "Enter_PriorLivingCat",
            "Enter_ProgramSetupCoC",
            "Enter_ProjectTypeCode",
            "Enter_AgencyName",
            "Enter_ProgramName",
            "Enter_ExitDestinationCat",
            "Enter_ExitDestination",
            "Enter_ProgramsContinuumProject",
            "Enter_AgeTieratEntry",
            "ReturnCategory",
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
            "Exit_ProgramsContinuumProject",
            "Exit_AgeTieratEntry",
        ]

        possible_cols = [col for col in breakdown_columns if col in final_ret_df.columns]
        default_breakdown = ["Exit_ProjectTypeCode"] if "Exit_ProjectTypeCode" in possible_cols else []
        chosen = st.multiselect(
            "Group By",
            possible_cols,
            default=default_breakdown,
            help="Select grouping columns"
        )
        if chosen:
            try:
                breakdown = return_breakdown_analysis(final_ret_df, chosen)

                # if any chosen column starts with "exit_", drop the "New (%)" column
                if any(col.lower().startswith("exit_") for col in chosen):
                    breakdown = breakdown.drop(columns=["New (%)"], errors="ignore")

                st.dataframe(
                    breakdown.style.format(thousands=",")
                        .background_gradient(cmap="Blues", subset=["Total Entries"])
                        .set_properties(**{'text-align': 'left'}),
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"📊 Breakdown Error: {str(e)}")


        st.divider()
        st.markdown("### 🌊 Client Flow Analysis")
        try:
            ra_flows_df = (
                final_ret_df[final_ret_df["ReturnCategory"].str.contains("Returning")]
                if "ReturnCategory" in final_ret_df.columns
                else pd.DataFrame()
            )

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
                "Exit_ProgramsContinuumProject",
                "Exit_AgeTieratEntry",
            ]
            entry_columns = [
                "Enter_HasIncome",
                "Enter_HasDisability",
                "Enter_HouseholdType",
                "Enter_CHStartHousehold",
                "Enter_LocalCoCCode",
                "Enter_PriorLivingCat",
                "Enter_ProgramSetupCoC",
                "Enter_ProjectTypeCode",
                "Enter_AgencyName",
                "Enter_ProgramName",
                "Enter_ExitDestinationCat",
                "Enter_ExitDestination",
                "Enter_ProgramsContinuumProject",
                "Enter_AgeTieratEntry",
            ]

            exit_cols_for_flow  = [c for c in exit_columns  if c in ra_flows_df.columns]
            entry_cols_for_flow = [c for c in entry_columns if c in ra_flows_df.columns]

            if exit_cols_for_flow and entry_cols_for_flow:
                # Pick your two dimensions
                flow_cols = st.columns(2)
                with flow_cols[0]:
                    exit_flow_col = st.selectbox(
                        "Exit Dimension: Rows",
                        exit_cols_for_flow,
                        index=exit_cols_for_flow.index("Exit_ProjectTypeCode") if "Exit_ProjectTypeCode" in exit_cols_for_flow else 0
                    )
                with flow_cols[1]:
                    entry_flow_col = st.selectbox(
                        "Entry Dimension: Columns",
                        entry_cols_for_flow,
                        index=entry_cols_for_flow.index("Enter_ProjectTypeCode") if "Enter_ProjectTypeCode" in entry_cols_for_flow else 0
                    )

                # --- build full pivot ---
                flow_pivot_ra = create_flow_pivot_ra(ra_flows_df, exit_flow_col, entry_flow_col)

                # --- drill-in controls ---
                drill_cols = st.columns(2)
                with drill_cols[0]:
                    focus_exit   = st.selectbox(
                        "🔍 Focus Exit Dimension",
                        ["All"] + flow_pivot_ra.index.tolist(),
                        help="Show only this exit in the flow"
                    )
                with drill_cols[1]:
                    focus_return = st.selectbox(
                        "🔍 Focus Return Dimension",
                        ["All"] + flow_pivot_ra.columns.tolist(),
                        help="Show only this return in the flow"
                    )

                # subset the pivot in place
                if focus_exit != "All":
                    flow_pivot_ra = flow_pivot_ra.loc[[focus_exit]]
                if focus_return != "All":
                    flow_pivot_ra = flow_pivot_ra[[focus_return]]

                # if there’s a “No Data” or “No Return” column, push it to the end
                if "No Data" in flow_pivot_ra.columns:
                    cols = [c for c in flow_pivot_ra.columns if c != "No Data"] + ["No Data"]
                    flow_pivot_ra = flow_pivot_ra[cols]

                # show the matrix
                with st.expander("🔍 Flow Matrix Details", expanded=True):
                    st.dataframe(
                        flow_pivot_ra
                            .style
                            .background_gradient(cmap="Blues")
                            .format(precision=0),
                        use_container_width=True
                    )

                # top pathways
                st.markdown("#### 🔝 Top Client Pathways")
                top_n = st.slider("Number of Flows", 5, 25, 10)
                top_flows_df = get_top_flows_from_pivot(flow_pivot_ra, top_n=top_n)
                if not top_flows_df.empty:
                    if "Percent" in top_flows_df.columns:
                        top_flows_df["Percent"] = top_flows_df["Percent"].astype(float)
                        st.dataframe(
                            top_flows_df
                                .style
                                .format({'Percent': '{:.1f}%'})
                                .background_gradient(cmap="Blues", subset=["Count", "Percent"]),
                            use_container_width=True
                        )
                    else:
                        st.dataframe(top_flows_df, use_container_width=True)
                else:
                    st.info("No significant flows detected")

                # sankey
                st.markdown("#### 🌐 Client Flow Network")
                sankey_ra = plot_flow_sankey_ra(flow_pivot_ra, f"{exit_flow_col} → {entry_flow_col}")
                st.plotly_chart(sankey_ra, use_container_width=True)

            else:
                st.info("📭 Insufficient data for flow analysis")
        except Exception as e:
            st.error(f"🌊 Flow Error: {str(e)}")

        st.divider()
        st.markdown("### 📤 Data Export")
        st.download_button(
            label="📥 Download Inbound Data",
            data=final_ret_df.to_csv(index=False),
            file_name="recidivism_analysis.csv",
            mime="text/csv",
            use_container_width=True
        )


if __name__ == "__main__":
    inbound_recidivism_page()
