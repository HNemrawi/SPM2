from typing import Any, Dict, Optional, Set, Tuple

import pandas as pd
import streamlit as st
from pandas import DataFrame, Timestamp

from analysis.general.data_utils import (
    calc_delta, households_served, inflow, outflow, 
    ph_exit_clients, ph_exit_rate, return_after_exit, served_clients,
    period_comparison  # Add this import
)
from analysis.general.filter_utils import init_section_state
from analysis.general.theme import (
    MAIN_COLOR, SECONDARY_COLOR, SUCCESS_COLOR, WARNING_COLOR, NEUTRAL_COLOR,
    fmt_int, fmt_pct, blue_divider
)

# Constants
SUMMARY_SECTION_KEY = "summary_metrics"

@st.cache_data(show_spinner=False)
def _get_summary_metrics(
    df_filt: DataFrame, 
    full_df: DataFrame,
    t0: Timestamp,
    t1: Timestamp,
    prev_start: Timestamp,
    prev_end: Timestamp,
    return_window: int = 180
) -> Dict[str, Any]:
    """
    Calculate summary metrics for both current and previous time periods.
    
    Parameters:
    -----------
    df_filt : DataFrame
        Filtered DataFrame
    full_df : DataFrame
        Full DataFrame (MUST be the complete unfiltered dataset for accurate returns tracking)
    t0 : Timestamp
        Current period start
    t1 : Timestamp
        Current period end
    prev_start : Timestamp
        Previous period start
    prev_end : Timestamp
        Previous period end
    return_window : int, optional
        Days to check for returns
        
    Returns:
    --------
    dict
        Dictionary of metrics for current and previous periods
    """
    # Current period metrics
    served_ids = served_clients(df_filt, t0, t1)
    inflow_ids = inflow(df_filt, t0, t1)
    outflow_ids = outflow(df_filt, t0, t1)
    ph_ids = ph_exit_clients(df_filt, t0, t1)
    
    # First, identify the PH exits
    ph_exits_mask = (
        (df_filt["ProjectExit"].between(t0, t1))
        & (df_filt["ExitDestinationCat"] == "Permanent Housing Situations")
    )
    ph_exits_df = df_filt[ph_exits_mask]
    ph_exits_in_period = set(ph_exits_df["ClientID"].unique())

    # Then pass ONLY those PH exits to return_after_exit
    # This ensures it only checks returns for the clients we're using in the denominator
    return_ids = return_after_exit(ph_exits_df, full_df, t0, t1, return_window)

    
    # Previous period metrics
    served_prev = served_clients(df_filt, prev_start, prev_end)
    inflow_prev = inflow(df_filt, prev_start, prev_end)
    outflow_prev = outflow(df_filt, prev_start, prev_end)
    ph_prev = ph_exit_clients(df_filt, prev_start, prev_end)
    
    # Get PH exits for previous period return rate
    ph_exits_mask_prev = (
        (df_filt["ProjectExit"].between(prev_start, prev_end))
        & (df_filt["ExitDestinationCat"] == "Permanent Housing Situations")
    )
    ph_exits_df_prev = df_filt[ph_exits_mask_prev]
    ph_exits_prev = set(ph_exits_df_prev["ClientID"].unique())
    return_ids_prev = return_after_exit(ph_exits_df_prev, full_df, prev_start, prev_end, return_window)
    
    # ADD PERIOD COMPARISON
    period_comp = period_comparison(df_filt, t0, t1, prev_start, prev_end)
    
    return {
        "served_ids": served_ids,
        "inflow_ids": inflow_ids,
        "outflow_ids": outflow_ids,
        "ph_ids": ph_ids,
        "ph_exits_in_period": ph_exits_in_period,
        "return_ids": return_ids,
        "return_window": return_window,
        "served_prev": served_prev,
        "inflow_prev": inflow_prev,
        "outflow_prev": outflow_prev,
        "ph_prev": ph_prev,
        "ph_exits_prev": ph_exits_prev,
        "return_ids_prev": return_ids_prev,
        "households_current": households_served(df_filt, t0, t1),
        "households_prev": households_served(df_filt, prev_start, prev_end),
        "period_comparison": period_comp  # ADD THIS
    }

def _get_metric_help_text(metric_name: str, is_filtered: bool) -> str:
    """Get appropriate help text based on metric and filter status."""
    help_texts = {
        "Households Served": "Total households (heads of household) active during the period",
        "Clients Served": "Unique clients with an active enrollment during the period",
        "Inflow": {
            True: "Clients entering filtered programs who weren't in any FILTERED programs the day before",
            False: "Clients entering the system who weren't in any programs the day before"
        },
        "Outflow": {
            True: "Clients exiting filtered programs who aren't in any FILTERED programs at period end",
            False: "Clients leaving the system who aren't in any programs at period end"
        },
        "PH Exits": "Clients exiting to permanent housing destinations",
        "PH Exit Rate": "Percentage of ALL exits that went to permanent housing",
        "Returns to Homelessness": "PH exits who returned to ANY homeless program (tracked system-wide)",
        "Return Rate": "Percentage of PH exits who return to homelessness",
        "Net Flow": "Difference between inflow and outflow (positive = growth)"
    }
    
    text = help_texts.get(metric_name, "")
    if isinstance(text, dict):
        return text.get(is_filtered, text.get(False, ""))
    return text

def _generate_flow_insight_html(inflow_count: int, outflow_count: int, net_flow: int) -> str:
    """Generate HTML for system flow insight."""
    if net_flow > 0:
        icon = "📈"
        status = "Growing"
        color = SUCCESS_COLOR
        description = f"More clients entering ({inflow_count:,}) than leaving ({outflow_count:,})"
    elif net_flow < 0:
        icon = "📉"
        status = "Reducing"
        color = WARNING_COLOR
        description = f"More clients leaving ({outflow_count:,}) than entering ({inflow_count:,})"
    else:
        icon = "➡️"
        status = "Balanced"
        color = NEUTRAL_COLOR
        description = f"Equal number entering and leaving ({inflow_count:,} each)"
    
    return f"""
    <div style="background-color: rgba(0,0,0,0.3); border: 2px solid {color}; 
                border-radius: 10px; padding: 20px; margin: 10px 0;">
        <h3 style="color: {color}; margin: 0 0 10px 0;">
            {icon} System Flow: {status}
        </h3>
        <p style="margin: 0; font-size: 16px;">
            {description}<br>
            <strong style="font-size: 20px;">Net change: {net_flow:+,} clients</strong>
        </p>
    </div>
    """

def _generate_housing_outcomes_html(ph_rate: float, return_rate: float, return_window: int) -> str:
    """Generate HTML for housing outcomes insights."""
    # PH Exit Rate Assessment
    if ph_rate >= 50:
        ph_icon = "🏆"
        ph_status = "Excellent"
        ph_color = SUCCESS_COLOR
        ph_desc = "More than half of all exits are to permanent housing"
    elif ph_rate >= 35:
        ph_icon = "✅"
        ph_status = "Good"
        ph_color = MAIN_COLOR
        ph_desc = "Solid performance in housing placements"
    elif ph_rate >= 20:
        ph_icon = "⚠️"
        ph_status = "Needs Improvement"
        ph_color = WARNING_COLOR
        ph_desc = "Below typical performance benchmarks"
    else:
        ph_icon = "❌"
        ph_status = "Critical"
        ph_color = SECONDARY_COLOR
        ph_desc = "Significant challenges in achieving housing exits"
    
    # Return Rate Assessment
    if return_rate <= 5:
        ret_icon = "🌟"
        ret_status = "Outstanding"
        ret_color = SUCCESS_COLOR
        ret_desc = "Exceptional housing stability"
    elif return_rate <= 10:
        ret_icon = "✅"
        ret_status = "Strong"
        ret_color = MAIN_COLOR
        ret_desc = "Good housing retention"
    elif return_rate <= 20:
        ret_icon = "⚠️"
        ret_status = "Moderate"
        ret_color = WARNING_COLOR
        ret_desc = "Some stability challenges"
    else:
        ret_icon = "🚨"
        ret_status = "High"
        ret_color = SECONDARY_COLOR
        ret_desc = "Significant housing stability issues"
    
    return f"""
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0;">
        
        <div style="background-color: rgba(0,0,0,0.3); border: 2px solid {ph_color}; 
                    border-radius: 10px; padding: 20px;">
            <h4 style="color: {ph_color}; margin: 0 0 10px 0;">
                {ph_icon} Housing Placement: {ph_status}
            </h4>
            <p style="margin: 0;">
                <strong style="font-size: 24px;">{ph_rate:.1f}%</strong> exit to permanent housing<br>
                <span style="font-size: 14px; color: #ccc;">{ph_desc}</span>
            </p>
        </div>
        
        <div style="background-color: rgba(0,0,0,0.3); border: 2px solid {ret_color}; 
                    border-radius: 10px; padding: 20px;">
            <h4 style="color: {ret_color}; margin: 0 0 10px 0;">
                {ret_icon} Housing Stability: {ret_status}
            </h4>
            <p style="margin: 0;">
                <strong style="font-size: 24px;">{return_rate:.1f}%</strong> return within {return_window} days<br>
                <span style="font-size: 14px; color: #ccc;">{ret_desc}</span>
            </p>
        </div>
        
    </div>
    """

@st.fragment
def render_summary_metrics(df_filt: DataFrame, full_df: Optional[DataFrame] = None) -> None:
    """
    Render summary metrics with key performance indicators comparing
    the current window with the previous period.
    
    Parameters:
    -----------
    df_filt : DataFrame
        Filtered DataFrame
    full_df : DataFrame, optional
        Full DataFrame for returns analysis
    """
    try:
        # CRITICAL: Ensure we always have the true unfiltered dataset for returns
        if full_df is None:
            # Get the original unfiltered data from session state
            full_df = st.session_state.get("df")
            if full_df is None:
                st.error("Original dataset not found. Please reload your data.")
                return

        # Initialize or retrieve cached section state
        state: Dict[str, Any] = init_section_state(SUMMARY_SECTION_KEY)
        filter_timestamp = st.session_state.get("last_filter_change", "")
        cache_valid = state.get("last_updated") == filter_timestamp

        if not cache_valid:
            state["last_updated"] = filter_timestamp

        # Header with info button
        col_header, col_info = st.columns([6, 1])
        with col_header:
            st.subheader("📊 Summary Metrics")
        with col_info:
            with st.popover("ℹ️ Help", use_container_width=True):
                st.markdown("""
                ### Understanding Summary Metrics
                
                **System Flow Metrics:**
                - **Inflow**: Clients entering programs who weren't in any programs the day before the reporting period started
                - **Outflow**: Clients exiting programs who aren't in any programs on the last day of the reporting period
                - **Net Flow**: Inflow minus Outflow (positive = system growth, negative = system reduction)
                
                **Housing Outcomes:**
                - **PH Exits**: Unique clients who exited to permanent housing destinations during the period
                - **PH Exit Rate**: Percentage of unique clients who exited that went to permanent housing (unique PH exit clients ÷ unique clients with any exit × 100)
                - **Returns**: Clients who exited to PH and returned to homelessness within the specified tracking window
                - **Return Rate**: Percentage of PH exits who returned (returns ÷ PH exits × 100)
                
                **Population Analysis:**
                - **Households**: Count of heads of household only (not all family members)
                - **Clients Served**: Unique clients with active enrollment anytime during the period
                - **Carryover**: Clients active in both current and previous periods
                - **New**: Clients in current period who weren't in previous period
                
                **Time Comparisons:**
                - Current vs previous period (can be same length or custom)
                - Percentages show relative change from previous period
                
                **Important Notes:**
                - When filters are active, metrics reflect filtered programs only
                - Returns are ALWAYS tracked system-wide regardless of filters
                - Each client is counted only once per metric, even with multiple enrollments
                """)
        
        # Time boundaries
        t0: Timestamp = st.session_state.get("t0")
        t1: Timestamp = st.session_state.get("t1")
        prev_start: Timestamp = st.session_state.get("prev_start")
        prev_end: Timestamp = st.session_state.get("prev_end")

        if not all([t0, t1, prev_start, prev_end]):
            st.warning("⏰ Please set date ranges in the filter panel to view metrics.")
            return

        # Calculate period lengths
        current_days = (t1 - t0).days + 1
        previous_days = (prev_end - prev_start).days + 1

        # Compute metrics if cache is invalid
        if not cache_valid:
            with st.spinner("Calculating key metrics..."):
                return_window = 180  # Default return window in days
                
                # Calculate all metrics using the cached function
                metrics = _get_summary_metrics(
                    df_filt, full_df, t0, t1, prev_start, prev_end, return_window
                )
                
                # Update state with calculated metrics
                state.update(metrics)

        # Retrieve cached metrics
        served_ids = state.get("served_ids", set())
        inflow_ids = state.get("inflow_ids", set())
        outflow_ids = state.get("outflow_ids", set())
        ph_ids = state.get("ph_ids", set())
        ph_exits_in_period = state.get("ph_exits_in_period", set())
        return_ids = state.get("return_ids", set())
        return_window = state.get("return_window", 180)

        served_prev = state.get("served_prev", set())
        inflow_prev = state.get("inflow_prev", set())
        outflow_prev = state.get("outflow_prev", set())
        ph_prev = state.get("ph_prev", set())
        ph_exits_prev = state.get("ph_exits_prev", set())
        return_ids_prev = state.get("return_ids_prev", set())
        
        households_current = state.get("households_current", 0)
        households_prev = state.get("households_prev", 0)
        
        # GET PERIOD COMPARISON DATA
        period_comp = state.get("period_comparison", {})

        # Compute rates
        ph_rate = ph_exit_rate(df_filt, t0, t1)
        ph_rate_prev = ph_exit_rate(df_filt, prev_start, prev_end)
        
        # Calculate return rates with safe division
        return_rate = 0 if not ph_exits_in_period else (len(return_ids) / len(ph_exits_in_period) * 100)
        return_rate_prev = 0 if not ph_exits_prev else (len(return_ids_prev) / len(ph_exits_prev) * 100)

        # Check if filters are active
        active_filters = st.session_state.get("filters", {})
        is_filtered = any(active_filters.values())
        
        # Display filter context if active
        if is_filtered:
            # Count and describe active filters
            filter_details = []
            for name, values in active_filters.items():
                if values:
                    filter_details.append(f"**{name}** ({len(values)} selected)")
            
            filter_context_html = f"""
            <div style="background-color: rgba(255,165,0,0.1); border: 2px solid {WARNING_COLOR}; 
                        border-radius: 10px; padding: 15px; margin: 15px 0;">
                <h4 style="color: {WARNING_COLOR}; margin: 0 0 10px 0;">
                    🔍 Filtered View Active
                </h4>
                <p style="margin: 0 0 10px 0;">
                    Metrics reflect filtered data only. Active filters: {", ".join(filter_details)}
                </p>
                <details style="margin-top: 10px;">
                    <summary style="cursor: pointer; color: {WARNING_COLOR};">
                        <strong>How filters affect metrics (click to expand)</strong>
                    </summary>
                    <div style="margin-top: 10px; padding: 10px; background-color: rgba(0,0,0,0.3); border-radius: 5px;">
                        <ul style="margin: 0; padding-left: 20px;">
                            <li><strong>Inflow/Outflow</strong>: Only tracks movement within filtered programs</li>
                            <li><strong>PH Exit Rate</strong>: Based on exits from filtered programs only</li>
                            <li><strong>Returns</strong>: Tracks returns to ANY program (system-wide) ✓</li>
                        </ul>
                        <p style="margin: 10px 0 0 0; font-style: italic;">
                            For true system-wide metrics, remove all filters.
                        </p>
                    </div>
                </details>
            </div>
            """
            st.html(filter_context_html)
        
        # Display period context
        period_context_html = f"""
        <div style="background-color: rgba(0,0,0,0.2); border-radius: 10px; padding: 15px; margin-bottom: 20px;">
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                <div>
                    <strong style="color: {MAIN_COLOR};">📅 Current Period</strong><br>
                    {t0.strftime('%B %d, %Y')} to {t1.strftime('%B %d, %Y')}<br>
                    <span style="color: #999;">({current_days} days)</span>
                </div>
                <div>
                    <strong style="color: {NEUTRAL_COLOR};">📅 Previous Period</strong><br>
                    {prev_start.strftime('%B %d, %Y')} to {prev_end.strftime('%B %d, %Y')}<br>
                    <span style="color: #999;">({previous_days} days)</span>
                </div>
            </div>
        </div>
        """
        st.html(period_context_html)
        
        # Prepare display in rows of metrics
        st.markdown("### 📈 Key Performance Indicators")
        
        row1_cols = st.columns(3)
        row2_cols = st.columns(3)
        row3_cols = st.columns(3)
        
        # Define all metrics to display
        metrics_data = [
            # Row 1
            (row1_cols[0], "Households Served", households_current, households_prev, 
             _get_metric_help_text("Households Served", is_filtered), "neutral"),
            
            (row1_cols[1], "Clients Served", len(served_ids), len(served_prev), 
             _get_metric_help_text("Clients Served", is_filtered), "neutral"),
            
            (row1_cols[2], "Inflow", len(inflow_ids), len(inflow_prev), 
             _get_metric_help_text("Inflow", is_filtered), "neutral"),
            
            # Row 2
            (row2_cols[0], "Outflow", len(outflow_ids), len(outflow_prev), 
             _get_metric_help_text("Outflow", is_filtered), "neutral"),
            
            (row2_cols[1], "PH Exits", len(ph_ids), len(ph_prev), 
             _get_metric_help_text("PH Exits", is_filtered), "neutral"),
            
            (row2_cols[2], "PH Exit Rate", ph_rate, ph_rate_prev, 
             _get_metric_help_text("PH Exit Rate", is_filtered), "normal"),
            
            # Row 3
            (row3_cols[0], f"Returns ({return_window}d)", len(return_ids), len(return_ids_prev), 
             _get_metric_help_text("Returns to Homelessness", is_filtered), "neutral"),
            
            (row3_cols[1], "Return Rate", return_rate, return_rate_prev, 
             _get_metric_help_text("Return Rate", is_filtered), "inverse"),
            
            (row3_cols[2], "Net Flow", len(inflow_ids) - len(outflow_ids), 
             len(inflow_prev) - len(outflow_prev), 
             _get_metric_help_text("Net Flow", is_filtered), "neutral")
        ]
        
        # Display each metric with filter indicators
        for col, label, current, previous, help_text, direction in metrics_data:
            # Add filter indicator to label if needed
            display_label = label
            if is_filtered and label in ["Inflow", "Outflow", "Net Flow"]:
                display_label = f"{label} 🔍"
            
            # Handle rate metrics differently (showing percentage points change)
            if "Rate" in label:
                # For rates, show absolute pp change
                delta = current - previous
                delta_display = f"{delta:+.1f} pp" if previous is not None else "n/a"
                display_value = fmt_pct(current)
                
                # Set color based on metric type and direction
                if direction == "normal":  # PH Exit Rate - higher is better
                    delta_color = "normal"
                elif direction == "inverse":  # Return Rate - lower is better
                    delta_color = "inverse"
                else:
                    delta_color = "off"  # Neutral
            else:
                # For counts, show both absolute and percentage change
                if previous:
                    delta, pct = calc_delta(current, previous)
                    # Net Flow doesn't show percentage
                    if label == "Net Flow":
                        delta_display = fmt_int(delta)
                    else:
                        delta_display = f"{fmt_int(delta)} ({fmt_pct(pct)})"
                else:
                    delta_display = "n/a"
                
                display_value = fmt_int(current)
                
                # All non-rate metrics use neutral color
                delta_color = "off"
            
            # Display the metric
            col.metric(
                display_label,
                display_value,
                delta_display,
                delta_color=delta_color,
                help=help_text,
            )

        # Enhanced System Insights
        blue_divider()
        st.markdown("### 🔍 System Analysis")
        
        # System Flow Analysis
        net_flow = len(inflow_ids) - len(outflow_ids)
        flow_html = _generate_flow_insight_html(len(inflow_ids), len(outflow_ids), net_flow)
        st.html(flow_html)
        
        # ADD CLIENT POPULATION ANALYSIS HERE
        if period_comp:
            # Create visual breakdown
            carryover_pct = (len(period_comp['carryover']) / len(period_comp['current_clients']) * 100) if period_comp['current_clients'] else 0
            new_pct = (len(period_comp['new']) / len(period_comp['current_clients']) * 100) if period_comp['current_clients'] else 0
            exited_pct = (len(period_comp['exited']) / len(period_comp['previous_clients']) * 100) if period_comp['previous_clients'] else 0
            
            period_breakdown_html = f"""
            <div style="background-color: rgba(0,0,0,0.2); border-radius: 10px; padding: 20px; margin: 20px 0;">
                <h3 style="color: {MAIN_COLOR}; margin: 0 0 15px 0;">📊 Client Population Analysis</h3>
                
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin-bottom: 20px;">
                    <div style="background-color: rgba(255,255,255,0.05); padding: 15px; border-radius: 8px;">
                        <h4 style="color: {NEUTRAL_COLOR}; margin: 0;">Previous Period</h4>
                        <h2 style="margin: 10px 0; color: {NEUTRAL_COLOR};">{len(period_comp['previous_clients']):,}</h2>
                        <p style="margin: 0; font-size: 14px;">Total clients</p>
                    </div>
                    
                    <div style="background-color: rgba(255,255,255,0.05); padding: 15px; border-radius: 8px;">
                        <h4 style="color: {MAIN_COLOR}; margin: 0;">Current Period</h4>
                        <h2 style="margin: 10px 0; color: {MAIN_COLOR};">{len(period_comp['current_clients']):,}</h2>
                        <p style="margin: 0; font-size: 14px;">Total clients</p>
                    </div>
                </div>
                
                <div style="background-color: rgba(255,255,255,0.05); padding: 20px; border-radius: 8px;">
                    <h4 style="margin: 0 0 15px 0;">Population Breakdown</h4>
                    
                    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;">
                        <div style="text-align: center;">
                            <div style="font-size: 14px; color: #999; margin-bottom: 5px;">Carryover</div>
                            <div style="font-size: 24px; font-weight: bold; color: {NEUTRAL_COLOR};">
                                {len(period_comp['carryover']):,}
                            </div>
                            <div style="font-size: 14px; color: {NEUTRAL_COLOR};">
                                {carryover_pct:.1f}%
                            </div>
                            <div style="font-size: 12px; color: #999; margin-top: 5px;">
                                Served in both periods
                            </div>
                        </div>
                        
                        <div style="text-align: center;">
                            <div style="font-size: 14px; color: #999; margin-bottom: 5px;">New</div>
                            <div style="font-size: 24px; font-weight: bold; color: {SUCCESS_COLOR};">
                                {len(period_comp['new']):,}
                            </div>
                            <div style="font-size: 14px; color: {SUCCESS_COLOR};">
                                {new_pct:.1f}%
                            </div>
                            <div style="font-size: 12px; color: #999; margin-top: 5px;">
                                Not served in previous period
                            </div>
                        </div>
                        
                        <div style="text-align: center;">
                            <div style="font-size: 14px; color: #999; margin-bottom: 5px;">Exited System</div>
                            <div style="font-size: 24px; font-weight: bold; color: {WARNING_COLOR};">
                                {len(period_comp['exited']):,}
                            </div>
                            <div style="font-size: 14px; color: {WARNING_COLOR};">
                                {exited_pct:.1f}%
                            </div>
                            <div style="font-size: 12px; color: #999; margin-top: 5px;">
                                No longer in system
                            </div>
                        </div>
                    </div>
                </div>
                
                <div style="margin-top: 15px; padding: 15px; background-color: rgba(33,102,172,0.1); 
                            border-left: 4px solid {MAIN_COLOR}; border-radius: 5px;">
                    <p style="margin: 0; font-size: 14px;">
                        <strong>Population Insights:</strong>
            """
            
            # Add specific insights based on the data
            if carryover_pct > 70:
                period_breakdown_html += f"""
                        High carryover rate ({carryover_pct:.0f}%) indicates a stable but potentially stuck population. 
                        Consider enhanced interventions for chronic homelessness.
                """
            elif new_pct > 50:
                period_breakdown_html += f"""
                        High influx of new clients ({new_pct:.0f}%) suggests growing need or improved outreach. 
                        Focus on prevention and diversion strategies.
                """
            else:
                period_breakdown_html += f"""
                        Balanced mix of carryover ({carryover_pct:.0f}%) and new clients ({new_pct:.0f}%). 
                        System shows healthy flow with room for improvement.
                """
            
            period_breakdown_html += """
                    </p>
                </div>
            </div>
            """
            
            st.html(period_breakdown_html)
        
        # Housing Outcomes Analysis
        housing_html = _generate_housing_outcomes_html(ph_rate, return_rate, return_window)
        st.html(housing_html)
        
        # Trend Analysis
        st.markdown("### 📊 Period-over-Period Changes")
        
        # Find most significant changes
        significant_changes = []
        
        # Check each metric for significant changes
        metric_changes = [
            ("Clients Served", len(served_ids), len(served_prev)),
            ("Inflow", len(inflow_ids), len(inflow_prev)),
            ("Outflow", len(outflow_ids), len(outflow_prev)),
            ("PH Exits", len(ph_ids), len(ph_prev))
        ]
        
        for name, current, previous in metric_changes:
            if previous > 0:
                _, pct_change = calc_delta(current, previous)
                if abs(pct_change) >= 10:  # 10% or more change
                    significant_changes.append((name, current, previous, pct_change))
        
        if significant_changes:
            # Sort by absolute percentage change
            significant_changes.sort(key=lambda x: abs(x[3]), reverse=True)
            
            change_html = """
            <div style="background-color: rgba(0,0,0,0.2); border-radius: 10px; padding: 20px; margin: 20px 0;">
                <h4 style="margin: 0 0 15px 0;">Notable Changes from Previous Period:</h4>
                <div style="display: grid; gap: 10px;">
            """
            
            for name, current, previous, pct_change in significant_changes[:3]:  # Show top 3
                if pct_change > 0:
                    icon = "📈"
                    color = SUCCESS_COLOR if name != "Outflow" else WARNING_COLOR
                else:
                    icon = "📉"
                    color = WARNING_COLOR if name != "Outflow" else SUCCESS_COLOR
                
                change_html += f"""
                <div style="display: flex; align-items: center; gap: 15px; padding: 10px; 
                            background-color: rgba(0,0,0,0.3); border-radius: 5px;">
                    <span style="font-size: 24px;">{icon}</span>
                    <div style="flex: 1;">
                        <strong>{name}</strong><br>
                        <span style="color: {color};">{pct_change:+.1f}% change</span>
                    </div>
                    <div style="text-align: right;">
                        <span style="color: #999;">{previous:,} → </span>
                        <strong>{current:,}</strong>
                    </div>
                </div>
                """
            
            change_html += """
                </div>
            </div>
            """
            st.html(change_html)
        else:
            st.info("📊 No significant changes (≥10%) detected between periods.")

    except Exception as e:
        st.error(f"Error calculating summary metrics: {str(e)}")
        st.info("💡 Try refreshing the page or reloading your data.")