"""
Summary Metrics Section.
"""

from typing import Any, Dict, Optional, Set, Tuple

import streamlit as st
from pandas import DataFrame, Timestamp

# Import theme and styling modules
from analysis.general.theme import (
    Colors, fmt_int, fmt_pct, create_insight_container, styled_metric
)
from ui.styling import create_info_box, create_styled_divider, style_metric_cards

# Import data utilities
from analysis.general.data_utils import (
    calc_delta, households_served, inflow, outflow, 
    ph_exit_clients, ph_exit_rate, return_after_exit, served_clients,
    period_comparison
)

# Import filter utilities
from analysis.general.filter_utils import init_section_state
from core.ph_destinations import apply_custom_ph_destinations

# ==================== CONSTANTS ====================

SUMMARY_SECTION_KEY = "summary_metrics"

# Help text definitions
HELP_TEXTS = {
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

# Housing outcome thresholds
HOUSING_THRESHOLDS = {
    "ph_exit_rate": {
        "excellent": {"min": 50, "icon": "🏆", "label": "Excellent", "desc": "More than half of all exits are to permanent housing"},
        "good": {"min": 35, "icon": "✅", "label": "Good", "desc": "Solid performance in housing placements"},
        "needs_improvement": {"min": 20, "icon": "⚠️", "label": "Needs Improvement", "desc": "Below typical performance benchmarks"},
        "critical": {"min": 0, "icon": "❌", "label": "Critical", "desc": "Significant challenges in achieving housing exits"}
    },
    "return_rate": {
        "outstanding": {"max": 5, "icon": "🌟", "label": "Outstanding", "desc": "Exceptional housing stability"},
        "strong": {"max": 10, "icon": "✅", "label": "Strong", "desc": "Good housing retention"},
        "moderate": {"max": 20, "icon": "⚠️", "label": "Moderate", "desc": "Some stability challenges"},
        "high": {"max": 100, "icon": "🚨", "label": "High", "desc": "Significant housing stability issues"}
    }
}

# ==================== CALCULATION FUNCTIONS ====================

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
    """
    # Apply custom PH destinations to both dataframes
    df_filt = apply_custom_ph_destinations(df_filt, force=True)
    full_df = apply_custom_ph_destinations(full_df, force=True)
    
    # Current period metrics
    served_ids = served_clients(df_filt, t0, t1)
    inflow_ids = inflow(df_filt, t0, t1)
    outflow_ids = outflow(df_filt, t0, t1)
    ph_ids = ph_exit_clients(df_filt, t0, t1)
    
    # Get PH exits for return tracking
    ph_exits_mask = (
        (df_filt["ProjectExit"].between(t0, t1))
        & (df_filt["ExitDestinationCat"] == "Permanent Housing Situations")
    )
    ph_exits_df = df_filt[ph_exits_mask]
    ph_exits_in_period = set(ph_exits_df["ClientID"].unique())
    
    # Track returns only for PH exits
    return_ids = return_after_exit(ph_exits_df, full_df, t0, t1, return_window)
    
    # Previous period metrics
    served_prev = served_clients(df_filt, prev_start, prev_end)
    inflow_prev = inflow(df_filt, prev_start, prev_end)
    outflow_prev = outflow(df_filt, prev_start, prev_end)
    ph_prev = ph_exit_clients(df_filt, prev_start, prev_end)
    
    # Get PH exits for previous period
    ph_exits_mask_prev = (
        (df_filt["ProjectExit"].between(prev_start, prev_end))
        & (df_filt["ExitDestinationCat"] == "Permanent Housing Situations")
    )
    ph_exits_df_prev = df_filt[ph_exits_mask_prev]
    ph_exits_prev = set(ph_exits_df_prev["ClientID"].unique())
    return_ids_prev = return_after_exit(ph_exits_df_prev, full_df, prev_start, prev_end, return_window)
    
    # Get period comparison
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
        "period_comparison": period_comp
    }

# ==================== DISPLAY HELPERS ====================

def _get_metric_help_text(metric_name: str, is_filtered: bool) -> str:
    """Get appropriate help text based on metric and filter status."""
    text = HELP_TEXTS.get(metric_name, "")
    if isinstance(text, dict):
        return text.get(is_filtered, text.get(False, ""))
    return text

def _get_housing_outcome_status(value: float, metric_type: str) -> Dict[str, Any]:
    """Get housing outcome assessment based on value and metric type."""
    thresholds = HOUSING_THRESHOLDS.get(metric_type, {})
    
    if metric_type == "ph_exit_rate":
        if value >= thresholds["excellent"]["min"]:
            return {**thresholds["excellent"], "color": Colors.SUCCESS}
        elif value >= thresholds["good"]["min"]:
            return {**thresholds["good"], "color": Colors.PRIMARY}
        elif value >= thresholds["needs_improvement"]["min"]:
            return {**thresholds["needs_improvement"], "color": Colors.WARNING}
        else:
            return {**thresholds["critical"], "color": Colors.DANGER}
    
    elif metric_type == "return_rate":
        if value <= thresholds["outstanding"]["max"]:
            return {**thresholds["outstanding"], "color": Colors.SUCCESS}
        elif value <= thresholds["strong"]["max"]:
            return {**thresholds["strong"], "color": Colors.PRIMARY}
        elif value <= thresholds["moderate"]["max"]:
            return {**thresholds["moderate"], "color": Colors.WARNING}
        else:
            return {**thresholds["high"], "color": Colors.DANGER}
    
    return {}

def _generate_flow_insight_html(inflow_count: int, outflow_count: int, net_flow: int) -> str:
    """Generate HTML for system flow insight with proper theme support."""
    if net_flow > 0:
        icon = "📈"
        status = "Growing"
        description = f"More clients entering ({inflow_count:,}) than leaving ({outflow_count:,})"
        insight_type = "info"
    elif net_flow < 0:
        icon = "📉"
        status = "Reducing"
        description = f"More clients leaving ({outflow_count:,}) than entering ({inflow_count:,})"
        insight_type = "warning"
    else:
        icon = "➡️"
        status = "Balanced"
        description = f"Equal number entering and leaving ({inflow_count:,} each)"
        insight_type = "info"
    
    # Use structured content that will be properly styled by create_info_box
    content = f"""
    <div>
        <strong>{icon} System Flow: {status}</strong><br/>
        {description}<br/>
        <strong>Net change: {net_flow:+,} clients</strong>
    </div>
    """
    
    return create_info_box(content, type=insight_type)

def _generate_housing_outcomes_html(ph_rate: float, return_rate: float, return_window: int) -> str:
    """Generate HTML for housing outcomes insights with proper theme support."""
    ph_status = _get_housing_outcome_status(ph_rate, "ph_exit_rate")
    ret_status = _get_housing_outcome_status(return_rate, "return_rate")
    
    # Use Streamlit columns instead of custom HTML for better theme support
    col1, col2 = st.columns(2)
    
    with col1:
        content = f"""
        <strong>{ph_status['icon']} Housing Placement: {ph_status['label']}</strong><br/>
        <strong style="font-size: 1.5em;">{ph_rate:.1f}%</strong> exit to permanent housing<br/>
        <em>{ph_status['desc']}</em>
        """
        st.html(create_info_box(content, type="info" if ph_rate >= 35 else "warning"))
    
    with col2:
        content = f"""
        <strong>{ret_status['icon']} Housing Stability: {ret_status['label']}</strong><br/>
        <strong style="font-size: 1.5em;">{return_rate:.1f}%</strong> return within {return_window} days<br/>
        <em>{ret_status['desc']}</em>
        """
        st.html(create_info_box(content, type="info" if return_rate <= 10 else "warning"))
    
    # Return empty string since we're using st.columns directly
    return ""

def _render_period_breakdown(period_comp: Dict[str, Any]) -> None:
    """Render client population analysis using native Streamlit components."""
    if not period_comp:
        return
    
    # Calculate percentages
    carryover_pct = (len(period_comp['carryover']) / len(period_comp['current_clients']) * 100) if period_comp['current_clients'] else 0
    new_pct = (len(period_comp['new']) / len(period_comp['current_clients']) * 100) if period_comp['current_clients'] else 0
    exited_pct = (len(period_comp['exited']) / len(period_comp['previous_clients']) * 100) if period_comp['previous_clients'] else 0
    
    # Display metrics using native Streamlit columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="🔄 Carryover Clients",
            value=f"{len(period_comp['carryover']):,}",
            delta=f"{carryover_pct:.1f}% of current",
            delta_color="off",
            help="Clients served in both periods"
        )
    
    with col2:
        st.metric(
            label="🆕 New Clients",
            value=f"{len(period_comp['new']):,}",
            delta=f"{new_pct:.1f}% of current",
            delta_color="off",
            help="Clients not served in previous period"
        )
    
    with col3:
        st.metric(
            label="🚪 Exited System",
            value=f"{len(period_comp['exited']):,}",
            delta=f"{exited_pct:.1f}% of previous",
            delta_color="off",
            help="Clients no longer in system"
        )
    
    # Generate insight
    if carryover_pct > 70:
        insight = f"High carryover rate ({carryover_pct:.0f}%) indicates a stable but potentially stuck population. Consider enhanced interventions for chronic homelessness."
        insight_type = "warning"
    elif new_pct > 50:
        insight = f"High influx of new clients ({new_pct:.0f}%) suggests growing need or improved outreach. Focus on prevention and diversion strategies."
        insight_type = "info"
    else:
        insight = f"Balanced mix of carryover ({carryover_pct:.0f}%) and new clients ({new_pct:.0f}%). System shows healthy flow with room for improvement."
        insight_type = "info"
    
    # Display insight
    st.html(create_info_box(insight, type=insight_type, title="Population Insights"))

# ==================== MAIN RENDER FUNCTION ====================

@st.fragment
def render_summary_metrics(df_filt: DataFrame, full_df: Optional[DataFrame] = None) -> None:
    """
    Render summary metrics with key performance indicators comparing
    the current window with the previous period.
    """
    try:
        # Ensure we have the unfiltered dataset for returns tracking
        if full_df is None:
            full_df = st.session_state.get("df")
            if full_df is None:
                st.error("Original dataset not found. Please reload your data.")
                return

        # Initialize section state
        state = init_section_state(SUMMARY_SECTION_KEY)
        filter_timestamp = st.session_state.get("last_filter_change", "")
        cache_valid = state.get("last_updated") == filter_timestamp

        if not cache_valid:
            state["last_updated"] = filter_timestamp

        # Header with help
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
                            
                **Housing Outcomes Classification:**
                - **PH Exit Rate (Housing Placement):**
                    - 🏆 **Excellent:** ≥ 50% exits to permanent housing
                    - ✅ **Good:** 35%–<50% exits to permanent housing
                    - ⚠️ **Needs Improvement:** 20%–<35% exits to permanent housing
                    - ❌ **Critical:** <20% exits to permanent housing
                - **Return Rate (Housing Stability):**
                    - 🌟 **Outstanding:** ≤ 5% returns to homelessness
                    - ✅ **Strong:** > 5%–≤ 10% returns to homelessness
                    - ⚠️ **Moderate:** > 10%–≤ 20% returns to homelessness
                    - 🚨 **High:** > 20% returns to homelessness
                
                **Population Analysis:**
                - **Households**: Count of heads of household only (not all family members)
                - **Clients Served**: Unique clients with active enrollment anytime during the period
                - **Carryover**: Clients active in both current and previous periods
                - **New**: Clients in current period who weren't in previous period
                
                **Time Comparisons:**
                - Current vs previous period (can be same length or custom)
                - Percentages show relative change from previous period
                            
                **Population Insights Classification:**
                - **High Carryover (>70%):** Stable but potentially stuck population; consider enhanced interventions for chronic homelessness.
                - **High New (>50%):** Growing need; focus on prevention and diversion strategies.
                - **Balanced:** Mix of carryover and new indicates healthy flow with room for improvement.
                
                **Important Notes:**
                - When filters are active, metrics reflect filtered programs only
                - Returns are ALWAYS tracked system-wide regardless of filters
                - Each client is counted only once per metric, even with multiple enrollments
                """)
        
        # Get time boundaries
        t0 = st.session_state.get("t0")
        t1 = st.session_state.get("t1")
        prev_start = st.session_state.get("prev_start")
        prev_end = st.session_state.get("prev_end")

        if not all([t0, t1, prev_start, prev_end]):
            st.warning("⏰ Please set date ranges in the filter panel to view metrics.")
            return

        # Calculate period lengths
        current_days = (t1 - t0).days + 1
        previous_days = (prev_end - prev_start).days + 1

        # Return window control
        col1, col2 = st.columns([2, 2])
        with col1:
            default_return_window = state.get("user_return_window", 180)
            return_window = st.number_input(
                "Return tracking window (days)",
                min_value=30,
                max_value=1095,
                value=default_return_window,
                step=30,
                help="Number of days after PH exit to track returns to homelessness"
            )
            
            if return_window != state.get("user_return_window", 180):
                state["user_return_window"] = return_window
                cache_valid = False

        with col2:
            st.info(f"📅 Tracking returns for {return_window} days ({return_window/30:.1f} months) after PH exit")

        # Compute metrics if needed
        if not cache_valid or state.get("last_return_window") != return_window:
            with st.spinner("Calculating key metrics..."):
                state["last_return_window"] = return_window
                metrics = _get_summary_metrics(
                    df_filt, full_df, t0, t1, prev_start, prev_end, return_window
                )
                state.update(metrics)

        # Retrieve metrics from state
        served_ids = state.get("served_ids", set())
        inflow_ids = state.get("inflow_ids", set())
        outflow_ids = state.get("outflow_ids", set())
        ph_ids = state.get("ph_ids", set())
        ph_exits_in_period = state.get("ph_exits_in_period", set())
        return_ids = state.get("return_ids", set())
        
        served_prev = state.get("served_prev", set())
        inflow_prev = state.get("inflow_prev", set())
        outflow_prev = state.get("outflow_prev", set())
        ph_prev = state.get("ph_prev", set())
        ph_exits_prev = state.get("ph_exits_prev", set())
        return_ids_prev = state.get("return_ids_prev", set())
        
        households_current = state.get("households_current", 0)
        households_prev = state.get("households_prev", 0)
        
        period_comp = state.get("period_comparison", {})

        # Calculate rates
        ph_rate = ph_exit_rate(df_filt, t0, t1)
        ph_rate_prev = ph_exit_rate(df_filt, prev_start, prev_end)
        
        return_rate = 0 if not ph_exits_in_period else (len(return_ids) / len(ph_exits_in_period) * 100)
        return_rate_prev = 0 if not ph_exits_prev else (len(return_ids_prev) / len(ph_exits_prev) * 100)

        # Check if filters are active
        active_filters = st.session_state.get("filters", {})
        is_filtered = any(active_filters.values())

        # Check if custom destinations are being used
        if "_custom_ph_destinations" in df_filt.columns and df_filt["_custom_ph_destinations"].any():
            custom_content = """
            <h4>🎯 Custom PH Destinations Active</h4>
            <p>Using customized permanent housing destination definitions.</p>
            """
            st.html(create_info_box(custom_content, type="info"))
        
        # Display filter context if active
        if is_filtered:
            filter_details = []
            for name, values in active_filters.items():
                if values:
                    filter_details.append(f"**{name}** ({len(values)} selected)")
            
            filter_content = f"""
            <h4>🔍 Filtered View Active</h4>
            <p>Metrics reflect filtered data only. Active filters: {", ".join(filter_details)}</p>
            <details>
                <summary><strong>How filters affect metrics (click to expand)</strong></summary>
                <div>
                    <ul>
                        <li><strong>Inflow/Outflow</strong>: Only tracks movement within filtered programs</li>
                        <li><strong>PH Exit Rate</strong>: Based on exits from filtered programs only</li>
                        <li><strong>Returns</strong>: Tracks returns to ANY program (system-wide) ✓</li>
                    </ul>
                    <p><em>For true system-wide metrics, remove all filters.</em></p>
                </div>
            </details>
            """
            st.html(create_info_box(filter_content, type="warning"))
        
        # Display period context
        col1, col2 = st.columns(2)
        with col1:
            st.html(create_info_box(
                f"<strong>{t0.strftime('%B %d, %Y')}</strong> to <strong>{t1.strftime('%B %d, %Y')}</strong><br>({current_days} days)",
                type="info",
                title="📅 Current Period"
            ))
        
        with col2:
            st.html(create_info_box(
                f"<strong>{prev_start.strftime('%B %d, %Y')}</strong> to <strong>{prev_end.strftime('%B %d, %Y')}</strong><br>({previous_days} days)",
                type="info",
                title="📅 Previous Period"
            ))
        
        # Apply metric card styling
        style_metric_cards(
            border_left_color=Colors.PRIMARY,
            box_shadow=True
        )
        
        # Display metrics in rows
        st.html(create_styled_divider())
        
        # Row 1
        row1_cols = st.columns(3)
        with row1_cols[0]:
            delta, pct = calc_delta(households_current, households_prev) if households_prev else (0, 0)
            delta_display = f"{fmt_int(delta)} ({fmt_pct(pct)})" if households_prev else "n/a"
            st.metric(
                "Households Served",
                fmt_int(households_current),
                delta_display,
                delta_color="off",
                help=_get_metric_help_text("Households Served", is_filtered)
            )
        
        with row1_cols[1]:
            delta, pct = calc_delta(len(served_ids), len(served_prev)) if served_prev else (0, 0)
            delta_display = f"{fmt_int(delta)} ({fmt_pct(pct)})" if served_prev else "n/a"
            st.metric(
                "Clients Served",
                fmt_int(len(served_ids)),
                delta_display,
                delta_color="off",
                help=_get_metric_help_text("Clients Served", is_filtered)
            )
        
        with row1_cols[2]:
            delta, pct = calc_delta(len(inflow_ids), len(inflow_prev)) if inflow_prev else (0, 0)
            delta_display = f"{fmt_int(delta)} ({fmt_pct(pct)})" if inflow_prev else "n/a"
            label = "Inflow" + (" 🔍" if is_filtered else "")
            st.metric(
                label,
                fmt_int(len(inflow_ids)),
                delta_display,
                delta_color="off",
                help=_get_metric_help_text("Inflow", is_filtered)
            )
        
        # Row 2
        row2_cols = st.columns(3)
        with row2_cols[0]:
            delta, pct = calc_delta(len(outflow_ids), len(outflow_prev)) if outflow_prev else (0, 0)
            delta_display = f"{fmt_int(delta)} ({fmt_pct(pct)})" if outflow_prev else "n/a"
            label = "Outflow" + (" 🔍" if is_filtered else "")
            st.metric(
                label,
                fmt_int(len(outflow_ids)),
                delta_display,
                delta_color="off",
                help=_get_metric_help_text("Outflow", is_filtered)
            )
        
        with row2_cols[1]:
            delta, pct = calc_delta(len(ph_ids), len(ph_prev)) if ph_prev else (0, 0)
            delta_display = f"{fmt_int(delta)} ({fmt_pct(pct)})" if ph_prev else "n/a"
            st.metric(
                "PH Exits",
                fmt_int(len(ph_ids)),
                delta_display,
                delta_color="normal",  # Positive is good for PH exits
                help=_get_metric_help_text("PH Exits", is_filtered)
            )
        
        with row2_cols[2]:
            delta = ph_rate - ph_rate_prev
            delta_display = f"{delta:+.1f} pp" if ph_rate_prev is not None else "n/a"
            st.metric(
                "PH Exit Rate",
                fmt_pct(ph_rate),
                delta_display,
                delta_color="normal",  # Higher is better
                help=_get_metric_help_text("PH Exit Rate", is_filtered)
            )
        
        # Row 3
        row3_cols = st.columns(3)
        with row3_cols[0]:
            delta, pct = calc_delta(len(return_ids), len(return_ids_prev)) if return_ids_prev else (0, 0)
            delta_display = f"{fmt_int(delta)} ({fmt_pct(pct)})" if return_ids_prev else "n/a"
            st.metric(
                f"Returns ({return_window}d)",
                fmt_int(len(return_ids)),
                delta_display,
                delta_color="inverse",  # Lower is better for returns
                help=_get_metric_help_text("Returns to Homelessness", is_filtered)
            )
        
        with row3_cols[1]:
            delta = return_rate - return_rate_prev
            delta_display = f"{delta:+.1f} pp" if return_rate_prev is not None else "n/a"
            st.metric(
                "Return Rate",
                fmt_pct(return_rate),
                delta_display,
                delta_color="inverse",  # Lower is better
                help=_get_metric_help_text("Return Rate", is_filtered)
            )
        
        with row3_cols[2]:
            net_flow_current = len(inflow_ids) - len(outflow_ids)
            net_flow_prev = len(inflow_prev) - len(outflow_prev)
            delta, _ = calc_delta(net_flow_current, net_flow_prev) if net_flow_prev is not None else (0, 0)
            delta_display = fmt_int(delta)
            label = "Net Flow" + (" 🔍" if is_filtered else "")
            st.metric(
                label,
                fmt_int(net_flow_current),
                delta_display,
                delta_color="off",
                help=_get_metric_help_text("Net Flow", is_filtered)
            )
        
        # System Analysis
        st.html(create_styled_divider())
        st.markdown("### 🔍 System Analysis")
        
        # System Flow Analysis
        net_flow = len(inflow_ids) - len(outflow_ids)
        flow_html = _generate_flow_insight_html(len(inflow_ids), len(outflow_ids), net_flow)
        st.html(flow_html)
        
        # Client Population Analysis
        if period_comp:
            st.markdown("#### 📊 Client Population Analysis")
            _render_period_breakdown(period_comp)
        
        # Housing Outcomes Analysis
        st.markdown("#### 🏠 Housing Outcomes Assessment")
        _generate_housing_outcomes_html(ph_rate, return_rate, return_window)
        
        # Period-over-Period Changes
        st.markdown("### 📈 Period-over-Period Changes")
        
        # Find significant changes
        significant_changes = []
        metric_changes = [
            ("Clients Served", len(served_ids), len(served_prev)),
            ("Inflow", len(inflow_ids), len(inflow_prev)),
            ("Outflow", len(outflow_ids), len(outflow_prev)),
            ("PH Exits", len(ph_ids), len(ph_prev))
        ]
        
        for name, current, previous in metric_changes:
            if previous > 0:
                _, pct_change = calc_delta(current, previous)
                if abs(pct_change) >= 10:
                    significant_changes.append((name, current, previous, pct_change))
        
        if significant_changes:
            significant_changes.sort(key=lambda x: abs(x[3]), reverse=True)
            
            for name, current, previous, pct_change in significant_changes[:3]:
                icon = "📈" if pct_change > 0 else "📉"
                # Use info type for neutral changes instead of success/warning
                if name in ["Clients Served", "Inflow", "PH Exits"]:
                    # These are generally positive when increasing
                    color_type = "info" if abs(pct_change) < 20 else ("success" if pct_change > 0 else "warning")
                elif name == "Outflow":
                    # Context-dependent - neither inherently good nor bad
                    color_type = "info"
                else:
                    color_type = "info"
                
                change_content = f"""
                <strong>{icon} {name}</strong><br>
                <span>{previous:,} → {current:,} ({pct_change:+.1f}% change)</span>
                """
                
                st.html(create_info_box(change_content, type=color_type))
        else:
            st.info("📊 No significant changes (≥10%) detected between periods.")

    except Exception as e:
        st.error(f"Error calculating summary metrics: {str(e)}")
        st.info("💡 Try refreshing the page or reloading your data.")

# ==================== PUBLIC API ====================

__all__ = ['render_summary_metrics', 'SUMMARY_SECTION_KEY']