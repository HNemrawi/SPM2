"""
Trend explorer section for HMIS dashboard with enhanced UI and auto-adjusting charts.
"""

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pandas import DataFrame, Timestamp


from analysis.general.data_utils import (
    DEMOGRAPHIC_DIMENSIONS, FREQUENCY_MAP, calculate_demographic_growth,
    inflow, outflow, ph_exit_clients, recalculated_metric_time_series,
    recalculated_metric_time_series_by_group, return_after_exit, served_clients
)
from analysis.general.filter_utils import (
    get_filter_timestamp, hash_data, init_section_state, is_cache_valid, invalidate_cache
)
from analysis.general.theme import (
    MAIN_COLOR, NEUTRAL_COLOR, PLOT_TEMPLATE, SECONDARY_COLOR,
    SUCCESS_COLOR, WARNING_COLOR, apply_chart_style, fmt_int, fmt_pct, blue_divider
)

CUSTOM_COLOR_SEQUENCE = [
    "#1f77b4",  # muted blue
    "#ff7f0e",  # safety orange
    "#2ca02c",  # cooked asparagus green
    "#d62728",  # brick red
    "#9467bd",  # muted purple
    "#8c564b",  # chestnut brown
    "#e377c2",  # raspberry yogurt pink
    "#7f7f7f",  # middle gray
    "#17becf",  # blue-teal
    "#bcbd22",  # curry yellow-green
    "#393b79",  # dark slate blue
    "#637939",  # olive green
    "#8c6d31",  # brownish yellow
    "#843c39",  # dark sienna
    "#7b4173",  # deep magenta
]

# Constants
TREND_SECTION_KEY = "trend_explorer"

# Define metric colors with meaningful associations
METRIC_COLORS = {
    "Active Clients": "#F3F4F6",  # Near white (brightest)
    "Inflow": "#9CA3AF",          # Medium-light gray
    "Outflow": "#4B5563",         # Dark gray
    "PH Exits": SUCCESS_COLOR,     # Success green
}

def _get_metric_color(metric_name: str) -> str:
    """Get appropriate color for a metric based on its type."""
    if "Returns to Homelessness" in metric_name:
        return SECONDARY_COLOR  # Warning/danger color
    return METRIC_COLORS.get(metric_name, MAIN_COLOR)

def _get_hover_template(freq: str) -> str:
    """Get appropriate hover template based on frequency."""
    if freq == "D":
        return "<b>%{x|%b %d, %Y}</b><br>Value: %{y:,.0f}<extra></extra>"
    elif freq == "W":
        return "<b>Week of %{x|%b %d, %Y}</b><br>Value: %{y:,.0f}<extra></extra>"
    elif freq in ["M", "Q"]:
        return "<b>%{x|%b %Y}</b><br>Value: %{y:,.0f}<extra></extra>"
    else:  # Year
        return "<b>%{x|%Y}</b><br>Value: %{y:,.0f}<extra></extra>"

def _get_x_axis_angle(freq: str, num_points: int) -> int:
    """Determine optimal x-axis label rotation to prevent overlap."""
    if freq == "D" and num_points > 30:
        return -90
    elif freq == "W" and num_points > 20:
        return -45
    elif freq == "M" and num_points > 24:
        return -45
    else:
        return 0

def _calculate_dynamic_height(num_points: int, base_height: int = 400) -> int:
    """Calculate dynamic chart height based on number of data points."""
    if num_points <= 12:
        return base_height
    elif num_points <= 24:
        return base_height + 50
    elif num_points <= 52:
        return base_height + 100
    else:
        return min(base_height + 200, 700)

def _calculate_dynamic_margins(num_points: int, has_text_labels: bool = False) -> dict:
    """Calculate dynamic margins to prevent cutoff."""
    base_margins = {"l": 80, "r": 80, "t": 100, "b": 100}
    
    # Increase bottom margin for rotated labels
    if num_points > 20:
        base_margins["b"] = 140
    elif num_points > 10:
        base_margins["b"] = 120
    
    # Increase top/bottom margins significantly for text labels
    if has_text_labels:
        base_margins["t"] = 120
        base_margins["b"] = max(base_margins["b"], 120)
        # Extra padding for value labels that appear above/below bars
        base_margins["t"] += 40
        base_margins["b"] += 20
    
    return base_margins

def _get_trend_icon(total_change: float, increase_is_negative: bool = False) -> str:
    """Get appropriate emoji icon for trend direction."""
    if abs(total_change) < 0.001:
        return "➡️"
    elif total_change > 0:
        return "📉" if increase_is_negative else "📈"
    else:
        return "📈" if increase_is_negative else "📉"

def _get_trend_color(change: float, increase_is_negative: bool = False) -> str:
    """Get appropriate color for a change value."""
    if abs(change) < 0.001:
        return "gray"
    elif (change > 0 and not increase_is_negative) or (change < 0 and increase_is_negative):
        return "green"
    else:
        return "red"

def _format_period_label(freq_label: str) -> str:
    """Convert frequency label to user-friendly period name."""
    period_map = {
        "Days": "day",
        "Weeks": "week", 
        "Months": "month",
        "Quarters": "quarter",
        "Years": "year"
    }
    return period_map.get(freq_label, "period")

def _render_metric_explanation_card():
    """Render a well-formatted metric explanation card using HTML."""
    explanation_html = """
    <div style="background-color: rgba(17, 24, 39, 0.8); border: 1px solid rgba(55, 65, 81, 0.5); 
                border-radius: 12px; padding: 20px; margin-bottom: 20px;">
        <h4 style="color: #60A5FA; margin-top: 0; margin-bottom: 16px; font-size: 18px;">
            📊 Understanding the Metrics
        </h4>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px;">
            <div style="background-color: rgba(31, 41, 55, 0.5); padding: 12px; border-radius: 8px;">
                <strong style="color: #93C5FD;">Active Clients</strong>
                <p style="margin: 4px 0; color: #E5E7EB; font-size: 14px;">
                    All clients enrolled at any point during the time period
                </p>
            </div>
            <div style="background-color: rgba(31, 41, 55, 0.5); padding: 12px; border-radius: 8px;">
                <strong style="color: #93C5FD;">Inflow</strong>
                <p style="margin: 4px 0; color: #E5E7EB; font-size: 14px;">
                    Clients entering during the period who weren't enrolled the day before it started
                </p>
            </div>
            <div style="background-color: rgba(31, 41, 55, 0.5); padding: 12px; border-radius: 8px;">
                <strong style="color: #93C5FD;">Outflow</strong>
                <p style="margin: 4px 0; color: #E5E7EB; font-size: 14px;">
                    Clients who exited during the period and aren't enrolled on the last day
                </p>
            </div>
            <div style="background-color: rgba(16, 185, 129, 0.1); padding: 12px; border-radius: 8px; 
                        border: 1px solid rgba(16, 185, 129, 0.3);">
                <strong style="color: #34D399;">PH Exits</strong> 
                <span style="color: #10B981; font-size: 12px;">✓ Positive Outcome</span>
                <p style="margin: 4px 0; color: #E5E7EB; font-size: 14px;">
                    Clients exiting to permanent housing during the period
                </p>
            </div>
            <div style="background-color: rgba(239, 68, 68, 0.1); padding: 12px; border-radius: 8px;
                        border: 1px solid rgba(239, 68, 68, 0.3);">
                <strong style="color: #F87171;">Returns to Homelessness</strong>
                <span style="color: #EF4444; font-size: 12px;">✗ Negative Outcome</span>
                <p style="margin: 4px 0; color: #E5E7EB; font-size: 14px;">
                    Clients who exited to PH and returned within the specified window
                </p>
            </div>
        </div>
    </div>
    """
    st.html(explanation_html)

def _render_insight_card(metric_name: str, df: DataFrame, period_label: str, 
                        increase_is_negative: bool, is_neutral_metric: bool):
    """Render a well-formatted insight card using HTML."""
    # Calculate insights
    avg_change = df["delta"].mean()
    total_change = df["delta"].sum()
    recent_periods = min(3, len(df) - 1)
    recent_change = df["delta"].iloc[-recent_periods:].mean() if recent_periods > 0 else 0
    
    max_increase = df["delta"].max()
    max_increase_date = df.loc[df["delta"].idxmax(), "bucket"]
    max_decrease = df["delta"].min()
    max_decrease_date = df.loc[df["delta"].idxmin(), "bucket"]
    
    # Get trend icon and direction
    trend_icon = _get_trend_icon(total_change, increase_is_negative)
    
    if total_change > 0:
        if increase_is_negative:
            trend_direction = "Increasing (Negative Trend)"
            trend_color = "#EF4444"
        else:
            trend_direction = "Increasing (Positive Trend)"
            trend_color = "#10B981"
    elif total_change < 0:
        if increase_is_negative:
            trend_direction = "Decreasing (Positive Trend)"
            trend_color = "#10B981"
        else:
            trend_direction = "Decreasing (Negative Trend)"
            trend_color = "#EF4444"
    else:
        trend_direction = "Stable"
        trend_color = "#6B7280"
    
    if is_neutral_metric:
        trend_color = "#60A5FA"
        trend_direction = trend_direction.split(" (")[0]  # Remove positive/negative label
    
    insight_html = f"""
    <div style="background-color: rgba(31, 41, 55, 0.5); border-radius: 12px; padding: 20px; height: 100%;">
        <h3 style="color: #E5E7EB; margin-top: 0; margin-bottom: 16px; font-size: 20px;">
            Key Insights {trend_icon}
        </h3>
        
        <div style="background-color: rgba(17, 24, 39, 0.5); border-radius: 8px; padding: 12px; margin-bottom: 16px;">
            <p style="margin: 0; color: #9CA3AF; font-size: 14px;">Current Trend</p>
            <p style="margin: 4px 0; color: {trend_color}; font-size: 18px; font-weight: bold;">
                {trend_direction}
            </p>
        </div>
        
        <div style="display: grid; gap: 12px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="color: #9CA3AF; font-size: 14px;">Average Change</span>
                <span style="color: #E5E7EB; font-size: 16px; font-weight: 500;">
                    {avg_change:+,.0f} per {period_label}
                </span>
            </div>
            
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="color: #9CA3AF; font-size: 14px;">Total Change</span>
                <span style="color: #E5E7EB; font-size: 16px; font-weight: 500;">
                    {total_change:+,.0f} clients
                </span>
            </div>
            
            <hr style="border-color: rgba(55, 65, 81, 0.5); margin: 8px 0;">
            
            <div style="color: #9CA3AF; font-size: 14px; margin-bottom: 8px;">Notable Changes</div>
            
            <div style="background-color: rgba(16, 185, 129, 0.1); border-radius: 6px; padding: 8px;">
                <div style="color: #34D399; font-size: 13px;">Largest Increase</div>
                <div style="color: #E5E7EB; font-size: 15px; font-weight: 500;">
                    {max_increase:+,.0f} ({max_increase_date:%b %Y})
                </div>
            </div>
            
            <div style="background-color: rgba(239, 68, 68, 0.1); border-radius: 6px; padding: 8px;">
                <div style="color: #F87171; font-size: 13px;">Largest Decrease</div>
                <div style="color: #E5E7EB; font-size: 15px; font-weight: 500;">
                    {max_decrease:+,.0f} ({max_decrease_date:%b %Y})
                </div>
            </div>
        </div>
    </div>
    """
    st.html(insight_html)

def _render_interpretation_note(metric_name: str, is_neutral_metric: bool, increase_is_negative: bool):
    """Render a clear interpretation note using HTML."""
    if is_neutral_metric:
        icon = "ℹ️"
        color = "#60A5FA"
        message = f"{metric_name} is a <strong>volume metric</strong>. Changes show fluctuations in client numbers but are neither positive nor negative outcomes."
    elif increase_is_negative:
        icon = "⚠️"
        color = "#F59E0B"
        message = f"For {metric_name}, <strong style='color: #EF4444;'>increases are negative</strong> - we want to see this number go down."
    else:
        icon = "✅"
        color = "#10B981"
        message = f"For {metric_name}, <strong style='color: #10B981;'>increases are positive</strong> - we want to see this number go up."
    
    note_html = f"""
    <div style="background-color: rgba(31, 41, 55, 0.5); border: 1px solid {color}40; 
                border-radius: 8px; padding: 12px; margin-top: 16px; display: flex; align-items: center; gap: 12px;">
        <span style="font-size: 24px;">{icon}</span>
        <p style="margin: 0; color: #E5E7EB; font-size: 14px;">
            {message}
        </p>
    </div>
    """
    st.html(note_html)

def _get_trend_data(
    df_filt: DataFrame,
    full_df: DataFrame,
    metric_funcs: Dict[str, Any],
    sel_metrics: List[str],
    sel_freq: str,
    group_col: Optional[str],
    t0: Timestamp,
    t1: Timestamp,
    return_window: int
) -> Dict[str, DataFrame]:
    """
    Calculate trend data for selected metrics and groups.
    
    Parameters:
    -----------
    df_filt : DataFrame
        Filtered DataFrame
    full_df : DataFrame
        Full DataFrame for returns analysis
    metric_funcs : dict
        Dictionary of metric functions
    sel_metrics : list
        List of selected metric names
    sel_freq : str
        Selected frequency code (D, W, M, Q, Y)
    group_col : str, optional
        Group column for breakdown
    t0 : Timestamp
        Start date
    t1 : Timestamp
        End date
    return_window : int
        Days to check for returns
        
    Returns:
    --------
    dict
        Dictionary of trend data by metric
    """
    multi_data = {}
    
    for metric_name in sel_metrics:
        try:
            metric_func = metric_funcs[metric_name]
            
            # Special handling for returns function
            if metric_name.startswith("Returns to Homelessness"):
                # Wrap return_after_exit to pass full_df and return_window
                metric_func = lambda df, s, e, mf=metric_func: mf(df, full_df, s, e, return_window)
            
            if group_col is None:
                # Overall trend (no breakdown)
                ts = recalculated_metric_time_series(df_filt, metric_func, t0, t1, sel_freq)
                
                if ts is None or ts.empty:
                    continue
                    
                ts = ts.sort_values("bucket")
                ts["delta"] = ts["count"].diff().fillna(0)
                
                # Calculate rolling average with min_periods handling
                min_periods = 1  # Always calculate rolling average even with few points
                ts["rolling"] = ts["count"].rolling(window=3, min_periods=min_periods).mean()
                ts["metric"] = metric_name
                ts["pct_change"] = ts["count"].pct_change().fillna(0) * 100
                
            else:
                # Breakdown by group
                ts = recalculated_metric_time_series_by_group(
                    df_filt, metric_func, group_col, t0, t1, sel_freq
                )
                
                if ts is None or ts.empty:
                    continue
                    
                ts = ts.sort_values(["group", "bucket"])
                ts["metric"] = metric_name
                
                # Process each group separately
                ts_groups = []
                for group, group_data in ts.groupby("group"):
                    group_data = group_data.sort_values("bucket")
                    group_data["delta"] = group_data["count"].diff().fillna(0)
                    group_data["pct_change"] = group_data["count"].pct_change().fillna(0) * 100
                    group_data["rolling"] = group_data["count"].rolling(window=3, min_periods=1).mean()
                    ts_groups.append(group_data)
                
                if ts_groups:
                    ts = pd.concat(ts_groups)
            
            multi_data[metric_name] = ts
            
        except Exception as e:
            st.error(f"Error calculating {metric_name}: {str(e)}")
    
    return multi_data

@st.fragment
def render_trend_explorer(df_filt: DataFrame, full_df: Optional[DataFrame] = None) -> None:
    """
    Render trend explorer section with enhanced time series analysis and auto-updating filters.
    
    Parameters:
    -----------
    df_filt : DataFrame
        Filtered DataFrame
    full_df : DataFrame, optional
        Full DataFrame for returns analysis
    """
    if full_df is None:
        full_df = df_filt.copy()
        
    # Initialize section state
    state = init_section_state(TREND_SECTION_KEY)
    summary_state = st.session_state.get(f"state_summary_metrics", {})
    
    # Get return window from summary section
    return_window = summary_state.get("return_window", 180)

    # Check if cache is valid
    filter_timestamp = get_filter_timestamp()
    cache_valid = is_cache_valid(state, filter_timestamp)

    if not cache_valid:
        invalidate_cache(state, filter_timestamp)
        # Clear cached data
        for k in list(state.keys()):
            if k not in ["last_updated"]:
                state.pop(k, None)

    # Header with better styling
    header_html = """
    <div style="margin-bottom: 24px;">
        <h2 style="color: #E5E7EB; margin-bottom: 8px; font-size: 28px;">
            📈 Trend Analysis
        </h2>
        <p style="color: #9CA3AF; font-size: 16px; margin: 0;">
            Track how metrics change over time, identify patterns, and understand your data trends
        </p>
    </div>
    """
    col_header, col_info = st.columns([6, 1])
    with col_header:
        st.html(header_html)
    with col_info:
        with st.popover("ℹ️ Help", use_container_width=True):
            st.markdown("""
            ### Understanding Trend Analysis
            
            This section tracks how metrics change over time, revealing patterns, seasonality, and the impact of interventions.
            
            **Available Metrics:**
            
            **Volume Metrics (Neutral - neither good nor bad):**
            - **Active Clients**: All unique clients enrolled at any point during each time period
            - **Inflow**: Clients entering who weren't in programs the day before
            - **Outflow**: Clients exiting who aren't in programs at period end
            
            **Outcome Metrics:**
            - **PH Exits** ✅: Clients exiting to permanent housing (increases are positive)
            - **Returns to Homelessness** ❌: PH exits who returned within tracking window (increases are negative)
            
            **Time Aggregation Options:**
            - **Days**: Daily values (best for short periods)
            - **Weeks**: Weekly aggregates (good for 1-3 month views)
            - **Months**: Monthly totals (recommended for most analyses)
            - **Quarters**: Quarterly summaries (good for annual views)
            - **Years**: Annual totals (for long-term trends)
            
            **Analysis Features:**
            
            **1. Overall Trends (No breakdown):**
            - Line charts showing metric values over time
            - Rolling averages (3-period default, auto-adjusts by frequency)
            - Period-over-period change analysis
            - Combined view for comparing multiple metrics
            
            **2. Demographic Breakdowns:**
            - Compare trends across demographic groups
            - Automatic limiting to top groups for clarity
            - Growth rate analysis (first vs last period)
            - Disparity tracking over time
            
            **Key Calculations:**
            - **Each time period is calculated independently** using full metric logic
            - **Not simple counts** - proper deduplication and business rules applied
            - **Period changes**: Show both absolute and percentage changes
            - **Growth rates**: (Last Value - First Value) ÷ First Value × 100
            
            **Visual Components:**
            
            **1. Trend Line Charts:**
            - Actual values with markers at each data point
            - Rolling average lines (dashed) for smoothing
            - Auto-scaling axes and dynamic height
            - Smart date formatting based on frequency
            
            **2. Change Analysis Charts:**
            - Bar charts showing period-over-period changes
            - Color-coded: Green = improvement, Red = decline
            - Special handling for metrics where decrease is good
            - Text labels showing exact change values
            
            **3. Insights Panel:**
            - Current trend direction and total change
            - Average change per period
            - Largest increases and decreases with dates
            - Volatility: Measures how much the numbers swing up or down over time. We calculate the typical size of these swings and group them into:
                - **Very Stable** — swings usually under 5%, indicating a steady trend.  
                - **Moderately Stable** — swings between 5% and 10%, showing mild fluctuations.  
                - **Somewhat Volatile** — swings between 10% and 20%, suggesting noticeable ups and downs.  
                - **Highly Volatile** — swings over 20%, highlighting rapid or unpredictable changes.  

            
            **4. Growth Analysis (for breakdowns):**
            - Side-by-side growth comparisons
            - Fastest growing and declining groups
            - Current disparity ratios
            - Resource utilization patterns
            """)
        
    # Add metric explanations with better formatting
    _render_metric_explanation_card()

    # Define available metrics
    metric_opts = {
        "Active Clients": served_clients,
        "Inflow": inflow,
        "Outflow": outflow,
        "PH Exits": ph_exit_clients,
        f"Returns to Homelessness ({return_window}d)": return_after_exit
    }

    # Create a unique key suffix based on filter timestamp
    key_suffix = hash_data(filter_timestamp)
    
    # Use columns for better filter layout
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        # Create metrics selection with state persistence
        metrics_key = f"trend_metrics_{key_suffix}"
        
        # Function to handle metrics selection change
        def on_metrics_change():
            # Update the session state with selected metrics
            state["needs_recalc"] = True
            st.session_state["prev_trend_metrics"] = st.session_state[metrics_key]
        
        # Initialize default metrics if needed
        if "prev_trend_metrics" in st.session_state:
            valid_prev_metrics = [m for m in st.session_state["prev_trend_metrics"] if m in metric_opts]
            default_metrics = valid_prev_metrics if valid_prev_metrics else ["Active Clients"]
        else:
            default_metrics = ["Active Clients"]
        
        # Metrics selection widget with appropriate defaults
        sel_metrics = st.multiselect(
            "Metrics to display",
            list(metric_opts.keys()),
            default=default_metrics,
            key=metrics_key,
            on_change=on_metrics_change,
            help="Select multiple metrics to compare trends over time"
        )
    
    with filter_col2:
        # Time bucket selection with user-friendly label
        freq_key = f"trend_freq_{key_suffix}"
        
        # Function to handle frequency change
        def on_freq_change():
            # Get the new frequency selected
            new_freq = st.session_state[freq_key]
            state["needs_recalc"] = True
            st.session_state["prev_trend_freq"] = new_freq
            
            # Adjust rolling window size based on frequency
            roll_win_key = f"trend_roll_win_{key_suffix}"
            
            # Suggested window sizes for different frequencies
            suggested_windows = {
                "Days": 7,      # Weekly rolling for daily data
                "Weeks": 4,     # Monthly rolling for weekly data
                "Months": 3,    # Quarterly rolling for monthly data
                "Quarters": 2,  # Biannual rolling for quarterly data
                "Years": 2      # Biannual rolling for yearly data
            }
            
            # Update window size based on new frequency
            st.session_state[roll_win_key] = suggested_windows.get(new_freq, 3)
        
        # Initialize frequency if needed
        freq_default = "Months"
        if "prev_trend_freq" in st.session_state:
            freq_default = st.session_state["prev_trend_freq"]
        
        # Frequency selectbox with user-friendly label
        sel_freq_label = st.selectbox(
            "View by",  # Changed from "Time bucket"
            list(FREQUENCY_MAP.keys()),
            index=list(FREQUENCY_MAP.keys()).index(freq_default),
            key=freq_key,
            on_change=on_freq_change,
            help="Choose how to group data points over time"
        )
        
        # Convert to frequency code
        sel_freq = FREQUENCY_MAP[sel_freq_label]
    
    with filter_col3:
        # Demographic breakdown selection with user-friendly label
        breakdown_opts = [("None (overall)", None)] + DEMOGRAPHIC_DIMENSIONS
        
        break_key = f"trend_break_{key_suffix}"
        
        # Function to handle breakdown change
        def on_break_change():
            state["needs_recalc"] = True
            st.session_state["prev_trend_break"] = st.session_state[break_key]
            # Clear any previous breakdown selections to avoid confusion
            if "trend_groups_selection" in state:
                state.pop("trend_groups_selection")
        
        # Initialize breakdown selection if needed
        break_default = "None (overall)"
        if "prev_trend_break" in st.session_state:
            valid_breaks = [lbl for lbl, _ in breakdown_opts]
            if st.session_state["prev_trend_break"] in valid_breaks:
                break_default = st.session_state["prev_trend_break"]
        
        # Get index for the selected breakdown
        break_options = [lbl for lbl, _ in breakdown_opts]
        
        # Breakdown selectbox with user-friendly label
        sel_break = st.selectbox(
            "Compare across",  # Changed from "Break down by"
            break_options,
            index=break_options.index(break_default),
            key=break_key,
            on_change=on_break_change,
            help="Choose a demographic dimension to compare trends across different groups"
        )
        
        # Get the actual column name
        group_col = dict(breakdown_opts)[sel_break]
    
    # Default settings for rolling average
    roll_win_key = f"trend_roll_win_{key_suffix}"
    
    # Set default rolling window size if not already set
    if roll_win_key not in st.session_state:
        # Use suggested window size based on current frequency
        suggested_windows = {
            "Days": 7,      # Weekly rolling for daily data
            "Weeks": 4,     # Monthly rolling for weekly data
            "Months": 3,    # Quarterly rolling for monthly data
            "Quarters": 2,  # Biannual rolling for quarterly data
            "Years": 2      # Biannual rolling for yearly data
        }
        st.session_state[roll_win_key] = suggested_windows.get(sel_freq_label, 3)
    
    # Always show rolling average
    do_roll = True
    roll_window = st.session_state[roll_win_key]
    
    # Default settings for display options
    display_key = f"trend_display_{key_suffix}"
    
    # Set default display mode if not already set
    if display_key not in st.session_state:
        if "prev_trend_display" in st.session_state:
            st.session_state[display_key] = st.session_state["prev_trend_display"]
        else:
            st.session_state[display_key] = "Combined view"
    
    # Use the stored display mode
    display_mode = st.session_state[display_key]
    
    # Always show period change
    do_delta = True
    
    # Define which metrics should be interpreted in reverse (where higher is worse)
    metric_interpretation = {
        m: m.startswith("Returns to Homelessness")
        for m in sel_metrics
    }
    
    # Define which metrics should use neutral colors (neither good nor bad)
    neutral_metrics = {"Active Clients", "Inflow", "Outflow"}

    # Get time boundaries
    t0 = st.session_state.get("t0")
    t1 = st.session_state.get("t1")
    
    if not all([t0, t1]):
        st.warning("Please set date ranges in the filter panel.")
        return
    
    if not sel_metrics:
        st.info("Please select at least one metric to display.")
        return

    # Generate cache key for selected options
    metrics_str = "-".join(sorted(sel_metrics))
    cache_key = (
        f"{metrics_str}_{sel_freq}_{sel_break}_{t0.strftime('%Y-%m-%d')}_"
        f"{t1.strftime('%Y-%m-%d')}_roll{roll_window}_rollon{int(do_roll)}"
    )
    
    # Check if we need to recalculate
    recalc = (
        state.get("cache_key") != cache_key or 
        state.get("needs_recalc", True)
    )

    # Calculate data for each metric
    if recalc or "multi_trend_data" not in state:
        # Reset needs_recalc flag
        state["needs_recalc"] = False
        state["cache_key"] = cache_key
        
        # Progress indicator
        with st.spinner("Calculating trends..."):
            multi_data = _get_trend_data(
                df_filt, full_df, metric_opts, sel_metrics, sel_freq, 
                group_col, t0, t1, return_window
            )
            
            # Cache the result
            state["multi_trend_data"] = multi_data
    else:
        multi_data = state.get("multi_trend_data", {})

    # Check if we have data to display
    if not multi_data or all(df.empty for df in multi_data.values() if df is not None):
        st.info("No data available for the selected metrics and time period.")
        return

    # Show a divider before the visualizations
    blue_divider()
    
    # Render charts based on display mode and breakdown
    if group_col is None:
        # No breakdown - show overall trends
        if display_mode == "Combined view" and len(sel_metrics) > 1:
            # Combined view for multiple metrics
            valid_dfs = [
                multi_data[m] for m in sel_metrics 
                if m in multi_data and multi_data[m] is not None and not multi_data[m].empty
            ]
            
            if not valid_dfs:
                st.info("No valid data available for the selected metrics.")
                return
                
            # Combine data from all metrics
            combined_df = pd.concat(valid_dfs)
            
            if combined_df.empty:
                st.info("No data available for the selected metrics.")
                return
            
            # Create color mapping for metrics
            color_map = {m: _get_metric_color(m) for m in sel_metrics}
            
            # Calculate dynamic height
            num_points = len(combined_df["bucket"].unique())
            chart_height = _calculate_dynamic_height(num_points)
            
            # Create combined line chart
            fig = px.line(
                combined_df, 
                x="bucket", 
                y="count", 
                color="metric",
                markers=True,
                template=PLOT_TEMPLATE,
                title=f"Metrics Trend ({sel_freq_label})",
                labels={
                    "bucket": "Time Period", 
                    "count": "Value", 
                    "metric": "Metric"
                },
                color_discrete_map=color_map
            )
            
            # Add rolling average lines if enabled
            if do_roll:
                for metric_name in sel_metrics:
                    if (metric_name in multi_data and multi_data[metric_name] is not None 
                            and not multi_data[metric_name].empty):
                        df = multi_data[metric_name]
                        fig.add_scatter(
                            x=df["bucket"],
                            y=df["rolling"],
                            mode="lines",
                            name=f"{metric_name} ({roll_window}-{_format_period_label(sel_freq_label)} avg)",
                            line=dict(dash="dash", width=1.5),
                            opacity=0.6,
                            showlegend=True
                        )
            
            # Apply consistent styling with dynamic height
            fig = apply_chart_style(
                fig,
                xaxis_title="Time Period",
                yaxis_title="Number of Clients",
                height=chart_height,
                legend_orientation="h"
            )
            
            # Update layout with dynamic margins
            margins = _calculate_dynamic_margins(num_points)
            fig.update_layout(
                margin=margins,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5
                )
            )
            
            # Format x-axis dates based on frequency with smart rotation
            fig.update_xaxes(
                tickformat="%b %Y" if sel_freq in ["M", "Q"] else ("%Y" if sel_freq == "Y" else "%b %d"),
                tickangle=_get_x_axis_angle(sel_freq, num_points)
            )
            
            # Improve hover template based on frequency
            fig.update_traces(hovertemplate=_get_hover_template(sel_freq))
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Show period changes if enabled
            if do_delta:
                period_label = _format_period_label(sel_freq_label)
                with st.expander(f"📊 {period_label.capitalize()}-to-{period_label} Changes", expanded=True):
                    # Create tabs for each metric
                    metric_tabs = st.tabs(sel_metrics)
                    
                    for i, metric_name in enumerate(sel_metrics):
                        if (metric_name in multi_data and multi_data[metric_name] is not None 
                                and not multi_data[metric_name].empty):
                            df = multi_data[metric_name]
                            
                            with metric_tabs[i]:
                                # Two-column layout
                                chart_col, insight_col = st.columns([3, 2])
                                
                                # Determine if increasing is good or bad
                                increase_is_negative = metric_interpretation.get(metric_name, False)
                                
                                with chart_col:
                                    # Check if this is a neutral metric
                                    is_neutral_metric = metric_name in neutral_metrics
                                    
                                    # Color bars based on direction and interpretation
                                    bar_colors = []
                                    for x in df["delta"]:
                                        if abs(x) < 0.001:
                                            bar_colors.append("#cccccc")
                                        elif is_neutral_metric:
                                            # Use main blue color for neutral metrics
                                            bar_colors.append(MAIN_COLOR)
                                        elif (x > 0 and not increase_is_negative) or (x < 0 and increase_is_negative):
                                            bar_colors.append(SUCCESS_COLOR)
                                        else:
                                            bar_colors.append(SECONDARY_COLOR)
                                    
                                    # Calculate dynamic height for delta chart
                                    delta_height = _calculate_dynamic_height(len(df), base_height=350)
                                    
                                    # Create delta bar chart
                                    fig_delta = px.bar(
                                        df, 
                                        x="bucket", 
                                        y="delta",
                                        template=PLOT_TEMPLATE,
                                        title=f"Change in {metric_name}",
                                        labels={"bucket": "", "delta": f"{period_label.capitalize()} Change"},
                                    )
                                    
                                    # Format bars and hover
                                    fig_delta.update_traces(
                                        marker_color=bar_colors,
                                        texttemplate="%{y:+,.0f}",
                                        textfont=dict(color="white", size=11),
                                        textposition="outside",
                                        hovertemplate="<b>%{x|%b %Y}</b><br>Change: %{y:+,.0f}<extra></extra>",
                                        cliponaxis=False  # Prevent clipping of text
                                    )
                                    
                                    # Apply consistent styling with dynamic height
                                    fig_delta = apply_chart_style(
                                        fig_delta,
                                        xaxis_title="Time Period",
                                        yaxis_title="Change in Clients",
                                        height=delta_height,
                                        showlegend=False
                                    )
                                    
                                    # Additional layout updates to ensure labels aren't cut off
                                    fig_delta.update_layout(
                                        yaxis=dict(
                                            # Extend the range to accommodate labels
                                            range=[
                                                min(df["delta"].min() * 1.4, df["delta"].min() - 50),
                                                max(df["delta"].max() * 1.4, df["delta"].max() + 50)
                                            ]
                                        )
                                    )
                                    
                                    # Update layout with dynamic margins
                                    delta_margins = _calculate_dynamic_margins(len(df), has_text_labels=True)
                                    fig_delta.update_layout(
                                        margin=delta_margins,
                                        yaxis=dict(
                                            automargin=True,
                                            rangemode="tozero",
                                            # Add padding to y-axis range for text labels
                                            range=[min(df["delta"].min() * 1.3, -10), 
                                                   max(df["delta"].max() * 1.3, 10)]
                                        )
                                    )
                                    
                                    # Format x-axis dates
                                    fig_delta.update_xaxes(
                                        tickformat="%b %Y" if sel_freq in ["M", "Q"] else ("%Y" if sel_freq == "Y" else "%b %d"),
                                        tickangle=_get_x_axis_angle(sel_freq, len(df))
                                    )
                                    
                                    # Display the chart
                                    st.plotly_chart(fig_delta, use_container_width=True)
                                    
                                    # Render interpretation note
                                    _render_interpretation_note(metric_name, is_neutral_metric, increase_is_negative)
                                
                                with insight_col:
                                    # Render insight card
                                    _render_insight_card(metric_name, df, period_label, 
                                                       increase_is_negative, is_neutral_metric)

        else:
            # Separate charts for each metric
            for metric_name in sel_metrics:
                if (metric_name in multi_data and multi_data[metric_name] is not None 
                        and not multi_data[metric_name].empty):
                    df = multi_data[metric_name]
                    
                    # Create expandable section for each metric
                    with st.expander(f"**{metric_name}** Analysis", expanded=True):
                        # Calculate dynamic height
                        num_points = len(df)
                        chart_height = _calculate_dynamic_height(num_points, base_height=350)
                        
                        # Line chart for metric values
                        fig = px.line(
                            df, 
                            x="bucket", 
                            y="count", 
                            markers=True,
                            template=PLOT_TEMPLATE,
                            title=f"{metric_name} Trend",
                            labels={"bucket": "", "count": "Number of Clients"},
                            color_discrete_sequence=[_get_metric_color(metric_name)]
                        )
                        
                        # Add rolling average if enabled
                        if do_roll:
                            fig.add_scatter(
                                x=df["bucket"],
                                y=df["rolling"],
                                mode="lines",
                                name=f"{roll_window}-{_format_period_label(sel_freq_label)} average",
                                line=dict(dash="dash", color=NEUTRAL_COLOR, width=1.5),
                                opacity=0.7,
                                hovertemplate="<b>%{x|%b %Y}</b><br>Average: %{y:,.1f}<extra></extra>"
                            )
                        
                        # Apply consistent styling with dynamic height
                        fig = apply_chart_style(
                            fig,
                            xaxis_title="Time Period",
                            yaxis_title="Number of Clients",
                            height=chart_height,
                            legend_orientation="h"
                        )
                        
                        # Update layout with dynamic margins
                        margins = _calculate_dynamic_margins(num_points)
                        fig.update_layout(margin=margins)
                        
                        # Format x-axis dates
                        fig.update_xaxes(
                            tickformat="%b %Y" if sel_freq in ["M", "Q"] else ("%Y" if sel_freq == "Y" else "%b %d"),
                            tickangle=_get_x_axis_angle(sel_freq, num_points)
                        )
                        
                        # Improve hover template
                        fig.update_traces(hovertemplate=_get_hover_template(sel_freq))
                        
                        # Display the chart
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show additional analysis if we have enough data
                        if len(df) >= 2:
                            # Two-column layout
                            col1, col2 = st.columns(2)
                            
                            # Determine if increasing is good or bad
                            increase_is_negative = metric_interpretation.get(metric_name, False)
                            period_label = _format_period_label(sel_freq_label)
                            
                            with col1:
                                # Show period-over-period changes if enabled
                                if do_delta:
                                    # Check if this is a neutral metric
                                    is_neutral_metric = metric_name in neutral_metrics
                                    
                                    # Color bars based on direction
                                    bar_colors = []
                                    for x in df["delta"]:
                                        if abs(x) < 0.001:
                                            bar_colors.append("#cccccc")
                                        elif is_neutral_metric:
                                            # Use main blue color for neutral metrics
                                            bar_colors.append(MAIN_COLOR)
                                        elif (x > 0 and not increase_is_negative) or (x < 0 and increase_is_negative):
                                            bar_colors.append(SUCCESS_COLOR)
                                        else:
                                            bar_colors.append(SECONDARY_COLOR)
                                    
                                    # Calculate dynamic height for delta chart
                                    delta_height = _calculate_dynamic_height(len(df), base_height=350)
                                    
                                    # Create delta bar chart
                                    fig_delta = px.bar(
                                        df, 
                                        x="bucket", 
                                        y="delta",
                                        template=PLOT_TEMPLATE,
                                        title=f"{period_label.capitalize()}-over-{period_label} Change",
                                        labels={"bucket": "", "delta": "Change"},
                                    )
                                    
                                    # Format bars and hover
                                    fig_delta.update_traces(
                                        marker_color=bar_colors,
                                        texttemplate="%{y:+,.0f}",
                                        textfont=dict(color="white", size=11),
                                        textposition="outside",
                                        hovertemplate="<b>%{x|%b %Y}</b><br>Change: %{y:+,.0f}<extra></extra>",
                                        cliponaxis=False  # Prevent clipping of text
                                    )
                                    
                                    # Apply consistent styling with dynamic height
                                    fig_delta = apply_chart_style(
                                        fig_delta,
                                        xaxis_title="Time Period",
                                        yaxis_title="Change in Clients",
                                        height=delta_height,
                                        showlegend=False
                                    )
                                    
                                    # Update layout with dynamic margins
                                    delta_margins = _calculate_dynamic_margins(len(df), has_text_labels=True)
                                    fig_delta.update_layout(
                                        margin=delta_margins,
                                        yaxis=dict(
                                            automargin=True,
                                            rangemode="tozero",
                                            # Add padding to y-axis range for text labels
                                            range=[
                                                min(df["delta"].min() * 1.4, df["delta"].min() - 50),
                                                max(df["delta"].max() * 1.4, df["delta"].max() + 50)
                                            ]
                                        )
                                    )
                                    
                                    # Format x-axis dates
                                    fig_delta.update_xaxes(
                                        tickformat="%b %Y" if sel_freq in ["M", "Q"] else ("%Y" if sel_freq == "Y" else "%b %d"),
                                        tickangle=_get_x_axis_angle(sel_freq, len(df))
                                    )
                                    
                                    # Display the chart
                                    st.plotly_chart(fig_delta, use_container_width=True)
                                    
                                    # Render interpretation note
                                    _render_interpretation_note(metric_name, is_neutral_metric, increase_is_negative)
                            
                            with col2:
                                # Insights section with better formatting
                                insights_content = []
                                insights_content.append("""
                                <div style="background-color: rgba(31, 41, 55, 0.5); border-radius: 12px; padding: 20px;">
                                    <h3 style="color: #E5E7EB; margin-top: 0; margin-bottom: 16px;">Insights</h3>
                                """)
                                
                                # Calculate key metrics
                                first_value = df["count"].iloc[0]
                                last_value = df["count"].iloc[-1]
                                total_change = last_value - first_value
                                pct_change = (total_change / first_value * 100) if first_value > 0 else 0
                                
                                recent_periods = min(3, len(df) - 1)
                                recent_values = df["count"].iloc[-recent_periods-1:]
                                recent_change = recent_values.iloc[-1] - recent_values.iloc[0]
                                recent_pct = (recent_change / recent_values.iloc[0] * 100) if recent_values.iloc[0] > 0 else 0
                                
                                # Get trend icon
                                trend_icon = _get_trend_icon(total_change, increase_is_negative)
                                
                                # Determine trend direction
                                if total_change > 0:
                                    trend_direction = "increasing"
                                elif total_change < 0:
                                    trend_direction = "decreasing" 
                                else:
                                    trend_direction = "stable"
                                
                                # Check if this is a neutral metric
                                is_neutral_metric = metric_name in neutral_metrics
                                
                                # Determine color based on metric type
                                if is_neutral_metric:
                                    change_color = "#9CA3AF"  # Neutral gray
                                    recent_color = "#9CA3AF"
                                else:
                                    change_color = _get_trend_color(total_change, increase_is_negative)
                                    recent_color = _get_trend_color(recent_change, increase_is_negative)
                                
                                # Map color names to hex
                                color_map = {"green": "#10B981", "red": "#EF4444", "gray": "#9CA3AF"}
                                change_color = color_map.get(change_color, change_color)
                                recent_color = color_map.get(recent_color, recent_color)
                                
                                # Calculate and display volatility
                                volatility = df["pct_change"].std()
                                if volatility < 5:
                                    volatility_desc = "Very stable"
                                    volatility_color = "#10B981"
                                elif volatility < 10:
                                    volatility_desc = "Moderately stable"
                                    volatility_color = "#60A5FA"
                                elif volatility < 20:
                                    volatility_desc = "Somewhat volatile"
                                    volatility_color = "#F59E0B"
                                else:
                                    volatility_desc = "Highly volatile"
                                    volatility_color = "#EF4444"
                                
                                insights_content.append(f"""
                                    <div style="display: grid; gap: 12px;">
                                        <div style="background-color: rgba(17, 24, 39, 0.5); border-radius: 8px; padding: 12px;">
                                            <p style="margin: 0; color: #9CA3AF; font-size: 14px;">Current Direction</p>
                                            <p style="margin: 4px 0; color: #E5E7EB; font-size: 18px; font-weight: 500;">
                                                {trend_direction.capitalize()} {trend_icon}
                                            </p>
                                        </div>
                                        
                                        <div style="background-color: rgba(17, 24, 39, 0.5); border-radius: 8px; padding: 12px;">
                                            <p style="margin: 0; color: #9CA3AF; font-size: 14px;">Current Value</p>
                                            <p style="margin: 4px 0; color: #E5E7EB; font-size: 18px; font-weight: 500;">
                                                {fmt_int(last_value)} clients
                                            </p>
                                        </div>
                                        
                                        <div style="background-color: rgba(17, 24, 39, 0.5); border-radius: 8px; padding: 12px;">
                                            <p style="margin: 0; color: #9CA3AF; font-size: 14px;">Overall Change</p>
                                            <p style="margin: 4px 0; color: {change_color}; font-size: 18px; font-weight: 500;">
                                                {total_change:+,.0f} ({pct_change:+.1f}%)
                                            </p>
                                        </div>
                                        
                                        <div style="background-color: rgba(17, 24, 39, 0.5); border-radius: 8px; padding: 12px;">
                                            <p style="margin: 0; color: #9CA3AF; font-size: 14px;">Recent Change (last {recent_periods} {period_label}s)</p>
                                            <p style="margin: 4px 0; color: {recent_color}; font-size: 18px; font-weight: 500;">
                                                {recent_change:+,.0f} ({recent_pct:+.1f}%)
                                            </p>
                                        </div>
                                        
                                        <div style="background-color: rgba(17, 24, 39, 0.5); border-radius: 8px; padding: 12px;">
                                            <p style="margin: 0; color: #9CA3AF; font-size: 14px;">Volatility</p>
                                            <p style="margin: 4px 0; color: {volatility_color}; font-size: 18px; font-weight: 500;">
                                                {volatility_desc} ({volatility:.1f}%)
                                            </p>
                                        </div>
                                    </div>
                                </div>
                                """)
                                
                                st.html(''.join(insights_content))
    else:
        # Demographic breakdown - show trends by group
        for metric_name in sel_metrics:
            if (metric_name in multi_data and multi_data[metric_name] is not None 
                    and not multi_data[metric_name].empty):
                df = multi_data[metric_name]
                
                # Create expandable section for each metric
                with st.expander(f"**{metric_name}** by {sel_break}", expanded=True):
                    # Group selection - avoid using form
                    groups_key = f"group_select_{metric_name}_{key_suffix}"
                    
                    # Function to handle group selection change
                    def on_groups_change():
                        # Store the selected groups in state
                        state["trend_groups_selection"] = {
                            metric_name: st.session_state[groups_key]
                        }
                        
                    # Get all unique groups
                    all_groups = sorted(df["group"].unique())
                    
                    # Determine default groups to show
                    if len(all_groups) > 8:
                        # Try to use previously selected groups
                        if "trend_groups_selection" in state and metric_name in state["trend_groups_selection"]:
                            default_groups = state["trend_groups_selection"][metric_name]
                            # Validate groups exist in current data
                            default_groups = [g for g in default_groups if g in all_groups]
                            if not default_groups:
                                default_groups = all_groups[:8]  # Default to first 8
                        else:
                            default_groups = all_groups[:8]  # Default to first 8
                            
                        st.markdown(f"**Filter {sel_break} groups:**")
                        filtered_groups = st.multiselect(
                            f"Select {sel_break} groups to display",
                            options=all_groups,
                            default=default_groups,
                            key=groups_key,
                            on_change=on_groups_change,
                            help=f"Choose which {sel_break} groups to include in the chart"
                        )
                        
                        if filtered_groups:
                            # Apply filter
                            filtered_df = df[df["group"].isin(filtered_groups)]
                            st.caption(f"Showing {len(filtered_groups)} of {len(all_groups)} total {sel_break} groups")
                        else:
                            st.warning(f"Please select at least one {sel_break} group to display.")
                            continue
                    else:
                        # If 8 or fewer groups, show all of them
                        filtered_df = df
                        filtered_groups = all_groups
                    
                    # Display visualization if we have data
                    if not filtered_df.empty:
                        # Calculate dynamic height
                        num_points = len(filtered_df["bucket"].unique())
                        num_groups = filtered_df["group"].nunique()
                        chart_height = _calculate_dynamic_height(num_points, base_height=400)
                        
                        # Add extra height for multiple groups
                        if num_groups > 5:
                            chart_height += 50
                        
                        # Build a palette of exactly num_groups colors
                        palette = CUSTOM_COLOR_SEQUENCE[:num_groups]
                        
                        # Create line chart by group
                        fig = px.line(
                            filtered_df, 
                            x="bucket", 
                            y="count", 
                            color="group", 
                            markers=True,
                            template=PLOT_TEMPLATE,
                            title=f"{metric_name} by {sel_break}",
                            labels={
                                "bucket": "", 
                                "count": "Number of Clients", 
                                "group": sel_break
                            },
                            color_discrete_sequence=palette
                        )
                        
                        # Apply consistent styling with dynamic height
                        fig = apply_chart_style(
                            fig,
                            xaxis_title="Time Period",
                            yaxis_title="Number of Clients",
                            height=chart_height,
                            legend_orientation="h" if num_groups <= 5 else "v"
                        )
                        
                        # Update layout with dynamic margins
                        margins = _calculate_dynamic_margins(num_points)
                        fig.update_layout(margin=margins)
                        
                        # Format x-axis dates
                        fig.update_xaxes(
                            tickformat="%b %Y" if sel_freq in ["M", "Q"] else ("%Y" if sel_freq == "Y" else "%b %d"),
                            tickangle=_get_x_axis_angle(sel_freq, num_points)
                        )
                        
                        # Improved hover: Group name on top, then date, then count
                        fig.update_traces(
                            hovertemplate=(
                                "<b>%{fullData.name}</b><br>"
                                "<b>%{x|%b %Y}</b><br>"
                                "Clients: %{y:,.0f}<extra></extra>"
                            )
                        )
                        
                        # Display the chart
                        st.plotly_chart(fig, use_container_width=True)
    
                        
                        # Add growth analysis if we have enough data
                        if len(filtered_df["bucket"].unique()) >= 2:
                            try:
                                # Calculate growth rates
                                growth_df = calculate_demographic_growth(filtered_df)
                                
                                if not growth_df.empty:
                                    # Create tabs for different views
                                    tab1, tab2 = st.tabs(["📈 Growth Analysis", "📊 Data Table"])
                                    
                                    with tab1:
                                        # Two-column layout
                                        insight_col, viz_col = st.columns([1, 2])
                                        
                                        with insight_col:
                                            # Build HTML content for insights
                                            insights_content = []
                                            insights_content.append("""
                                            <div style="background-color: rgba(31, 41, 55, 0.5); border-radius: 12px; padding: 20px;">
                                                <h3 style="color: #E5E7EB; margin-top: 0; margin-bottom: 16px;">Key Findings</h3>
                                            """)
                                            
                                            if len(growth_df) >= 2:
                                                try:
                                                    # Find max and min growth groups
                                                    max_growth_idx = growth_df["growth_pct"].idxmax()
                                                    min_growth_idx = growth_df["growth_pct"].idxmin()
                                                    
                                                    if max_growth_idx is not None and min_growth_idx is not None:
                                                        # Get row data
                                                        max_growth = growth_df.loc[max_growth_idx]
                                                        min_growth = growth_df.loc[min_growth_idx]
                                                        
                                                        insights_content.append(f"""
                                                        <div style="background-color: rgba(16, 185, 129, 0.1); border-radius: 8px; padding: 12px; margin-bottom: 12px;">
                                                            <p style="margin: 0; color: #34D399; font-size: 14px; font-weight: 500;">Fastest Growing Group</p>
                                                            <p style="margin: 4px 0; color: #E5E7EB; font-size: 16px; font-weight: bold;">{max_growth['group']}</p>
                                                            <p style="margin: 4px 0; color: #34D399; font-size: 18px;">{fmt_pct(max_growth['growth_pct'])}</p>
                                                            <p style="margin: 0; color: #9CA3AF; font-size: 13px;">
                                                                {fmt_int(max_growth['first_count'])} → {fmt_int(max_growth['last_count'])} clients
                                                            </p>
                                                        </div>
                                                        """)
                                                        
                                                        # Display min growth
                                                        color = "#EF4444" if min_growth['growth_pct'] < 0 else "#F59E0B"
                                                        bg_color = "rgba(239, 68, 68, 0.1)" if min_growth['growth_pct'] < 0 else "rgba(245, 158, 11, 0.1)"
                                                        label = "Fastest Declining Group" if min_growth['growth_pct'] < 0 else "Slowest Growing Group"
                                                        
                                                        insights_content.append(f"""
                                                        <div style="background-color: {bg_color}; border-radius: 8px; padding: 12px; margin-bottom: 12px;">
                                                            <p style="margin: 0; color: {color}; font-size: 14px; font-weight: 500;">{label}</p>
                                                            <p style="margin: 4px 0; color: #E5E7EB; font-size: 16px; font-weight: bold;">{min_growth['group']}</p>
                                                            <p style="margin: 4px 0; color: {color}; font-size: 18px;">{fmt_pct(min_growth['growth_pct'])}</p>
                                                            <p style="margin: 0; color: #9CA3AF; font-size: 13px;">
                                                                {fmt_int(min_growth['first_count'])} → {fmt_int(min_growth['last_count'])} clients
                                                            </p>
                                                        </div>
                                                        """)
                                                    
                                                    # Add disparity analysis
                                                    latest_date = filtered_df["bucket"].max()
                                                    latest_values = filtered_df[filtered_df["bucket"] == latest_date]
                                                    
                                                    if not latest_values.empty and len(latest_values) >= 2:
                                                        # Find min and max groups at latest date
                                                        max_idx = latest_values["count"].idxmax()
                                                        min_idx = latest_values["count"].idxmin()
                                                        
                                                        max_group = latest_values.loc[max_idx]
                                                        min_group = latest_values.loc[min_idx]
                                                        
                                                        # Calculate disparity ratio
                                                        if min_group["count"] > 0:
                                                            disparity_ratio = max_group["count"] / min_group["count"]
                                                            
                                                            insights_content.append(f"""
                                                            <hr style="border-color: rgba(55, 65, 81, 0.5); margin: 16px 0;">
                                                            <div style="background-color: rgba(17, 24, 39, 0.5); border-radius: 8px; padding: 12px;">
                                                                <p style="margin: 0; color: #9CA3AF; font-size: 14px; font-weight: 500; margin-bottom: 8px;">Current Disparity</p>
                                                                <div style="margin-bottom: 8px;">
                                                                    <span style="color: #60A5FA;">Highest:</span> 
                                                                    <span style="color: #E5E7EB; font-weight: bold;">{max_group['group']}</span>
                                                                    <span style="color: #9CA3AF;">({fmt_int(max_group['count'])} clients)</span>
                                                                </div>
                                                                <div style="margin-bottom: 8px;">
                                                                    <span style="color: #F59E0B;">Lowest:</span> 
                                                                    <span style="color: #E5E7EB; font-weight: bold;">{min_group['group']}</span>
                                                                    <span style="color: #9CA3AF;">({fmt_int(min_group['count'])} clients)</span>
                                                                </div>
                                                                <div style="background-color: rgba(96, 165, 250, 0.1); border-radius: 6px; padding: 8px; margin-top: 8px;">
                                                                    <p style="margin: 0; color: #93C5FD; font-size: 15px; text-align: center;">
                                                                        Ratio: <strong>{disparity_ratio:.1f}x</strong> difference
                                                                    </p>
                                                                </div>
                                                            </div>
                                                            """)
                                                except Exception as e:
                                                    insights_content.append(f"""
                                                    <p style="color: #9CA3AF; font-size: 14px;">Could not calculate group insights: {str(e)}</p>
                                                    """)
                                            else:
                                                insights_content.append("""
                                                <p style="color: #60A5FA; font-size: 14px;">Not enough groups for comparison. Select more groups to see insights.</p>
                                                """)
                                            
                                            insights_content.append("</div>")
                                            st.html(''.join(insights_content))
                                        
                                        with viz_col:
                                            if len(growth_df) >= 2:
                                                # Sort for visualization - limit to top/bottom groups
                                                growth_vis_df = growth_df.sort_values("growth_pct", ascending=False)
                                                
                                                # Mark groups with significant size
                                                growth_vis_df['significant'] = growth_vis_df['first_count'] >= 5
                                                
                                                # Limit to reasonable number of groups for clarity
                                                MAX_GROUPS_TO_SHOW = 12
                                                if len(growth_vis_df) > MAX_GROUPS_TO_SHOW:
                                                    # Show top 6 and bottom 6
                                                    growth_vis_df = pd.concat([
                                                        growth_vis_df.head(6),
                                                        growth_vis_df.tail(6)
                                                    ])
                                                    
                                                    # Add note about limited display
                                                    showing_subset = True
                                                else:
                                                    showing_subset = False
                                                
                                                # Calculate dynamic height for growth chart
                                                growth_height = _calculate_dynamic_height(len(growth_vis_df), base_height=500)
                                                
                                                # Create growth bar chart with better design
                                                fig_growth = go.Figure()
                                                
                                                # Add bars with custom colors based on growth direction
                                                for idx, row in growth_vis_df.iterrows():
                                                    # Determine color based on growth
                                                    if row['growth_pct'] > 50:
                                                        color = '#4BB543'  # Strong green for high growth
                                                    elif row['growth_pct'] > 0:
                                                        color = '#90EE90'  # Light green for moderate growth
                                                    elif row['growth_pct'] > -50:
                                                        color = '#FFA500'  # Orange for moderate decline
                                                    else:
                                                        color = '#FF4B4B'  # Red for strong decline
                                                    
                                                    # Adjust opacity for small groups
                                                    opacity = 1.0 if row['significant'] else 0.6
                                                    
                                                    fig_growth.add_trace(go.Bar(
                                                        x=[row['group']],
                                                        y=[row['growth_pct']],
                                                        marker_color=color,
                                                        opacity=opacity,
                                                        hovertemplate=(
                                                            f"<b>{row['group']}</b><br>"
                                                            f"<br>"
                                                            f"<b>Growth Rate:</b> {row['growth_pct']:+.1f}%<br>"
                                                            f"<br>"
                                                            f"<b>Start Period:</b> {int(row['first_count'])} clients<br>"
                                                            f"<b>End Period:</b> {int(row['last_count'])} clients<br>"
                                                            f"<b>Net Change:</b> {int(row['growth'])} clients<br>"
                                                            f"<extra></extra>"
                                                        ),
                                                        showlegend=False
                                                    ))
                                                
                                                # Update layout with better spacing and dynamic height
                                                fig_growth.update_layout(
                                                    title=dict(
                                                        text=f"Growth Rates by {sel_break}",
                                                        font=dict(size=18),
                                                        y=0.98
                                                    ),
                                                    template=PLOT_TEMPLATE,
                                                    height=growth_height,
                                                    margin=_calculate_dynamic_margins(len(growth_vis_df), has_text_labels=True),
                                                    xaxis=dict(
                                                        title=dict(
                                                            text=sel_break,
                                                            font=dict(size=14)
                                                        ),
                                                        tickangle=-45,
                                                        automargin=True,
                                                        tickfont=dict(size=11)
                                                    ),
                                                    yaxis=dict(
                                                        title=dict(
                                                            text="Growth Rate (%)",
                                                            font=dict(size=14)
                                                        ),
                                                        automargin=True,
                                                        zeroline=True,
                                                        zerolinewidth=2,
                                                        zerolinecolor='rgba(255,255,255,0.3)',
                                                        gridcolor='rgba(255,255,255,0.1)',
                                                        # Set range to accommodate text labels
                                                        range=[min(growth_vis_df['growth_pct'].min() * 1.2, -10),
                                                               max(growth_vis_df['growth_pct'].max() * 1.2, 10)]
                                                    ),
                                                    bargap=0.3,  # More space between bars
                                                    plot_bgcolor='rgba(0,0,0,0)',
                                                    paper_bgcolor='rgba(0,0,0,0)'
                                                )
                                                
                                                # Add value labels on bars with smart positioning
                                                for i, trace in enumerate(fig_growth.data):
                                                    y_val = trace.y[0]
                                                    # Position text inside for large bars, outside for small ones
                                                    if abs(y_val) > 30:
                                                        fig_growth.data[i].update(
                                                            text=[f'{y_val:+.0f}%'],
                                                            textposition='inside',
                                                            textfont=dict(color='white', size=12, weight='bold')
                                                        )
                                                    else:
                                                        # For smaller bars, position outside
                                                        fig_growth.data[i].update(
                                                            text=[f'{y_val:+.0f}%'],
                                                            textposition='outside',
                                                            textfont=dict(color='white', size=10)
                                                        )
                                                
                                                # Apply consistent styling
                                                fig_growth = apply_chart_style(
                                                    fig_growth,
                                                    xaxis_title=sel_break,
                                                    yaxis_title="Growth Rate (%)",
                                                    height=growth_height
                                                )
                                                
                                                # Add annotations if needed
                                                if showing_subset:
                                                    fig_growth.add_annotation(
                                                        text=f"📊 Showing top 6 and bottom 6 of {len(growth_df)} total groups for clarity",
                                                        xref="paper", yref="paper",
                                                        x=0.5, y=-0.32,
                                                        showarrow=False,
                                                        font=dict(size=13, color="white", weight='bold'),
                                                        bgcolor="rgba(0,0,0,0.7)",
                                                        bordercolor="rgba(255,255,255,0.3)",
                                                        borderwidth=1,
                                                        borderpad=8
                                                    )
                                                    
                                                if not all(growth_vis_df['significant']):
                                                    fig_growth.add_annotation(
                                                        text="ℹ️ Groups with fewer than 5 clients shown with reduced opacity",
                                                        xref="paper", yref="paper",
                                                        x=0.5, y=1.08,
                                                        showarrow=False,
                                                        font=dict(size=11, color="lightgray"),
                                                        bgcolor="rgba(0,0,0,0.5)",
                                                        borderpad=4
                                                    )
                                                
                                                # Display the chart
                                                st.plotly_chart(fig_growth, use_container_width=True)
                                            else:
                                                st.info("Select at least two groups to see growth comparisons.")
                                    
                                    with tab2:
                                        # Data table
                                        if not growth_df.empty:
                                            # Format for display
                                            display_df = growth_df.copy()
                                            display_df["Growth"] = display_df["growth"].map(lambda x: fmt_int(x))
                                            display_df["Growth %"] = display_df["growth_pct"].map(lambda x: fmt_pct(x))
                                            display_df = display_df.rename(columns={
                                                "group": sel_break,
                                                "first_count": "Initial Value",
                                                "last_count": "Latest Value"
                                            })
                                            
                                            # Format count columns
                                            display_df["Initial Value"] = display_df["Initial Value"].map(fmt_int)
                                            display_df["Latest Value"] = display_df["Latest Value"].map(fmt_int)
                                            
                                            # Show table
                                            st.dataframe(
                                                display_df[[sel_break, "Initial Value", "Latest Value", "Growth", "Growth %"]],
                                                use_container_width=True,
                                                hide_index=True
                                            )
                                        else:
                                            st.info("No growth data available.")
                            except Exception as e:
                                st.warning(f"Could not calculate growth trends: {str(e)}")
                    else:
                        st.info("No data available for the selected groups. Please adjust your selection.")