"""
Trend explorer section for HMIS dashboard.
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
    CUSTOM_COLOR_SEQUENCE, MAIN_COLOR, NEUTRAL_COLOR, PLOT_TEMPLATE, SECONDARY_COLOR,
    SUCCESS_COLOR, WARNING_COLOR, apply_chart_style, fmt_int, fmt_pct
)

# Constants
TREND_SECTION_KEY = "trend_explorer"

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

    # Header and description
    st.subheader("📈 Trend Analysis")
    st.markdown("Track how metrics change over time, identify patterns, and forecast trends.")
    
    with st.expander("How to use this section"):
        st.markdown("""
        - **Compare metrics**: Select multiple metrics to analyze side-by-side
        - **Break down by demographics**: View how trends differ across groups
        - **Smooth data**: Apply rolling averages to see underlying patterns
        - **View changes**: Explore period-over-period differences
        """)

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
            help="Select multiple metrics to compare over time"
        )
    
    with filter_col2:
        # Time bucket selection
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
        
        # Frequency selectbox with properly initialized value
        sel_freq_label = st.selectbox(
            "Time bucket",
            list(FREQUENCY_MAP.keys()),
            index=list(FREQUENCY_MAP.keys()).index(freq_default),
            key=freq_key,
            on_change=on_freq_change,
            help="Choose how to group data points over time"
        )
        
        # Convert to frequency code
        sel_freq = FREQUENCY_MAP[sel_freq_label]
    
    with filter_col3:
        # Demographic breakdown selection
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
        
        # Breakdown selectbox with properly initialized value
        sel_break = st.selectbox(
            "Break down by",
            break_options,
            index=break_options.index(break_default),
            key=break_key,
            on_change=on_break_change,
            help="Choose a demographic dimension to break down the metrics"
        )
        
        # Get the actual column name
        group_col = dict(breakdown_opts)[sel_break]
    
    # REMOVED: Analysis Options section
    # Instead, set default values for previously optional settings
    
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
    st.divider()
    
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
                color_discrete_sequence=CUSTOM_COLOR_SEQUENCE
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
                            name=f"{metric_name} (avg)",
                            line=dict(dash="dash", width=1.5),
                            opacity=0.6,
                            showlegend=True
                        )
            
            # Apply consistent styling
            fig = apply_chart_style(
                fig,
                xaxis_title="Time Period",
                yaxis_title="Value",
                height=450,
                legend_orientation="h"
            )
            
            # Format x-axis dates based on frequency
            fig.update_xaxes(
                tickformat="%b %Y" if sel_freq in ["M", "Q"] else ("%Y" if sel_freq == "Y" else "%b %d"),
                tickangle=-45 if sel_freq in ["D", "W"] else 0
            )
            
            # Improve hover template
            fig.update_traces(hovertemplate="<b>%{x|%b %Y}</b><br>%{y:.0f}<extra></extra>")
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Show period changes if enabled
            if do_delta:
                with st.expander("Period-over-Period Changes", expanded=True):
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
                                    # Color bars based on direction
                                    bar_colors = []
                                    for x in df["delta"]:
                                        if abs(x) < 0.001:
                                            bar_colors.append("#cccccc")
                                        elif (x > 0 and not increase_is_negative) or (x < 0 and increase_is_negative):
                                            bar_colors.append(SUCCESS_COLOR)
                                        else:
                                            bar_colors.append(SECONDARY_COLOR)
                                    
                                    # Create delta bar chart
                                    fig_delta = px.bar(
                                        df, 
                                        x="bucket", 
                                        y="delta",
                                        template=PLOT_TEMPLATE,
                                        title=f"Change in {metric_name}",
                                        labels={"bucket": "", "delta": "Period Change"},
                                    )
                                    
                                    # Format bars and hover
                                    fig_delta.update_traces(
                                        marker_color=bar_colors,
                                        texttemplate="%{y:+.0f}",
                                        hovertemplate="<b>%{x|%b %Y}</b><br>Change: %{y:+.0f}<extra></extra>"
                                    )
                                    
                                    # Apply consistent styling
                                    fig_delta = apply_chart_style(
                                        fig_delta,
                                        xaxis_title="Time Period",
                                        yaxis_title="Change",
                                        height=300,
                                        showlegend=False
                                    )
                                    
                                    # Format x-axis dates
                                    fig_delta.update_xaxes(
                                        tickformat="%b %Y" if sel_freq in ["M", "Q"] else ("%Y" if sel_freq == "Y" else "%b %d"),
                                        tickangle=-45 if sel_freq in ["D", "W"] else 0
                                    )
                                    
                                    # Display the chart
                                    st.plotly_chart(fig_delta, use_container_width=True)
                                    
                                    # Note about interpretation
                                    if increase_is_negative:
                                        st.caption("*Note: For this metric, increases are considered negative*")
                                    else:
                                        st.caption("*Note: For this metric, increases are considered positive*")
                                
                                with insight_col:
                                    # Calculate insights
                                    avg_change = df["delta"].mean()
                                    total_change = df["delta"].sum()
                                    recent_periods = min(3, len(df) - 1)
                                    recent_change = df["delta"].iloc[-recent_periods:].mean()
                                    
                                    max_increase = df["delta"].max()
                                    max_increase_date = df.loc[df["delta"].idxmax(), "bucket"]
                                    max_decrease = df["delta"].min()
                                    max_decrease_date = df.loc[df["delta"].idxmin(), "bucket"]
                                    
                                    # Determine trend direction
                                    if total_change > 0:
                                        trend_direction = "increasing"
                                        trend_icon = "📈"
                                    elif total_change < 0:
                                        trend_direction = "decreasing" 
                                        trend_icon = "📉"
                                    else:
                                        trend_direction = "stable"
                                        trend_icon = "➡️"
                                    
                                    # Display insights
                                    st.markdown(f"### Key Insights {trend_icon}")
                                    
                                    st.markdown(f"""
                                    **Current trend:** {trend_direction.capitalize()}
                                    
                                    **Average change:** {avg_change:.1f} per period
                                    
                                    **Total change:** {total_change:.0f} over all periods
                                    """)
                                    
                                    st.markdown("**Notable changes:**")
                                    st.markdown(f"- Largest increase: **{max_increase:.0f}** ({max_increase_date:%b %Y})")
                                    st.markdown(f"- Largest decrease: **{max_decrease:.0f}** ({max_decrease_date:%b %Y})")
                                    
                                    # Interpretation note
                                    st.markdown("---")
                                    if increase_is_negative:
                                        st.markdown("*Note: For this metric, increases are considered negative.*")

        else:
            # Separate charts for each metric
            for metric_name in sel_metrics:
                if (metric_name in multi_data and multi_data[metric_name] is not None 
                        and not multi_data[metric_name].empty):
                    df = multi_data[metric_name]
                    
                    # Create expandable section for each metric
                    with st.expander(f"**{metric_name}** Analysis", expanded=True):
                        # Line chart for metric values
                        fig = px.line(
                            df, 
                            x="bucket", 
                            y="count", 
                            markers=True,
                            template=PLOT_TEMPLATE,
                            title=f"{metric_name} Trend",
                            labels={"bucket": "", "count": "Value"},
                            color_discrete_sequence=[MAIN_COLOR]
                        )
                        
                        # Add rolling average if enabled
                        if do_roll:
                            fig.add_scatter(
                                x=df["bucket"],
                                y=df["rolling"],
                                mode="lines",
                                name=f"{roll_window}-period average",
                                line=dict(dash="dash", color=NEUTRAL_COLOR, width=1.5),
                                opacity=0.7,
                                hovertemplate="<b>%{x|%b %Y}</b><br>Average: %{y:.1f}<extra></extra>"
                            )
                        
                        # Apply consistent styling
                        fig = apply_chart_style(
                            fig,
                            xaxis_title="Time Period",
                            yaxis_title="Value",
                            height=350,
                            legend_orientation="h"
                        )
                        
                        # Format x-axis dates
                        fig.update_xaxes(
                            tickformat="%b %Y" if sel_freq in ["M", "Q"] else ("%Y" if sel_freq == "Y" else "%b %d"),
                            tickangle=-45 if sel_freq in ["D", "W"] else 0
                        )
                        
                        # Improve hover template
                        fig.update_traces(hovertemplate="<b>%{x|%b %Y}</b><br>Value: %{y:.0f}<extra></extra>")
                        
                        # Display the chart
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show additional analysis if we have enough data
                        if len(df) >= 2:
                            # Two-column layout
                            col1, col2 = st.columns(2)
                            
                            # Determine if increasing is good or bad - MOVED HERE
                            increase_is_negative = metric_interpretation.get(metric_name, False)
                            
                            with col1:
                                # Show period-over-period changes if enabled
                                if do_delta:
                                    # Color bars based on direction
                                    bar_colors = []
                                    for x in df["delta"]:
                                        if abs(x) < 0.001:
                                            bar_colors.append("#cccccc")
                                        elif (x > 0 and not increase_is_negative) or (x < 0 and increase_is_negative):
                                            bar_colors.append(SUCCESS_COLOR)
                                        else:
                                            bar_colors.append(SECONDARY_COLOR)
                                    
                                    # Create delta bar chart
                                    fig_delta = px.bar(
                                        df, 
                                        x="bucket", 
                                        y="delta",
                                        template=PLOT_TEMPLATE,
                                        title="Period-over-Period Change",
                                        labels={"bucket": "", "delta": "Change"},
                                    )
                                    
                                    # Format bars and hover
                                    fig_delta.update_traces(
                                        marker_color=bar_colors,
                                        texttemplate="%{y:+.0f}",
                                        hovertemplate="<b>%{x|%b %Y}</b><br>Change: %{y:+.0f}<extra></extra>"
                                    )
                                    
                                    # Apply consistent styling
                                    fig_delta = apply_chart_style(
                                        fig_delta,
                                        xaxis_title="Time Period",
                                        yaxis_title="Change",
                                        height=300,
                                        showlegend=False
                                    )
                                    
                                    # Format x-axis dates
                                    fig_delta.update_xaxes(
                                        tickformat="%b %Y" if sel_freq in ["M", "Q"] else ("%Y" if sel_freq == "Y" else "%b %d"),
                                        tickangle=-45 if sel_freq in ["D", "W"] else 0
                                    )
                                    
                                    # Display the chart
                                    st.plotly_chart(fig_delta, use_container_width=True)
                                    
                                    # Note about interpretation
                                    if increase_is_negative:
                                        st.caption("*Note: For this metric, increases are considered negative*")
                                    else:
                                        st.caption("*Note: For this metric, increases are considered positive*")
                            
                            with col2:
                                # Insights section
                                st.subheader("Insights")
                                
                                # Calculate key metrics
                                first_value = df["count"].iloc[0]
                                last_value = df["count"].iloc[-1]
                                total_change = last_value - first_value
                                pct_change = (total_change / first_value * 100) if first_value > 0 else 0
                                
                                recent_periods = min(3, len(df) - 1)
                                recent_values = df["count"].iloc[-recent_periods-1:]
                                recent_change = recent_values.iloc[-1] - recent_values.iloc[0]
                                recent_pct = (recent_change / recent_values.iloc[0] * 100) if recent_values.iloc[0] > 0 else 0
                                
                                # Determine trend direction
                                if total_change > 0:
                                    trend_direction = "increasing"
                                    trend_icon = "📈"
                                elif total_change < 0:
                                    trend_direction = "decreasing" 
                                    trend_icon = "📉"
                                else:
                                    trend_direction = "stable"
                                    trend_icon = "➡️"
                                
                                # Display trend insights
                                st.markdown(f"**Current direction:** {trend_direction.capitalize()} {trend_icon}")
                                st.markdown(f"**Current value:** {last_value:.0f}")
                                
                                # Determine color based on whether increases are good
                                change_color = "green" if (total_change > 0 and not increase_is_negative) or \
                                                         (total_change < 0 and increase_is_negative) else "red"
                                                         
                                # Display overall change
                                st.markdown(f"**Overall change:** <span style='color:{change_color};'>{total_change:+.0f} ({pct_change:+.1f}%)</span> since start", unsafe_allow_html=True)
                                
                                # Determine color for recent change
                                recent_color = "green" if (recent_change > 0 and not increase_is_negative) or \
                                                         (recent_change < 0 and increase_is_negative) else "red"
                                
                                # Display recent change
                                st.markdown(f"**Recent change:** <span style='color:{recent_color};'>{recent_change:+.0f} ({recent_pct:+.1f}%)</span> over last {recent_periods} periods", unsafe_allow_html=True)
                                
                                # Calculate and display volatility
                                volatility = df["pct_change"].std()
                                if volatility < 5:
                                    volatility_desc = "Very stable"
                                elif volatility < 10:
                                    volatility_desc = "Moderately stable"
                                elif volatility < 20:
                                    volatility_desc = "Somewhat volatile"
                                else:
                                    volatility_desc = "Highly volatile"
                                
                                st.markdown(f"**Volatility:** {volatility_desc} ({volatility:.1f}%)")
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
                                "count": "Value", 
                                "group": sel_break
                            },
                            line_shape="spline"
                        )
                        
                        # Apply consistent styling
                        fig = apply_chart_style(
                            fig,
                            xaxis_title="Time Period",
                            yaxis_title="Value",
                            height=400,
                            legend_orientation="h" if len(filtered_df["group"].unique()) <= 5 else "v"
                        )
                        
                        # Format x-axis dates
                        fig.update_xaxes(
                            tickformat="%b %Y" if sel_freq in ["M", "Q"] else ("%Y" if sel_freq == "Y" else "%b %d"),
                            tickangle=-45 if sel_freq in ["D", "W"] else 0
                        )
                        
                        # Improve hover template
                        fig.update_traces(
                            hovertemplate="<b>%{x|%b %Y}</b><br>%{y:.0f}<extra>%{fullData.name}</extra>"
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
                                    tab1, tab2 = st.tabs(["Growth Analysis", "Data Table"])
                                    
                                    with tab1:
                                        # Two-column layout
                                        insight_col, viz_col = st.columns([1, 2])
                                        
                                        with insight_col:
                                            st.subheader("Key Findings")
                                            
                                            if len(growth_df) >= 2:
                                                try:
                                                    # Find max and min growth groups
                                                    max_growth_idx = growth_df["growth_pct"].idxmax()
                                                    min_growth_idx = growth_df["growth_pct"].idxmin()
                                                    
                                                    if max_growth_idx is not None and min_growth_idx is not None:
                                                        # Get row data
                                                        max_growth = growth_df.loc[max_growth_idx]
                                                        min_growth = growth_df.loc[min_growth_idx]
                                                        
                                                        # Display max growth
                                                        st.markdown(f"**Fastest growing:**")
                                                        st.markdown(f"**{max_growth['group']}**")
                                                        st.markdown(f"+{max_growth['growth_pct']:.1f}% increase")
                                                        st.markdown(f"From {max_growth['first_count']:.0f} to {max_growth['last_count']:.0f}")
                                                        
                                                        st.markdown("---")
                                                        
                                                        # Display min growth
                                                        st.markdown(f"**Fastest declining:**")
                                                        st.markdown(f"**{min_growth['group']}**")
                                                        st.markdown(f"{min_growth['growth_pct']:.1f}% change")
                                                        st.markdown(f"From {min_growth['first_count']:.0f} to {min_growth['last_count']:.0f}")
                                                    
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
                                                            
                                                            st.markdown("---")
                                                            st.markdown("**Disparity Analysis:**")
                                                            st.markdown(f"Highest: **{max_group['group']}** ({max_group['count']:.0f})")
                                                            st.markdown(f"Lowest: **{min_group['group']}** ({min_group['count']:.0f})")
                                                            st.markdown(f"Ratio: **{disparity_ratio:.1f}x**")
                                                except Exception as e:
                                                    st.markdown(f"Could not calculate group insights: {str(e)}")
                                            else:
                                                st.info("Not enough groups for comparison. Select more groups to see insights.")
                                        
                                        with viz_col:
                                            if len(growth_df) >= 2:
                                                # Sort for visualization
                                                growth_vis_df = growth_df.sort_values("growth_pct", ascending=False)
                                                
                                                # Mark groups with significant size
                                                growth_vis_df['significant'] = growth_vis_df['first_count'] >= 5
                                                
                                                # Limit to reasonable number of groups
                                                if len(growth_vis_df) > 15:
                                                    growth_vis_df = pd.concat([
                                                        growth_vis_df.head(8),
                                                        growth_vis_df.tail(7)
                                                    ])
                                                
                                                # Create growth bar chart
                                                fig_growth = px.bar(
                                                    growth_vis_df,
                                                    x="group",
                                                    y="growth_pct",
                                                    title=f"Growth Rates by {sel_break}",
                                                    labels={"group": sel_break, "growth_pct": "Growth (%)", "significant": "Significant Count"},
                                                    color="growth_pct",
                                                    color_continuous_scale=["#ff4b4b", "#f7dc6f", "#4bb543"],
                                                    height=400,
                                                    opacity=growth_vis_df['significant'].map({True: 1.0, False: 0.4})
                                                )
                                                
                                                # Add note about opacity
                                                if not all(growth_vis_df['significant']):
                                                    fig_growth.add_annotation(
                                                        text="Groups with fewer than 5 members are shown with lower opacity",
                                                        xref="paper", yref="paper",
                                                        x=0.5, y=-0.15,
                                                        showarrow=False,
                                                        font=dict(size=10, color="gray")
                                                    )
                                                
                                                # Improve hover and text
                                                fig_growth.update_traces(
                                                    texttemplate="%{y:+.1f}%",
                                                    hovertemplate="<b>%{x}</b><br>Growth: %{y:+.1f}%<br>Count: %{customdata}<extra></extra>",
                                                    customdata=growth_vis_df[['first_count', 'last_count']].astype(int).values
                                                )
                                                
                                                # Apply consistent styling
                                                fig_growth = apply_chart_style(
                                                    fig_growth,
                                                    xaxis_title=sel_break,
                                                    yaxis_title="Growth Rate (%)",
                                                    height=400
                                                )
                                                
                                                # Handle long labels
                                                if max(len(str(x)) for x in growth_vis_df["group"]) > 10:
                                                    fig_growth.update_layout(
                                                        xaxis_tickangle=-45
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
                                            display_df["Growth"] = display_df["growth"].map(lambda x: f"{x:+.0f}")
                                            display_df["Growth %"] = display_df["growth_pct"].map(lambda x: f"{x:+.1f}%")
                                            display_df = display_df.rename(columns={
                                                "group": sel_break,
                                                "first_count": "Initial Value",
                                                "last_count": "Latest Value"
                                            })
                                            
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