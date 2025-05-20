"""
Summary metrics section for HMIS dashboard.
"""

from typing import Any, Dict, Optional, Set, Tuple

import pandas as pd
import streamlit as st
from pandas import DataFrame, Timestamp

from analysis.general.data_utils import (
    calc_delta, households_served, inflow, outflow, 
    ph_exit_clients, ph_exit_rate, return_after_exit, served_clients
)
from analysis.general.filter_utils import init_section_state
from analysis.general.theme import (
    MAIN_COLOR, SECONDARY_COLOR, SUCCESS_COLOR, WARNING_COLOR,
    fmt_int, fmt_pct
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
        Full DataFrame
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
    
    # Get PH exits for return rate calculation
    ph_exits_in_period = set(
        df_filt.loc[
            (df_filt["ProjectExit"].between(t0, t1))
            & (df_filt["ExitDestinationCat"] == "Permanent Housing Situations"),
            "ClientID",
        ]
    )
    
    # Calculate returns
    return_ids = return_after_exit(df_filt, full_df, t0, t1, return_window)
    
    # Previous period metrics
    served_prev = served_clients(df_filt, prev_start, prev_end)
    inflow_prev = inflow(df_filt, prev_start, prev_end)
    outflow_prev = outflow(df_filt, prev_start, prev_end)
    ph_prev = ph_exit_clients(df_filt, prev_start, prev_end)
    
    # Get PH exits for previous period return rate
    ph_exits_prev = set(
        df_filt.loc[
            (df_filt["ProjectExit"].between(prev_start, prev_end))
            & (df_filt["ExitDestinationCat"] == "Permanent Housing Situations"),
            "ClientID",
        ]
    )
    
    # Calculate previous period returns
    return_ids_prev = return_after_exit(df_filt, full_df, prev_start, prev_end, return_window)
    
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
    }

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
        # Fallback full_df to df_filt if not provided
        if full_df is None:
            full_df = df_filt.copy()

        # Initialize or retrieve cached section state
        state: Dict[str, Any] = init_section_state(SUMMARY_SECTION_KEY)
        filter_timestamp = st.session_state.get("last_filter_change", "")
        cache_valid = state.get("last_updated") == filter_timestamp

        if not cache_valid:
            state["last_updated"] = filter_timestamp

        # Header
        st.subheader("📊 Summary Metrics", help="Key performance indicators comparing current and previous periods")
        
        # Time boundaries
        t0: Timestamp = st.session_state.get("t0")
        t1: Timestamp = st.session_state.get("t1")
        prev_start: Timestamp = st.session_state.get("prev_start")
        prev_end: Timestamp = st.session_state.get("prev_end")

        if not all([t0, t1, prev_start, prev_end]):
            st.warning("Please set date ranges in the filter panel.")
            return

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

        # Compute rates
        ph_rate = ph_exit_rate(outflow_ids, ph_ids)
        ph_rate_prev = ph_exit_rate(outflow_prev, ph_prev)
        
        # Calculate return rates with safe division
        return_rate = 0 if not ph_exits_in_period else (len(return_ids) / len(ph_exits_in_period) * 100)
        return_rate_prev = 0 if not ph_exits_prev else (len(return_ids_prev) / len(ph_exits_prev) * 100)

        # Display context
        period_description = f"{t0.strftime('%b %d, %Y')} to {t1.strftime('%b %d, %Y')}"
        previous_description = f"{prev_start.strftime('%b %d, %Y')} to {prev_end.strftime('%b %d, %Y')}"
        
        st.caption(f"**Current period:** {period_description} | **Previous period:** {previous_description}")
        
        # Prepare display in rows of metrics
        row1_cols = st.columns(3)
        row2_cols = st.columns(3)
        row3_cols = st.columns(3)
        
        # Define all metrics to display
        metrics_data = [
            # Row 1
            (row1_cols[0], "Households Served", households_current, households_prev, 
             "Total households active during the period", "normal"),
            
            (row1_cols[1], "Clients Served", len(served_ids), len(served_prev), 
             "Unique clients with an active enrollment", "normal"),
            
            (row1_cols[2], "Inflow", len(inflow_ids), len(inflow_prev), 
             "Clients entering the system during this period", "normal"),
            
            # Row 2
            (row2_cols[0], "Outflow", len(outflow_ids), len(outflow_prev), 
             "Clients exiting the system during this period", "normal"),
            
            (row2_cols[1], "PH Exits", len(ph_ids), len(ph_prev), 
             "Clients exiting to permanent housing", "normal"),
            
            (row2_cols[2], "PH Exit Rate", ph_rate, ph_rate_prev, 
             "Percentage of all exits going to permanent housing", "normal"),
            
            # Row 3
            (row3_cols[0], f"Returns ({return_window}d)", len(return_ids), len(return_ids_prev), 
             f"Clients returning to homelessness after exiting to PH within {return_window} days", "inverse"),
            
            (row3_cols[1], "Return Rate", return_rate, return_rate_prev, 
             "Percentage of PH exits who return to homelessness", "inverse"),
            
            (row3_cols[2], "Net Flow", len(inflow_ids) - len(outflow_ids), 
             len(inflow_prev) - len(outflow_prev), 
             "Difference between inflow and outflow (positive means growing)", "normal")
        ]
        
        # Display each metric
        for col, label, current, previous, help_text, direction in metrics_data:
            # Handle rate metrics differently (showing percentage points change)
            if "Rate" in label:
                # For rates, show absolute pp change
                delta = current - previous
                delta_display = f"{delta:+.1f} pp" if previous is not None else "n/a"
                display_value = fmt_pct(current)
                help_msg = f"{help_text}\nChange from previous period: {delta:+.1f} percentage points"
                
                # Set color direction based on whether higher is better
                delta_color = "normal" if direction == "normal" else "inverse" 
            else:
                # For counts, show both absolute and percentage change
                if previous:
                    delta, pct = calc_delta(current, previous)
                    delta_display = f"{fmt_int(delta)} ({fmt_pct(pct)})"
                else:
                    delta_display = "n/a"
                
                display_value = fmt_int(current)
                help_msg = f"{help_text}\nChange from previous period: {delta_display}"
                
                # Set color direction based on whether higher is better
                delta_color = "normal" if direction == "normal" else "inverse"
            
            # Display the metric
            col.metric(
                label,
                display_value,
                delta_display,
                delta_color=delta_color,
                help=help_msg,
            )

        # Insights
        with st.expander("System Insights", expanded=False):
            # Calculate key insights
            max_metric_name = ""
            max_metric_pct = 0
            
            # Find metric with largest percentage change
            metrics_to_check = [
                ("Clients Served", len(served_ids), len(served_prev)),
                ("Inflow", len(inflow_ids), len(inflow_prev)),
                ("Outflow", len(outflow_ids), len(outflow_prev)),
                ("PH Exits", len(ph_ids), len(ph_prev))
            ]
            
            for name, current, previous in metrics_to_check:
                if previous:
                    _, pct = calc_delta(current, previous)
                    if abs(pct) > abs(max_metric_pct):
                        max_metric_pct = pct
                        max_metric_name = name
            
            # Generate insights based on the data
            insights = []
            
            # Flow status
            inflow_outflow_gap = len(inflow_ids) - len(outflow_ids)
            flow_status = ""
            if inflow_outflow_gap > 0:
                flow_status = f"🔴 **System growth:** Inflow exceeds outflow by **{fmt_int(inflow_outflow_gap)}** clients"
            elif inflow_outflow_gap < 0:
                flow_status = f"🟢 **System reduction:** Outflow exceeds inflow by **{fmt_int(abs(inflow_outflow_gap))}** clients"
            else:
                flow_status = "🟡 **System balanced:** Inflow equals outflow"
            
            insights.append(flow_status)
            
            # PH exit performance
            if ph_rate >= 50:
                insights.append(f"🟢 **Strong permanent housing outcomes:** {fmt_pct(ph_rate)} of exits were to permanent housing")
            elif ph_rate >= 30:
                insights.append(f"🟡 **Moderate permanent housing outcomes:** {fmt_pct(ph_rate)} of exits were to permanent housing")
            else:
                insights.append(f"🔴 **Low permanent housing outcomes:** Only {fmt_pct(ph_rate)} of exits were to permanent housing")
            
            # Returns performance
            if ph_exits_in_period:
                if return_rate < 10:
                    insights.append(f"🟢 **Strong housing stability:** Only {fmt_pct(return_rate)} of clients returned within {return_window} days")
                elif return_rate < 20:
                    insights.append(f"🟡 **Moderate housing stability:** {fmt_pct(return_rate)} of clients returned within {return_window} days")
                else:
                    insights.append(f"🔴 **Concerning housing stability:** {fmt_pct(return_rate)} of clients returned within {return_window} days")
            
            # Largest change metric
            if max_metric_name and abs(max_metric_pct) > 5:
                change_direction = "increase" if max_metric_pct > 0 else "decrease"
                insights.append(f"📊 **Notable change:** {max_metric_name} saw a {fmt_pct(abs(max_metric_pct))} {change_direction}")
            
            # Display all insights
            for insight in insights:
                st.markdown(insight)
            
            # Extra context about the data
            st.markdown("---")
            st.markdown(f"During this period, **{fmt_int(len(inflow_ids))}** clients entered and **{fmt_int(len(outflow_ids))}** exited the system.")
            
            if ph_exits_in_period:
                st.markdown(
                    f"**{fmt_int(len(return_ids))}** clients ({fmt_pct(return_rate)}) "
                    f"returned within {return_window} days of exiting to permanent housing."
                )
                st.caption("_Returns capture clients who exit to Permanent Housing and "
                            "then re-enter within the window—either in any homeless project "
                            "or in Permanent Housing when start date ≠ move-in date._")

    except Exception as e:
        st.error(f"Error in rendering summary metrics: {e}")
