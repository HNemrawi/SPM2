"""
Demographic breakdown section for HMIS dashboard.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pandas import DataFrame
from typing import Dict, List, Set, Tuple, Optional, Any

from analysis.general.data_utils import (
    DEMOGRAPHIC_DIMENSIONS, _safe_div, category_counts, inflow, 
    outflow, ph_exit_clients, return_after_exit, served_clients
)
from analysis.general.filter_utils import (
    get_filter_timestamp, hash_data, init_section_state, is_cache_valid, invalidate_cache
)
from analysis.general.theme import (
    CUSTOM_COLOR_SEQUENCE, MAIN_COLOR, PLOT_TEMPLATE, SECONDARY_COLOR, SUCCESS_COLOR,
    WARNING_COLOR, apply_chart_style, create_insight_container, fmt_int, fmt_pct
)

# Constants
BREAKDOWN_SECTION_KEY = "demographic_breakdown"

def _calculate_breakdown_data(
    df_filt: DataFrame, 
    full_df: DataFrame,
    dim_col: str,
    t0, 
    t1, 
    return_window: int = 180
) -> DataFrame:
    """
    Calculate demographic breakdown data with improved metrics.
    
    Parameters:
    -----------
    df_filt : DataFrame
        Filtered DataFrame
    full_df : DataFrame
        Full DataFrame for returns analysis
    dim_col : str
        Demographic dimension column name
    t0 : Timestamp
        Start date
    t1 : Timestamp
        End date
    return_window : int, optional
        Days to check for returns
        
    Returns:
    --------
    DataFrame
        DataFrame with breakdown metrics
    """
    # Get client sets for different metrics
    served_ids = served_clients(df_filt, t0, t1)
    inflow_ids = inflow(df_filt, t0, t1)
    outflow_ids = outflow(df_filt, t0, t1)
    ph_ids = ph_exit_clients(df_filt, t0, t1)
    
    # Get PH exits for return rate calculation
    ph_exits = set(
        df_filt.loc[
            (df_filt["ProjectExit"].between(t0, t1))
            & (df_filt["ExitDestinationCat"] == "Permanent Housing Situations"),
            "ClientID",
        ]
    )
    
    # Calculate returns
    return_ids = return_after_exit(df_filt, full_df, t0, t1, return_window)
    
    # Calculate counts by demographic dimension
    bdf = pd.concat(
        [
            category_counts(df_filt, served_ids, dim_col, "Served"),
            category_counts(df_filt, inflow_ids, dim_col, "Inflow"),
            category_counts(df_filt, outflow_ids, dim_col, "Outflow"),
            category_counts(df_filt, ph_ids, dim_col, "PH Exits"),
        ],
        axis=1,
    ).fillna(0).reset_index().rename(columns={"index": dim_col})
    
    # Skip if no data
    if bdf.empty:
        return pd.DataFrame()
    
    # Calculate PH Exit Rate where Outflow > 0
    bdf["PH Exit Rate"] = (
        bdf["PH Exits"] / bdf["Outflow"] * 100
    ).where(bdf["Outflow"] > 0).round(1)
    
    # Prepare returns by demographic
    clients_demo = df_filt[["ClientID", dim_col]].drop_duplicates()
    ph_demo = clients_demo[clients_demo["ClientID"].isin(ph_exits)]
    ret_demo = clients_demo[clients_demo["ClientID"].isin(return_ids)]
    
    # Get counts by group
    ph_counts = (
        ph_demo.groupby(dim_col)["ClientID"]
        .nunique()
        .reset_index(name="PH Exit Count")
    )
    ret_counts = (
        ret_demo.groupby(dim_col)["ClientID"]
        .nunique()
        .reset_index(name="Returns Count")
    )
    
    # Merge return counts
    returns_df = pd.merge(ph_counts, ret_counts, on=dim_col, how="left")
    returns_df["PH Exit Count"] = returns_df["PH Exit Count"].fillna(0)
    returns_df["Returns Count"] = returns_df["Returns Count"].fillna(0)
    
    # Calculate Returns Rate where PH Exit Count > 0
    returns_df["Returns to Homelessness Rate"] = (
        returns_df["Returns Count"] / returns_df["PH Exit Count"] * 100
    ).where(returns_df["PH Exit Count"] > 0).round(1)
    
    # Merge with main breakdown dataframe
    bdf = pd.merge(bdf, returns_df, on=dim_col, how="left")
    
    return bdf

def _create_counts_chart(df: DataFrame, dim_col: str) -> go.Figure:
    """
    Create counts chart for demographic breakdown.
    
    Parameters:
    -----------
    df : DataFrame
        Breakdown data
    dim_col : str
        Demographic dimension column name
        
    Returns:
    --------
    Figure
        Plotly figure
    """
    # Reshape for plotting
    counts_df = df.melt(
        id_vars=dim_col,
        value_vars=["Served", "Inflow", "Outflow", "PH Exits", "Returns Count"],
        var_name="Metric",
        value_name="Count",
    )
    
    # Create grouped bar chart
    fig = px.bar(
        counts_df,
        x=dim_col,
        y="Count",
        color="Metric",
        barmode="group",
        template=PLOT_TEMPLATE,
        text_auto=".0f",
        title=f"Client Counts by {dim_col}",
        color_discrete_sequence=CUSTOM_COLOR_SEQUENCE,
    )
    
    # Apply consistent styling
    fig = apply_chart_style(
        fig,
        xaxis_title=dim_col,
        yaxis_title="Number of Clients",
        height=500
    )
    
    return fig

def _create_rates_charts(df: DataFrame, dim_col: str, return_window: int) -> Tuple[go.Figure, go.Figure]:
    """
    Create rate charts for PH exits and returns.
    
    Parameters:
    -----------
    df : DataFrame
        Breakdown data
    dim_col : str
        Demographic dimension column name
    return_window : int
        Days to check for returns
        
    Returns:
    --------
    Tuple[Figure, Figure]
        Tuple of PH exit rate and returns rate figures
    """
    # PH Exit Rate chart
    ph_df = df.dropna(subset=["PH Exit Rate"])
    fig_ph = px.bar(
        ph_df,
        x=dim_col,
        y="PH Exit Rate",
        template=PLOT_TEMPLATE,
        text_auto=".1f",
        title="Permanent Housing Exit Rate (%)",
        color="PH Exit Rate",
        color_continuous_scale="Blues",
    )
    fig_ph.update_traces(texttemplate="%{y:.1f}%")
    
    # Apply consistent styling
    fig_ph = apply_chart_style(
        fig_ph,
        xaxis_title=dim_col,
        yaxis_title="PH Exit Rate (%)",
        height=400
    )
    
    # Returns Rate chart
    ret_df = df.dropna(subset=["Returns to Homelessness Rate"])
    if not ret_df.empty:
        fig_ret = px.bar(
            ret_df,
            x=dim_col,
            y="Returns to Homelessness Rate",
            template=PLOT_TEMPLATE,
            text_auto=".1f",
            title=f"Returns to Homelessness within {return_window} days (%)",
            color="Returns to Homelessness Rate",
            color_continuous_scale="Reds",
        )
        fig_ret.update_traces(texttemplate="%{y:.1f}%")
        
        # Apply consistent styling
        fig_ret = apply_chart_style(
            fig_ret,
            xaxis_title=dim_col,
            yaxis_title="Returns Rate (%)",
            height=400
        )
    else:
        # Create empty figure if no returns data
        fig_ret = go.Figure()
        fig_ret.update_layout(
            title=f"Returns to Homelessness within {return_window} days (%)",
            annotations=[{
                "text": "No returns data available",
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.5,
                "showarrow": False
            }]
        )
        fig_ret = apply_chart_style(
            fig_ret,
            xaxis_title=dim_col,
            yaxis_title="Returns Rate (%)",
            height=400
        )
    
    return fig_ph, fig_ret

def _create_outcome_quadrant_chart(df: DataFrame, dim_col: str) -> go.Figure:
    """
    Create outcome quadrant chart comparing PH exits and returns.
    
    Parameters:
    -----------
    df : DataFrame
        Breakdown data
    dim_col : str
        Demographic dimension column name
        
    Returns:
    --------
    Figure
        Plotly figure
    """
    # Filter to groups with both metrics
    comparison_df = df.dropna(subset=["PH Exit Rate", "Returns to Homelessness Rate"])
    
    if comparison_df.empty:
        # Create empty figure if no comparison data
        fig = go.Figure()
        fig.update_layout(
            title="PH Exit vs Return Rate: Not enough data",
            annotations=[{
                "text": "Insufficient data for comparison",
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.5,
                "showarrow": False
            }]
        )
        return apply_chart_style(fig, height=600)
    
    # Averages for quadrant lines
    avg_exit_rate = comparison_df["PH Exit Rate"].mean()
    avg_return_rate = comparison_df["Returns to Homelessness Rate"].mean()
    
    # Create scatter plot
    fig = px.scatter(
        comparison_df,
        x="PH Exit Rate",
        y="Returns to Homelessness Rate",
        size="Served",
        color=dim_col,
        hover_name=dim_col,
        template=PLOT_TEMPLATE,
        title="Success Quadrant: High PH Exits & Low Returns",
        labels={
            "PH Exit Rate": "Permanent Housing Exit Rate (%)",
            "Returns to Homelessness Rate": "Returns to Homelessness Rate (%)",
        },
        hover_data=["Served", "PH Exits", "Returns Count"],
    )
    
    # Add quadrant lines
    fig.add_shape(
        type="line",
        x0=avg_exit_rate, x1=avg_exit_rate,
        y0=comparison_df["Returns to Homelessness Rate"].min(),
        y1=comparison_df["Returns to Homelessness Rate"].max(),
        line=dict(color="white", dash="dash"),
    )
    fig.add_shape(
        type="line",
        x0=comparison_df["PH Exit Rate"].min(),
        x1=comparison_df["PH Exit Rate"].max(),
        y0=avg_return_rate, y1=avg_return_rate,
        line=dict(color="white", dash="dash"),
    )
    
    # Add quadrant annotations
    fig.add_annotation(
        x=avg_exit_rate + 5,
        y=avg_return_rate - 5,
        text="🏆 Ideal: High Exits, Low Returns",
        showarrow=False,
        font=dict(size=14, color="green"),
        bgcolor="rgba(0,0,0,0.7)",
        bordercolor="green",
        borderwidth=1,
    )
    fig.add_annotation(
        x=avg_exit_rate - 10,
        y=avg_return_rate + 5,
        text="⚠️ Concern: Low Exits, High Returns",
        showarrow=False,
        font=dict(size=14, color="red"),
        bgcolor="rgba(0,0,0,0.7)",
        bordercolor="red",
        borderwidth=1,
    )
    
    # Highlight best-performing group
    best_groups = comparison_df.loc[
        (comparison_df["PH Exit Rate"] >= avg_exit_rate) &
        (comparison_df["Returns to Homelessness Rate"] <= avg_return_rate)
    ].sort_values("PH Exit Rate", ascending=False)
    
    if not best_groups.empty:
        best_group = best_groups.iloc[0]
        best_label = best_group[dim_col]
        fig.add_annotation(
            x=best_group["PH Exit Rate"],
            y=best_group["Returns to Homelessness Rate"],
            text=f"🌟 Best: {best_label}",
            showarrow=True,
            arrowhead=2,
            ax=40,
            ay=-40,
            bgcolor="gold",
            font=dict(color="black", size=14)
        )
    
    # Apply consistent styling
    fig = apply_chart_style(
        fig, 
        xaxis_title="PH Exit Rate (%)",
        yaxis_title="Returns to Homelessness Rate (%)",
        height=650
    )
    
    return fig

@st.fragment
def render_breakdown_section(df_filt: DataFrame, full_df: Optional[DataFrame] = None) -> None:
    """
    Render demographic breakdown with enhanced visualizations.
    
    Parameters:
    -----------
    df_filt : DataFrame
        Filtered DataFrame
    full_df : DataFrame, optional
        Full DataFrame for returns analysis
    """
    # fallback
    if full_df is None:
        full_df = df_filt.copy()

    # Initialize or retrieve section state
    state: Dict[str, Any] = init_section_state(BREAKDOWN_SECTION_KEY)
    summary_state = st.session_state.get(f"state_summary_metrics", {})

    # Check if cache is valid
    filter_ts = get_filter_timestamp()
    cache_valid = is_cache_valid(state, filter_ts)
    
    if not cache_valid:
        invalidate_cache(state, filter_ts)
        state.pop("breakdown_df", None)

    # Header and description
    st.subheader("👥 Breakdown by Demographics", help="Analyze how key metrics distribute across different demographic groups")
    st.markdown("""
    Analyze how key metrics distribute across different demographic groups.
    Choose a dimension to break down the data and identify patterns or disparities.
    """)

    # Get required data from summary metrics
    served_ids = summary_state.get("served_ids", set())
    inflow_ids = summary_state.get("inflow_ids", set())
    outflow_ids = summary_state.get("outflow_ids", set())
    ph_ids = summary_state.get("ph_ids", set())
    return_window = summary_state.get("return_window", 180)

    # Check if we have the necessary data
    if not all([served_ids, inflow_ids, outflow_ids, ph_ids]):
        st.warning("Please calculate summary metrics first.")
        return

    # Choose breakdown dimension
    key_suffix = hash_data(filter_ts)
    dim_label = st.selectbox(
        "Break down by…",
        [lbl for lbl, _ in DEMOGRAPHIC_DIMENSIONS],
        key=f"breakdown_dim_{key_suffix}",
        help="Choose a demographic dimension for analysis"
    )
    dim_col = dict(DEMOGRAPHIC_DIMENSIONS)[dim_label]

    # If dimension changed, clear cache
    if state.get("selected_dimension") != dim_label:
        state["selected_dimension"] = dim_label
        state.pop("breakdown_df", None)

    # Group filter: let user focus on specific categories
    try:
        unique_groups = sorted(df_filt[dim_col].dropna().unique())
        selected_groups = st.multiselect(
            f"Focus on specific {dim_label} group(s):",
            options=unique_groups,
            default=unique_groups,
            key=f"group_filter_{key_suffix}",
            help=f"Select one or more {dim_label} values to filter results"
        )
    except Exception as e:
        st.error(f"Error loading groups: {e}")
        return

    if not selected_groups:
        st.info(f"Select one or more {dim_label} group(s) to continue.")
        return

    # Compute breakdown if needed
    if "breakdown_df" not in state:
        if dim_col not in df_filt.columns:
            st.error(f"Column '{dim_col}' not found in the dataset.")
            return

        with st.spinner(f"Calculating breakdown by {dim_label}..."):
            try:
                # Calculate breakdown data with improved metrics
                bdf = _calculate_breakdown_data(
                    df_filt, full_df, dim_col, 
                    st.session_state.get("t0"), 
                    st.session_state.get("t1"),
                    return_window
                )
                
                if bdf.empty:
                    st.info("No data available for the selected breakdown.")
                    return
                    
                # Sort by served count for consistent display
                bdf = bdf.sort_values("Served", ascending=False)
                
                # Cache the result
                state["breakdown_df"] = bdf
                
            except Exception as e:
                st.error(f"Error calculating breakdown: {e}")
                return
    else:
        bdf = state["breakdown_df"]

    # Apply group filter
    bdf = bdf[bdf[dim_col].isin(selected_groups)]

    if bdf.empty:
        st.info("No data available for the selected breakdown.")
        return

    # Display analysis in tabs
    tab_counts, tab_rates, tab_table = st.tabs(
        ["Counts", "Outcome Rates", "Data Table"]
    )

    with tab_counts:
        st.subheader("Key Metrics by Category")
        
        # Create counts chart
        fig_counts = _create_counts_chart(bdf, dim_col)
        st.plotly_chart(fig_counts, use_container_width=True)

    with tab_rates:
        # Create two-column layout for rate charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("PH Exit Rate by Category")
            fig_ph, _ = _create_rates_charts(bdf, dim_col, return_window)
            st.plotly_chart(fig_ph, use_container_width=True)

        with col2:
            st.subheader(f"Returns to Homelessness Rate ({return_window}d)")
            _, fig_ret = _create_rates_charts(bdf, dim_col, return_window)
            st.plotly_chart(fig_ret, use_container_width=True)

        # Outcome Comparison (quadrant chart)
        st.subheader("PH Exit vs Return Rate: Which Groups Are Succeeding?")
        fig_outcome = _create_outcome_quadrant_chart(bdf, dim_col)
        st.plotly_chart(fig_outcome, use_container_width=True)

    with tab_table:
        st.subheader("Detailed Data Table")
        display_cols = [
            dim_col,
            "Served",
            "Inflow",
            "Outflow",
            "PH Exits",
            "PH Exit Rate",
            "Returns Count",
            "Returns to Homelessness Rate",
        ]
        display_cols = [c for c in display_cols if c in bdf.columns]
        display_df = bdf[display_cols].copy().sort_values("Served", ascending=False)

        # Format the table
        format_dict = {
            "Served": fmt_int,
            "Inflow": fmt_int,
            "Outflow": fmt_int,
            "PH Exits": fmt_int,
            "PH Exit Rate": "{:.1f}%",
            "Returns Count": fmt_int,
            "Returns to Homelessness Rate": "{:.1f}%",
        }
        format_dict = {k: v for k, v in format_dict.items() if k in display_df.columns}

        st.dataframe(
            display_df
            .style.format(format_dict)
            .background_gradient(subset=["PH Exit Rate"], cmap="Blues")
            .background_gradient(
                subset=["Returns to Homelessness Rate"] if "Returns to Homelessness Rate" in display_df.columns else [],
                cmap="Reds"
            )
        )
    
    # Add insightful commentary
    if not bdf.empty:
        with st.expander("Analysis & Insights", expanded=True):
            # Find groups with highest/lowest PH exit rates
            ph_rate_filtered = bdf.dropna(subset=["PH Exit Rate"])
            if not ph_rate_filtered.empty:
                best_ph_idx = ph_rate_filtered["PH Exit Rate"].idxmax()
                worst_ph_idx = ph_rate_filtered["PH Exit Rate"].idxmin()
                
                if best_ph_idx is not None and worst_ph_idx is not None:
                    best_ph_row = ph_rate_filtered.loc[best_ph_idx]
                    worst_ph_row = ph_rate_filtered.loc[worst_ph_idx]
                    ph_gap = best_ph_row["PH Exit Rate"] - worst_ph_row["PH Exit Rate"]
                    
                    st.markdown(f"""
                    ### Exit Rate Analysis
                    * The **{best_ph_row[dim_col]}** group has the highest permanent housing exit rate at **{fmt_pct(best_ph_row['PH Exit Rate'])}**
                    * The **{worst_ph_row[dim_col]}** group has the lowest rate at **{fmt_pct(worst_ph_row['PH Exit Rate'])}**
                    * Gap between highest and lowest: **{fmt_pct(ph_gap)}**
                    """)
            
            # Add returns analysis if data available
            if "Returns to Homelessness Rate" in bdf.columns and not bdf["Returns Count"].sum() == 0:
                # Only analyze groups with sufficient PH exits
                returns_df = bdf[bdf["PH Exit Count"] >= 5].copy()
                
                if not returns_df.empty:
                    try:
                        best_returns_idx = returns_df["Returns to Homelessness Rate"].idxmin()
                        worst_returns_idx = returns_df["Returns to Homelessness Rate"].idxmax()
                        
                        if best_returns_idx is not None and worst_returns_idx is not None:
                            best_returns_row = returns_df.loc[best_returns_idx]
                            worst_returns_row = returns_df.loc[worst_returns_idx]
                            returns_gap = worst_returns_row["Returns to Homelessness Rate"] - best_returns_row["Returns to Homelessness Rate"]
                            
                            st.markdown(f"""
                            ### Returns to Homelessness Analysis
                            * The **{best_returns_row[dim_col]}** group has the lowest returns rate at **{fmt_pct(best_returns_row['Returns to Homelessness Rate'])}**
                            * The **{worst_returns_row[dim_col]}** group has the highest returns rate at **{fmt_pct(worst_returns_row['Returns to Homelessness Rate'])}**
                            * Gap between highest and lowest: **{fmt_pct(returns_gap)}**
                            """)
                    except Exception as e:
                        st.warning(f"Couldn't generate returns insights: {e}")
            
            # Nuanced insight based on inflow vs outflow
            try:
                inflow_max_idx = bdf["Inflow"].idxmax()
                if inflow_max_idx is not None:
                    high = bdf.loc[inflow_max_idx]
                    inflow_val = high["Inflow"]
                    outflow_val = high["Outflow"]
                    gap = inflow_val - outflow_val
                    gap_pct = (gap / outflow_val * 100) if outflow_val else 0
                    stable_threshold = 1.0  # percent
                    
                    st.markdown("### System Flow Analysis")
                    
                    if abs(gap_pct) < stable_threshold:
                        st.markdown(
                            f"* The **{high[dim_col]}** group has roughly the same inflow "
                            f"({fmt_int(inflow_val)}) and outflow ({fmt_int(outflow_val)}), indicating a stable population."
                        )
                    else:
                        direction = "higher" if gap > 0 else "lower"
                        sign = "+" if gap > 0 else ""
                        st.markdown(
                            f"* The **{high[dim_col]}** group has {abs(gap_pct):.1f}% {direction} inflow "
                            f"({fmt_int(inflow_val)}) than outflow ({fmt_int(outflow_val)}), indicating "
                            f"a {'growing' if gap > 0 else 'shrinking'} population ({sign}{fmt_int(gap)})."
                        )
            except Exception as e:
                st.warning(f"Couldn't generate flow insights: {e}")
            
            # Find groups with unusual patterns
            try:
                bdf["PH_to_outflow_ratio"] = bdf["PH Exits"] / bdf["Outflow"].replace(0, np.nan)
                unusual = bdf[bdf["PH_to_outflow_ratio"] < 0.2]  # Less than 20% of exits to PH
                
                if not unusual.empty and len(unusual) < len(bdf) / 2:  # Only if it's not too many groups
                    groups = ", ".join(f"**{row[dim_col]}**" for _, row in unusual.iterrows())
                    st.markdown(f"""
                    ### Intervention Opportunities
                    * **Exit rate intervention focus:** {groups} show unusually low PH exit rates
                      relative to their total outflow. Consider targeted strategies for these groups.
                    """)
            except Exception as e:
                st.warning(f"Couldn't generate intervention insights: {e}")
