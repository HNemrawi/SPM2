"""
Length of stay analysis section for HMIS dashboard.
"""

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pandas import DataFrame, Timestamp

from analysis.general.data_utils import DEMOGRAPHIC_DIMENSIONS
from analysis.general.filter_utils import (
    get_filter_timestamp, hash_data, init_section_state, is_cache_valid, invalidate_cache
)
from analysis.general.theme import (
    CUSTOM_COLOR_SEQUENCE, MAIN_COLOR, NEUTRAL_COLOR, PLOT_TEMPLATE, SECONDARY_COLOR,
    apply_chart_style, create_insight_container, fmt_int, fmt_pct
)

# Constants
LOS_SECTION_KEY = "length_of_stay"

def length_of_stay(df: DataFrame, start: Timestamp, end: Timestamp) -> Dict[str, Any]:
    """
    Calculate length of stay statistics for all enrollments.
    
    Parameters:
    -----------
    df : DataFrame
        Filtered dataframe containing enrollment data
    start : Timestamp
        Start date of reporting period
    end : Timestamp
        End date of reporting period
        
    Returns:
    --------
    Dict containing:
        - los_by_enrollment: DataFrame with LOS for each enrollment
        - avg_los: Average length of stay
        - median_los: Median length of stay
        - distribution: DataFrame with counts by LOS category
    """
    # Validate required columns
    required_cols = ["EnrollmentID", "ProjectStart", "ProjectExit", "ClientID"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
    
    # Filter active enrollments
    active = df[
        ((df["ProjectExit"] >= start) | df["ProjectExit"].isna())
        & (df["ProjectStart"] <= end)
    ].copy()
    
    # Handle empty dataframe
    if active.empty:
        empty_dist = pd.DataFrame(columns=["LOS_Category", "count"])
        empty_los = pd.DataFrame(columns=["EnrollmentID", "ClientID", "LOS"])
        return {
            "avg_los": 0.0,
            "median_los": 0.0,
            "distribution": empty_dist,
            "los_by_enrollment": empty_los,
        }
    
    # Calculate LOS
    active["LOS"] = (
        (active["ProjectExit"].fillna(end) - active["ProjectStart"]).dt.days + 1
    )
    
    # Handle potential data errors - LOS should be at least 1 day
    active.loc[active["LOS"] < 1, "LOS"] = 1
    
    # Calculate average and median LOS
    avg_los = round(active["LOS"].mean(), 1)
    median_los = round(active["LOS"].median(), 1)
    
    # Create LOS distribution categories
    bins = [0, 7, 30, 90, 180, 365, float('inf')]
    labels = ['0–7 days', '8–30 days', '31–90 days', '91–180 days', '181–365 days', '365+ days']
    
    # Count by category
    dist_df = (
        active["LOS"]
        .pipe(pd.cut, bins=bins, labels=labels, right=False)
        .value_counts(sort=False)
        .rename_axis("LOS_Category")
        .reset_index(name="count")
    )
    
    return {
        "los_by_enrollment": active[["EnrollmentID", "ClientID", "LOS"]],
        "avg_los": avg_los,
        "median_los": median_los,
        "distribution": dist_df
    }

def los_by_demographic(
    df: DataFrame,
    dim_col: str,
    t0: Timestamp,
    t1: Timestamp, 
    min_group_size: int = 10
) -> DataFrame:
    """
    Calculate length of stay statistics by demographic group.
    
    Parameters:
    -----------
    df : DataFrame
        Filtered dataframe
    dim_col : str
        Demographic dimension column
    t0 : Timestamp
        Start date
    t1 : Timestamp
        End date
    min_group_size : int
        Minimum group size to include
        
    Returns:
    --------
    DataFrame
        LOS statistics by demographic group
    """
    # Validate the column exists
    if dim_col not in df.columns:
        raise ValueError(f"Column '{dim_col}' not found in dataframe")
    
    # Get basic LOS calculation
    los_data = length_of_stay(df, t0, t1)
    
    # Get LOS by enrollment with demographic info
    los_df = los_data["los_by_enrollment"].copy()
    
    # Add demographic info
    demo_map = df[["ClientID", dim_col]].copy()
    
    # Handle missing values
    demo_map[dim_col] = demo_map[dim_col].fillna("Not Reported")
    
    # Get most common value for each client if there are duplicates
    if demo_map.duplicated("ClientID").any():
        # Get most common value for each client
        demo_mode = demo_map.groupby("ClientID")[dim_col].agg(
            lambda x: x.mode().iloc[0] if not x.mode().empty else "Not Reported"
        ).reset_index()
        demo_map = demo_mode
    
    # Merge with LOS data
    los_with_demo = los_df.merge(demo_map, on="ClientID", how="left")
    
    # Fill any remaining NAs
    los_with_demo[dim_col] = los_with_demo[dim_col].fillna("Not Reported")
    
    # Calculate summary statistics by group
    los_by_demo = los_with_demo.groupby(dim_col)["LOS"].agg(
        # Use safe aggregations that handle empty groups
        mean=lambda x: x.mean() if len(x) > 0 else 0,
        median=lambda x: x.median() if len(x) > 0 else 0,
        count="count",
        q1=lambda x: x.quantile(0.25) if len(x) > 0 else 0,
        q3=lambda x: x.quantile(0.75) if len(x) > 0 else 0
    ).reset_index()
    
    # Filter to groups that meet minimum size
    los_by_demo = los_by_demo[los_by_demo["count"] >= min_group_size]
    
    # Calculate total bed days by group
    los_by_demo["total_days"] = los_by_demo["mean"] * los_by_demo["count"]
    
    # Calculate disparity index relative to overall average
    if los_data['avg_los'] > 0:
        los_by_demo["disparity"] = (los_by_demo["mean"] / los_data['avg_los']).round(2)
    else:
        los_by_demo["disparity"] = 1.0
    
    # Calculate IQR (interquartile range)
    los_by_demo["iqr"] = los_by_demo["q3"] - los_by_demo["q1"]
    
    # Sort by mean LOS (descending)
    los_by_demo = los_by_demo.sort_values("mean", ascending=False)
    
    return los_by_demo

@st.fragment
def render_length_of_stay(df_filt: DataFrame) -> None:
    """
    Render the length of stay analysis section with enhanced visualizations.
    
    Parameters:
    -----------
    df_filt : DataFrame
        Filtered DataFrame
    """
    # Initialize section state
    state = init_section_state(LOS_SECTION_KEY)
    
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
    st.subheader("⏱️ Length of Stay Analysis")
    st.markdown("""
    Analyze how long clients remain in programs before exiting. Use this analysis to identify:
    
    - Patterns in service utilization across different demographic groups
    - Distribution patterns that may indicate system bottlenecks
    - Opportunities to improve flow through the homeless system
    
    **How LOS is calculated:**  
    LOS = (ProjectExit − ProjectStart).days + 1.  
    If ProjectExit is missing (i.e. client is still enrolled), we use the **current reporting end date** in place of ProjectExit.
    """)
    
    # Get time boundaries
    t0 = st.session_state.get("t0")
    t1 = st.session_state.get("t1")
    
    if not (t0 and t1):
        st.warning("Please set date ranges in the filter panel.")
        return
    
    # Calculate LOS data if needed
    if "los_data" not in state:
        with st.spinner("Calculating length of stay statistics..."):
            try:
                los_data = length_of_stay(df_filt, t0, t1)
                
                # Validate los_data structure and handle potential issues
                if not isinstance(los_data, dict):
                    st.error("LOS calculation produced invalid data structure.")
                    return
                
                required_keys = ["los_by_enrollment", "avg_los", "median_los", "distribution"]
                if not all(key in los_data for key in required_keys):
                    st.error("LOS calculation missing required data components.")
                    return
                
                # Ensure critical dataframes aren't empty
                if los_data["los_by_enrollment"].empty:
                    st.info("No enrollment data available for length of stay analysis.")
                    return
                    
                state["los_data"] = los_data
            except Exception as e:
                st.error(f"Error calculating LOS: {str(e)}")
                return
    else:
        los_data = state["los_data"]
    
    # Create tabs for different views
    tab_overview, tab_demo = st.tabs(["Overview", "Demographics Breakdown"])
    
    with tab_overview:
        # Display summary metrics with more context
        col_a, col_b, col_c = st.columns(3)
        
        # Calculate additional benchmark metrics with robust handling
        all_los = los_data["los_by_enrollment"]["LOS"]
        
        # Safe calculation of percentiles with empty check
        q1 = all_los.quantile(0.25) if not all_los.empty else 0
        q3 = all_los.quantile(0.75) if not all_los.empty else 0
        iqr = q3 - q1
        
        col_a.metric(
            "Average LOS", 
            f"{los_data['avg_los']:.1f} days",
            help="Mean number of days clients stayed in programs"
        )
        col_b.metric(
            "Median LOS", 
            f"{los_data['median_los']:.1f} days",
            help="Middle value of all stay durations (less affected by outliers)"
        )
        col_c.metric(
            "Interquartile Range",
            f"{int(q1)}-{int(q3)} days",
            help="Middle 50% of stays fall within this range"
        )
        
        if los_data["distribution"].empty:
            st.info("No length of stay data available for the selected period.")
            return
        
        # Display LOS distribution with improved visualization
        st.subheader("LOS Distribution")
        
        # Create a better chart with color gradient based on stay length
        dist_df = los_data["distribution"].copy()
        
        # Add a numeric order column for coloring
        category_order = {
            "0–7 days": 1,
            "8–30 days": 2,
            "31–90 days": 3,
            "91–180 days": 4,
            "181–365 days": 5,
            "365+ days": 6
        }
        dist_df["category_order"] = dist_df["LOS_Category"].map(category_order)
        
        # Calculate percentage for better context
        total_count = dist_df["count"].sum()
        if total_count > 0:  # Avoid division by zero
            dist_df["percentage"] = (dist_df["count"] / total_count * 100).round(1)
        else:
            dist_df["percentage"] = 0
        
        # Create enhanced distribution chart
        dist_fig = px.bar(
            dist_df.sort_values("category_order"),
            x="LOS_Category",
            y="count",
            color_discrete_sequence=[MAIN_COLOR],
            text=dist_df["percentage"].apply(lambda x: f"{x:.1f}%" if total_count > 0 else "0%"),
            title="Length of Stay Distribution",
            labels={
                "LOS_Category": "Length of Stay", 
                "count": "Number of Enrollments"
            }
        )
        
        # Apply consistent styling
        dist_fig = apply_chart_style(
            dist_fig,
            xaxis_title="Length of Stay Category",
            yaxis_title="Enrollments",
            height=400
        )
        
        # Make the text more readable
        dist_fig.update_traces(
            textposition="outside",
            textfont=dict(size=14),
            hovertemplate="<b>%{x}</b><br>Enrollments: %{y}<br>Percentage: %{text}"
        )
        
        # Display the chart
        st.plotly_chart(dist_fig, use_container_width=True)
        
        # Add enhanced insights based on distribution
        with st.expander("Distribution Insights", expanded=True):
            # Find most common category
            if not dist_df.empty:
                largest_bucket = dist_df.loc[dist_df["count"].idxmax()]
                
                # Find stay percentages by category
                short_stay_pct = (
                    dist_df[dist_df["LOS_Category"].isin(["0–7 days", "8–30 days"])]
                    ["percentage"].sum()
                )
                
                medium_stay_pct = (
                    dist_df[dist_df["LOS_Category"].isin(["31–90 days", "91–180 days"])]
                    ["percentage"].sum()
                )
                
                long_stay_pct = (
                    dist_df[dist_df["LOS_Category"].isin(["181–365 days", "365+ days"])]
                    ["percentage"].sum()
                )
                
                # Create a pie chart for stay length distribution
                stay_categories = ["Short (<30 days)", "Medium (31-180 days)", "Long (>180 days)"]
                stay_values = [short_stay_pct, medium_stay_pct, long_stay_pct]
                
                # Use defined color palette
                colors = [MAIN_COLOR, NEUTRAL_COLOR, SECONDARY_COLOR]
                
                fig_pie = px.pie(
                    values=stay_values,
                    names=stay_categories,
                    color_discrete_sequence=colors,
                    title="Stay Length Categories"
                )
                
                # Format the pie chart
                fig_pie.update_traces(
                    textinfo="percent+label",
                    textfont_size=14,
                    hole=0.4,  # Create a donut chart
                    pull=[0.03, 0, 0]  # Slightly pull out the first slice
                )
                
                # Apply consistent styling
                fig_pie = apply_chart_style(fig_pie, height=300)
                
                # Create a visualized insight section
                col_insight, col_pie = st.columns([3, 2])
                
                with col_insight:
                    st.markdown(f"""
                    ### Key Observations
                    * **{largest_bucket['LOS_Category']}** is the most common length of stay (**{fmt_int(largest_bucket['count'])}** enrollments or **{largest_bucket['percentage']:.1f}%**)
                    * **{fmt_pct(short_stay_pct)}** of enrollments are short-term (<30 days)
                    * **{fmt_pct(medium_stay_pct)}** are medium-term (31-180 days)
                    * **{fmt_pct(long_stay_pct)}** are long-term (>180 days)
                    
                    ### Stay Length Distribution
                    * The median stay (**{los_data['median_los']:.1f} days**) is {'longer' if los_data['median_los'] > los_data['avg_los'] else 'shorter'} than the average (**{los_data['avg_los']:.1f} days**)
                    """)
                    
                    if los_data['median_los'] < los_data['avg_los']:
                        st.markdown("""
                        * **Statistical note:** The average being higher than the median indicates a right-skewed distribution with some extremely long stays influencing the average.
                        """)
                    elif los_data['median_los'] > los_data['avg_los']:
                        st.markdown("""
                        * **Statistical note:** The median being higher than the average indicates a left-skewed distribution with a high number of very short stays.
                        """)
                
                with col_pie:
                    st.plotly_chart(fig_pie)
            else:
                st.info("No distribution data available for insights.")
        
        # Display histogram of actual LOS values
        st.subheader("Detailed LOS Histogram")
        
        # Create bins for the histogram
        if not all_los.empty:
            max_los = all_los.max()
            
            # Create a copy with capped values for visualization
            viz_data = los_data["los_by_enrollment"].copy()
            los_cap = 730  # Cap at 2 years for visualization
            
            # Determine bin size dynamically based on data range
            if max_los <= 90:
                bin_size = 7  # Weekly bins for shorter stays
                bin_text = "week"
            elif max_los <= 365:
                bin_size = 30  # Monthly bins for medium stays
                bin_text = "month"
            else:
                bin_size = 90  # Quarterly bins for longer stays
                bin_text = "quarter"
            
            num_bins = min(15, max(6, int(min(max_los, los_cap) / bin_size) + 1))
            
            # Add message about capped values if needed
            if max_los > los_cap:
                above_cap_count = (viz_data["LOS"] > los_cap).sum()
                above_cap_pct = (above_cap_count / len(viz_data) * 100).round(1)
                cap_message = f" (Note: {above_cap_count} enrollments ({above_cap_pct}%) with stays > {los_cap} days are grouped at {los_cap})" if above_cap_count > 0 else ""
                # Cap values for visualization
                viz_data.loc[viz_data["LOS"] > los_cap, "LOS"] = los_cap
            else:
                cap_message = ""
            
            # Create histogram
            hist_fig = px.histogram(
                viz_data,
                x="LOS",
                nbins=num_bins,
                title=f"Length of Stay Distribution by {bin_text}{cap_message}",
                labels={"LOS": "Length of Stay (days)"},
                opacity=0.8,
                color_discrete_sequence=[MAIN_COLOR]
            )
            
            # Apply consistent styling
            hist_fig = apply_chart_style(
                hist_fig,
                xaxis_title="Length of Stay (days)",
                yaxis_title="Number of Enrollments",
                height=400
            )
            
            # Add mean and median lines
            hist_fig.add_vline(
                x=los_data['median_los'],
                line=dict(dash="dash", color=SECONDARY_COLOR, width=2),
                annotation=dict(
                    text=f"Median: {los_data['median_los']:.1f} days",
                    font=dict(size=14, color=SECONDARY_COLOR),
                    bgcolor="rgba(0,0,0,0.7)",
                    bordercolor=SECONDARY_COLOR,
                    borderwidth=1
                ),
                annotation_position="top right"
            )
            
            hist_fig.add_vline(
                x=los_data['avg_los'],
                line=dict(dash="dash", color="white", width=2),
                annotation=dict(
                    text=f"Average: {los_data['avg_los']:.1f} days",
                    font=dict(size=14, color="white"),
                    bgcolor="rgba(0,0,0,0.7)",
                    bordercolor="white",
                    borderwidth=1
                ),
                annotation_position="top left"
            )
            
            # Format x-axis
            hist_fig.update_xaxes(
                dtick=bin_size,
                tickmode="linear"
            )
            
            # Display the chart
            st.plotly_chart(hist_fig, use_container_width=True)
        else:
            st.info("No detailed LOS data available for histogram.")
    
    with tab_demo:
        # Demographic breakdown of LOS
        st.subheader("Length of Stay by Demographic Group")
        
        # Create unique key for widgets
        key_suffix = hash_data(filter_timestamp)
        
        # Filter dimensions to only those in the dataset
        available_dimensions = [
            (lbl, col) for lbl, col in DEMOGRAPHIC_DIMENSIONS 
            if col in df_filt.columns
        ]
        
        if not available_dimensions:
            st.warning("No demographic dimensions available in the filtered dataset.")
            return
        
        # Dimension selection
        demo_label = st.selectbox(
            "Break down by…", 
            [lbl for lbl, _ in available_dimensions],
            key=f"los_dim_{key_suffix}",
            help="Analyze how LOS varies across demographic groups"
        )
        
        # Get the actual column name
        demo_col = next((col for lbl, col in available_dimensions if lbl == demo_label), None)

        if not demo_col:
            st.error(f"Column mapping not found for {demo_label}.")
            return

        # Add filter for values within selected demographic
        try:
            demo_values = sorted(df_filt[demo_col].dropna().unique())
            selected_values = st.multiselect(
                f"Select {demo_label} groups to include",
                options=demo_values,
                default=demo_values,
                key=f"los_val_filter_{key_suffix}",
                help=f"Filter which {demo_label} groups to show in this breakdown"
            )
        except Exception as e:
            st.error(f"Error loading demographic values: {e}")
            return

        if not selected_values:
            st.warning(f"No {demo_label} values selected. Please select at least one.")
            return
        
        # Minimum group size
        min_group = st.slider(
            "Minimum group size",
            min_value=5,
            max_value=50,
            value=10,
            step=5,
            key=f"los_min_group_{key_suffix}",
            help="Minimum number of clients required for a group to be included"
        )

        # Track dimension and selected values for recompute trigger
        previous_selection = state.get("selected_dimension", "")
        previous_value_filter = state.get("selected_values", [])
        previous_min_group = state.get("min_group_size", 10)

        dim_changed = previous_selection != demo_label
        values_changed = sorted(previous_value_filter) != sorted(selected_values)
        min_group_changed = previous_min_group != min_group

        if dim_changed or values_changed or min_group_changed:
            state["selected_dimension"] = demo_label
            state["selected_values"] = selected_values
            state["min_group_size"] = min_group
            if "los_by_demo" in state:
                state.pop("los_by_demo")
        
        # Calculate LOS by demographic if needed
        if "los_by_demo" not in state:
            with st.spinner(f"Calculating LOS by {demo_label}..."):
                try:
                    # Filter to selected demographic values
                    df_subset = df_filt[df_filt[demo_col].isin(selected_values)]
                    
                    if df_subset.empty:
                        st.info(f"No data available for the selected {demo_label} groups.")
                        return
                    
                    # Calculate LOS by demographic
                    los_by_demo = los_by_demographic(df_subset, demo_col, t0, t1, min_group)
                    
                    if los_by_demo.empty:
                        st.info(f"No groups meet the minimum size threshold of {min_group} clients.")
                        return
                    
                    state["los_by_demo"] = los_by_demo
                    
                except Exception as e:
                    st.error(f"Error calculating LOS by {demo_label}: {e}")
                    return
        else:
            los_by_demo = state["los_by_demo"]
        
        # Sort data for visualization
        sorted_los_by_demo = los_by_demo.sort_values("mean", ascending=False)
        
        # Limit number of groups to display
        MAX_GROUPS = 12
        if len(sorted_los_by_demo) > MAX_GROUPS:
            st.info(f"Displaying top {MAX_GROUPS} groups by average LOS. {len(sorted_los_by_demo) - MAX_GROUPS} smaller groups are not shown.")
            sorted_los_by_demo = sorted_los_by_demo.head(MAX_GROUPS)
        
        # Create combined chart comparing median, mean, and quartiles
        fig_combined = go.Figure()
        
        # Add IQR range as a horizontal bar
        for idx, row in sorted_los_by_demo.iterrows():
            fig_combined.add_trace(
                go.Bar(
                    y=[row[demo_col]],
                    x=[row["q3"] - row["q1"]],  # Width of bar = IQR
                    base=row["q1"],  # Start position = Q1
                    orientation='h',
                    name='IQR Range',
                    marker=dict(color=f'rgba({MAIN_COLOR.split("(")[1].split(")")[0]},0.6)'),
                    showlegend=idx==0,  # Only show in legend once
                    hoverinfo='text',
                    hovertext=f"25th-75th percentile: {row['q1']:.0f}-{row['q3']:.0f} days"
                )
            )
        
        # Add median markers
        fig_combined.add_trace(
            go.Scatter(
                y=sorted_los_by_demo[demo_col],
                x=sorted_los_by_demo["median"],
                mode='markers',
                name='Median LOS',
                marker=dict(
                    symbol='line-ns',
                    color=MAIN_COLOR,
                    size=12,
                    line=dict(width=2, color=MAIN_COLOR)
                ),
                text=sorted_los_by_demo["median"].apply(lambda x: f"{x:.0f} days"),
                hovertemplate='<b>Median:</b> %{text}<extra></extra>'
            )
        )
        
        # Add mean markers
        fig_combined.add_trace(
            go.Scatter(
                y=sorted_los_by_demo[demo_col],
                x=sorted_los_by_demo["mean"],
                mode='markers',
                name='Mean LOS',
                marker=dict(
                    symbol='diamond',
                    color=SECONDARY_COLOR,
                    size=8,
                    line=dict(width=1.5, color=SECONDARY_COLOR)
                ),
                text=sorted_los_by_demo["mean"].apply(lambda x: f"{x:.1f} days"),
                hovertemplate='<b>Mean:</b> %{text}<extra></extra>'
            )
        )
        
        # Add annotation to explain the chart
        fig_combined.add_annotation(
            text="Bar shows 25th-75th percentile range, blue line is median, red diamond is mean",
            xref="paper", yref="paper",
            x=1, y=-0.12,
            showarrow=False,
            font=dict(size=12),
            align="right"
        )
        
        # Add reference line for overall system median
        los_data = state.get("los_data", {})
        if "median_los" in los_data:
            fig_combined.add_vline(
                x=los_data['median_los'],
                line=dict(dash="dash", color=MAIN_COLOR, width=1),
                annotation=dict(
                    text=f"Overall Median: {los_data['median_los']:.1f}",
                    font=dict(size=12, color=MAIN_COLOR),
                    bordercolor=MAIN_COLOR,
                    borderwidth=1,
                    bgcolor="rgba(0,0,0,0.7)"
                ),
                annotation_position="top right"
            )
        
        # Apply consistent styling
        height = min(650, max(400, len(sorted_los_by_demo) * 35))
        fig_combined = apply_chart_style(
            fig_combined,
            title=f"Length of Stay Distribution by {demo_label}",
            xaxis_title="Length of Stay (days)",
            yaxis_title=demo_label,
            height=height
        )
        
        # Display the chart
        st.plotly_chart(fig_combined, use_container_width=True)
        
        # Resource utilization chart
        st.subheader("System Resource Usage Analysis")
        
        # Calculate proportional metrics
        sorted_los_by_demo["days_per_enrollment"] = sorted_los_by_demo["total_days"] / sorted_los_by_demo["count"]
        if "avg_los" in los_data and los_data["avg_los"] > 0:
            sorted_los_by_demo["efficiency_ratio"] = sorted_los_by_demo["days_per_enrollment"] / los_data["avg_los"]
        else:
            sorted_los_by_demo["efficiency_ratio"] = 1.0
        
        # Calculate totals for percentages
        total_days = sorted_los_by_demo["total_days"].sum()
        total_enrollments = sorted_los_by_demo["count"].sum()
        
        # Add percentage columns
        sorted_los_by_demo["days_pct"] = (sorted_los_by_demo["total_days"] / total_days * 100).round(1)
        sorted_los_by_demo["enrollment_pct"] = (sorted_los_by_demo["count"] / total_enrollments * 100).round(1)
        
        # Create resource usage chart
        fig_resource = go.Figure()
        
        # Add bars for total days
        fig_resource.add_trace(
            go.Bar(
                y=sorted_los_by_demo[demo_col],
                x=sorted_los_by_demo["total_days"],
                orientation='h',
                name='Total Days',
                text=sorted_los_by_demo.apply(
                    lambda row: f"{int(row['total_days']):,} days ({row['days_pct']:.1f}%)", 
                    axis=1
                ),
                hovertemplate='<b>%{y}</b><br>%{text}<extra></extra>',
                marker_color=MAIN_COLOR,
                textposition='auto'
            )
        )
        
        # Apply consistent styling
        height = min(650, max(400, len(sorted_los_by_demo) * 35))
        fig_resource = apply_chart_style(
            fig_resource,
            title=f"System Resource Usage by {demo_label}",
            xaxis_title="Total System Days",
            yaxis_title=demo_label,
            height=height
        )
        
        # Display the chart
        st.plotly_chart(fig_resource, use_container_width=True)
        
        # Create proportional usage comparison for top groups
        compare_df = sorted_los_by_demo.head(6).copy()
        
        # Create grouped bar chart for comparison
        fig_compare = go.Figure()
        
        # Add bars for enrollment percentage
        fig_compare.add_trace(
            go.Bar(
                y=compare_df[demo_col],
                x=compare_df["enrollment_pct"],
                orientation='h',
                name='% of Enrollments',
                marker_color=NEUTRAL_COLOR,
                text=compare_df["enrollment_pct"].apply(lambda x: f"{x:.1f}%"),
                textposition='auto'
            )
        )
        
        # Add bars for days percentage
        fig_compare.add_trace(
            go.Bar(
                y=compare_df[demo_col],
                x=compare_df["days_pct"],
                orientation='h',
                name='% of System Days',
                marker_color=MAIN_COLOR,
                text=compare_df["days_pct"].apply(lambda x: f"{x:.1f}%"),
                textposition='auto'
            )
        )
        
        # Apply consistent styling
        fig_compare = apply_chart_style(
            fig_compare,
            title="Resource Usage Proportion (Top 6 Groups)",
            xaxis_title="Percentage (%)",
            yaxis_title=demo_label,
            height=400
        )
        
        # Update layout for grouped bars
        fig_compare.update_layout(barmode='group', bargap=0.15, bargroupgap=0.1)
        
        # Set x-axis range
        fig_compare.update_xaxes(
            range=[0, max(compare_df["days_pct"].max(), compare_df["enrollment_pct"].max()) * 1.1]
        )
        
        # Display the chart
        st.plotly_chart(fig_compare, use_container_width=True)
        
        # Insights section
        with st.expander("Analysis & Data Observations", expanded=True):
            # Find groups with longest/shortest stays
            if len(sorted_los_by_demo) > 1:
                longest_stay = sorted_los_by_demo.iloc[0]
                shortest_stay = sorted_los_by_demo.iloc[-1]
                
                # Calculate resource usage by top group
                top_group_pct = (longest_stay["total_days"] / total_days * 100).round(1) if total_days > 0 else 0
                top_group_count_pct = (longest_stay["count"] / total_enrollments * 100).round(1) if total_enrollments > 0 else 0
                
                # Create insight sections with improved styling
                st.markdown(f"""
                <div style="padding:15px; border-radius:5px; margin-bottom:15px; border:1px solid {MAIN_COLOR}">
                    <h3 style="color:{MAIN_COLOR}">Key Length of Stay Findings</h3>
                    <ul>
                        <li>The <b>{longest_stay[demo_col]}</b> group has the longest average stay at <b>{longest_stay['mean']:.1f} days</b>, which is <b>{longest_stay['disparity']:.2f}x</b> the overall average.</li>
                        <li>The <b>{shortest_stay[demo_col]}</b> group has the shortest average stay at <b>{shortest_stay['mean']:.1f} days</b>.</li>
                        <li>The gap between longest and shortest average stays is <b>{(longest_stay['mean'] - shortest_stay['mean']):.1f} days</b>.</li>
                    </ul>
                </div>

                <div style="padding:15px; border-radius:5px; border:1px solid {SECONDARY_COLOR}">
                    <h3 style="color:{SECONDARY_COLOR}">System Resource Utilization</h3>
                    <ul>
                        <li>The <b>{longest_stay[demo_col]}</b> group accounts for <b>{fmt_pct(top_group_pct)}</b> of total system days while representing only <b>{fmt_pct(top_group_count_pct)}</b> of enrollments.</li>
                        <li>This represents a <b>{(top_group_pct/top_group_count_pct).round(1)}x</b> disproportionate use of system resources.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
                # Check for high disparity groups
                high_disparity = sorted_los_by_demo[sorted_los_by_demo["disparity"] > 1.5]
                if not high_disparity.empty:
                    groups = ", ".join(f"<b>{row[demo_col]}</b>" for _, row in high_disparity.iterrows())
                    st.markdown(f"""
                    <div style="padding:15px; border-radius:5px; margin-top:15px;">
                        <h3 style="color:#2166ac">Notable Statistical Patterns</h3>
                        <ul>
                            <li>Groups with significantly longer than average stays: {groups}</li>
                            <li>These groups show average stays at least 50% longer than the system average.</li>
                            <li>These differences may warrant further investigation to understand the factors contributing to these patterns.</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Not enough demographic groups to provide comparative analysis.")
                
        # Show detailed stats table
        with st.expander("Detailed Statistics Table"):
            # Create a display version with better column names
            display_df = sorted_los_by_demo.copy()
            display_df.columns = [
                demo_label, "Average LOS", "Median LOS", "Enrollment Count",
                "25th Percentile", "75th Percentile", "Total System Days",
                "LOS Ratio", "IQR", "Days per Enrollment", "Efficiency Ratio",
                "% of Total Days", "% of Enrollments"
            ]
            
            # Format the dataframe for better readability
            st.dataframe(
                display_df.style.format(
                    {
                        "Average LOS": "{:.1f}",
                        "Median LOS": "{:.1f}",
                        "Enrollment Count": "{:,}",
                        "25th Percentile": "{:.1f}",
                        "75th Percentile": "{:.1f}",
                        "Total System Days": "{:,.0f}",
                        "LOS Ratio": "{:.2f}",
                        "IQR": "{:.1f}",
                        "Days per Enrollment": "{:.1f}",
                        "Efficiency Ratio": "{:.2f}",
                        "% of Total Days": "{:.1f}%",
                        "% of Enrollments": "{:.1f}%"
                    }
                ).background_gradient(
                    subset=["Average LOS", "Total System Days", "LOS Ratio", "Efficiency Ratio"],
                    cmap="Blues"
                )
            )
            
            # Add explanatory notes
            st.markdown("""
            **Metrics Explanation:**
            - **Average LOS**: Mean length of stay in days  
            - **Median LOS**: Middle value (50th percentile) of lengths of stay  
            - **LOS Ratio**: Group's average LOS compared to the overall average (1.0 = same as average)  
            - **Efficiency Ratio**: Average days per enrollment compared to system average  
            - **IQR**: Interquartile range (difference between 75th and 25th percentiles)  
            """)
