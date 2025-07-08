"""
Demographic breakdown section for HMIS dashboard - Improved Version
"""

import numpy as np
import math
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pandas import DataFrame
from typing import Dict, List, Set, Tuple, Optional, Any

from analysis.general.data_utils import (
    DEMOGRAPHIC_DIMENSIONS, _safe_div, category_counts, inflow, 
    outflow, ph_exit_clients, ph_exit_rate, return_after_exit, served_clients
)
from analysis.general.filter_utils import (
    get_filter_timestamp, hash_data, init_section_state, is_cache_valid, invalidate_cache
)
from analysis.general.theme import (
    CUSTOM_COLOR_SEQUENCE, MAIN_COLOR, PLOT_TEMPLATE, SECONDARY_COLOR, SUCCESS_COLOR,
    WARNING_COLOR, NEUTRAL_COLOR, apply_chart_style, create_insight_container, 
    fmt_int, fmt_pct, blue_divider
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
    
    # Get ALL unique clients who exited for PH exit rate calculation
    all_exits_mask = df_filt["ProjectExit"].between(t0, t1)
    all_exit_ids = set(df_filt.loc[all_exits_mask, "ClientID"])
    
    # Calculate returns using HUD-compliant logic with FULL dataset
    # First, get the PH exits subset for returns calculation
    ph_exits_mask = (
        (df_filt["ProjectExit"].between(t0, t1))
        & (df_filt["ExitDestinationCat"] == "Permanent Housing Situations")
    )
    ph_exits_df = df_filt[ph_exits_mask]
    ph_exits_ids = set(ph_exits_df["ClientID"].unique())
    
    # Calculate returns only for those who exited to PH
    return_ids = return_after_exit(ph_exits_df, full_df, t0, t1, return_window)
    
    # Validate returns are subset of PH exits
    if not return_ids.issubset(ph_exits_ids):
        print(f"WARNING: {len(return_ids - ph_exits_ids)} returns not in PH exits set")
        return_ids = return_ids.intersection(ph_exits_ids)
    
    # Calculate counts by demographic dimension - ensuring unique clients
    bdf = pd.concat(
        [
            category_counts(df_filt, served_ids, dim_col, "Served"),
            category_counts(df_filt, inflow_ids, dim_col, "Inflow"),
            category_counts(df_filt, outflow_ids, dim_col, "Outflow"),
            category_counts(df_filt, all_exit_ids, dim_col, "Total Exits"),
            category_counts(df_filt, ph_ids, dim_col, "PH Exits"),
        ],
        axis=1,
    ).fillna(0).reset_index().rename(columns={"index": dim_col})
    
    # Skip if no data
    if bdf.empty:
        return pd.DataFrame()
    
    # FIXED: Calculate PH Exit Rate using the same logic as ph_exit_rate()
    # For each demographic group, calculate rate properly
    ph_exit_rates = []
    for _, row in bdf.iterrows():
        demo_value = row[dim_col]
        
        # Use safe division with same logic as ph_exit_rate function
        total_exits = row["Total Exits"]
        ph_exits = row["PH Exits"]
        
        if total_exits > 0:
            rate = (ph_exits / total_exits) * 100
        else:
            rate = 0.0
        
        ph_exit_rates.append(round(rate, 1))
    
    bdf["PH Exit Rate"] = ph_exit_rates
    
    # FIXED: Prepare returns by demographic using validated return IDs
    # Get demographic info for clients
    clients_demo = df_filt[["ClientID", dim_col]].drop_duplicates()
    
    # Filter to PH exits and returns
    ph_demo = clients_demo[clients_demo["ClientID"].isin(ph_exits_ids)]
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
    returns_df["PH Exit Count"] = returns_df["PH Exit Count"].fillna(0).astype(int)
    returns_df["Returns Count"] = returns_df["Returns Count"].fillna(0).astype(int)
    
    # Calculate Returns Rate with proper validation
    returns_df["Returns to Homelessness Rate"] = returns_df.apply(
        lambda row: round((row["Returns Count"] / row["PH Exit Count"]) * 100, 1) 
        if row["PH Exit Count"] > 0 else np.nan,
        axis=1
    )
    
    # Merge with main breakdown dataframe
    bdf = pd.merge(bdf, returns_df, on=dim_col, how="left")
    
    # Fill any missing values appropriately
    bdf["PH Exit Count"] = bdf["PH Exit Count"].fillna(0).astype(int)
    bdf["Returns Count"] = bdf["Returns Count"].fillna(0).astype(int)
    bdf["Returns to Homelessness Rate"] = bdf["Returns to Homelessness Rate"].fillna(np.nan)
    
    # Add net flow column
    bdf["Net Flow"] = bdf["Inflow"] - bdf["Outflow"]
    
    # Ensure all numeric columns are properly typed
    numeric_cols = ["Served", "Inflow", "Outflow", "Total Exits", "PH Exits", 
                    "PH Exit Count", "Returns Count", "Net Flow"]
    for col in numeric_cols:
        if col in bdf.columns:
            bdf[col] = bdf[col].astype(int)
    
    return bdf

def _create_counts_chart(df: DataFrame, dim_col: str) -> go.Figure:
    """
    Create counts chart for demographic breakdown with improved layout.
    """
    # Determine optimal chart height based on number of groups
    num_groups = len(df[dim_col].unique())
    chart_height = max(500, min(900, 450 + (num_groups * 25)))
    
    # Reshape for plotting - exclude Total Exits from visual to avoid clutter
    counts_df = df.melt(
        id_vars=dim_col,
        value_vars=["Served", "Inflow", "Outflow", "PH Exits", "Returns Count"],
        var_name="Metric",
        value_name="Count",
    )
    
    # Calculate label characteristics
    labels = df[dim_col].astype(str).tolist()
    max_label_length = max(len(label) for label in labels)
    
    # For very long labels, use horizontal bar chart instead
    if max_label_length > 50 or num_groups > 15:
        # Create horizontal bar chart
        fig = px.bar(
            counts_df,
            y=dim_col,  # Note: x and y are swapped
            x="Count",
            color="Metric",
            orientation='h',
            barmode="group",
            template=PLOT_TEMPLATE,
            text_auto=".0f",
            title=f"Client Count by {dim_col}",
            color_discrete_sequence=CUSTOM_COLOR_SEQUENCE,
        )
        
        # Update text positioning
        fig.update_traces(
            textposition='outside',
            textfont=dict(size=9)
        )
        
        # Calculate left margin based on label length
        left_margin = max(150, min(400, 100 + (max_label_length * 5)))
        
        # Update layout
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(0,0,0,0.5)",
                bordercolor="white",
                borderwidth=1
            ),
            margin=dict(
                l=left_margin,
                r=80,
                t=120,
                b=80
            ),
            height=chart_height,
            bargap=0.2,
            bargroupgap=0.1
        )
        
        fig.update_xaxes(
            title="Count",
            automargin=True
        )
        
        fig.update_yaxes(
            title="",
            automargin=True,
            tickmode='linear',
            dtick=1
        )
        
    else:
        # Create vertical bar chart with aggressive label handling
        fig = px.bar(
            counts_df,
            x=dim_col,
            y="Count",
            color="Metric",
            barmode="group",
            template=PLOT_TEMPLATE,
            text_auto=".0f",
            title=f"Client Count by {dim_col}",
            color_discrete_sequence=CUSTOM_COLOR_SEQUENCE,
        )
        
        # Update text positioning
        fig.update_traces(
            textposition='outside',
            textfont=dict(size=10)
        )
        
        # Smart label wrapping for vertical charts
        if max_label_length > 25:
            wrapped_labels = []
            for label in labels:
                if len(label) > 25:
                    # Prioritize natural break points
                    if ',' in label:
                        parts = label.split(',', 1)
                        wrapped = parts[0].strip() + ',<br>' + parts[1].strip()
                    elif ' or ' in label:
                        wrapped = label.replace(' or ', '<br>or ')
                    elif '/' in label:
                        parts = label.split('/', 1)
                        wrapped = parts[0].strip() + '/<br>' + parts[1].strip()
                    elif ' - ' in label:
                        parts = label.split(' - ', 1)
                        wrapped = parts[0].strip() + ' -<br>' + parts[1].strip()
                    else:
                        # Find best space to break at
                        words = label.split()
                        if len(words) > 1:
                            mid_point = len(label) // 2
                            best_break = 0
                            min_diff = float('inf')
                            
                            current_pos = 0
                            for i, word in enumerate(words[:-1]):
                                current_pos += len(word) + 1
                                diff = abs(current_pos - mid_point)
                                if diff < min_diff:
                                    min_diff = diff
                                    best_break = i
                            
                            wrapped = ' '.join(words[:best_break+1]) + '<br>' + ' '.join(words[best_break+1:])
                        else:
                            wrapped = label
                else:
                    wrapped = label
                wrapped_labels.append(wrapped)
            
            # Create mapping for wrapped labels
            label_mapping = dict(zip(df[dim_col], wrapped_labels))
            counts_df = counts_df.copy()
            counts_df[dim_col] = counts_df[dim_col].map(label_mapping)
        
        # Calculate rotation and margins
        if num_groups <= 4 and max_label_length <= 20:
            rotation_angle = 0
            bottom_margin = 120
        elif num_groups <= 6 and max_label_length <= 30:
            rotation_angle = -30
            bottom_margin = 150
        elif num_groups <= 10:
            rotation_angle = -45
            bottom_margin = 200
        else:
            rotation_angle = -60
            bottom_margin = 250
        
        # Add extra margin for wrapped labels
        if max_label_length > 25:
            bottom_margin += 50
        
        # Update layout
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(0,0,0,0.5)",
                bordercolor="white",
                borderwidth=1
            ),
            margin=dict(
                l=80,
                r=80,
                t=120,
                b=bottom_margin
            ),
            height=chart_height,
            bargap=0.2,
            bargroupgap=0.1,
            xaxis=dict(
                tickangle=rotation_angle,
                automargin=True,
                title_standoff=30,
                tickmode='linear',
                dtick=1,
                tickfont=dict(size=11 if max_label_length > 40 else 12)
            ),
            yaxis=dict(
                title_standoff=20,
                automargin=True,
                rangemode="tozero"
            )
        )
    
    return fig

def _create_rates_charts(
    df: pd.DataFrame, dim_col: str, return_window: int
) -> Tuple[go.Figure, go.Figure]:
    """
    Create rate charts for PH exits and returns with improved layout,
    excluding groups with 0% PH Exit Rate, and ensuring the returns
    chart shows the same groups (even if their return rate is zero).

    Parameters:
    -----------
    df : pd.DataFrame
        Breakdown data containing these columns:
        - dim_col
        - "PH Exit Rate", "PH Exits", "Total Exits"
        - "Returns to Homelessness Rate", "Returns Count", "PH Exit Count"
    dim_col : str
        Demographic dimension column name.
    return_window : int
        Days to check for returns.

    Returns:
    --------
    Tuple[go.Figure, go.Figure]
        (PH exit rate figure, returns rate figure)

    Raises:
    -------
    ValueError
        If any required column is missing.
    """
    # 1) Validate required columns
    required = {
        dim_col,
        "PH Exit Rate", "PH Exits", "Total Exits",
        "Returns to Homelessness Rate", "Returns Count", "PH Exit Count",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for rate charts: {missing}")

    # 2) Filter out groups with 0% PH Exit Rate
    ph_df = df[df["PH Exit Rate"] > 0].copy()
    if ph_df.empty:
        # No groups to show
        empty_ph = go.Figure()
        empty_ph.update_layout(
            title="Permanent Housing Exit Rate (%)",
            template=PLOT_TEMPLATE,
            height=400,
            annotations=[{
                "text": "No PH exit data available",
                "xref": "paper", "yref": "paper",
                "x": 0.5, "y": 0.5, "showarrow": False
            }],
        )
        # Also produce a matching empty returns chart
        empty_ret = go.Figure()
        empty_ret.update_layout(
            title=f"Returns to Homelessness within {return_window} days (%)",
            template=PLOT_TEMPLATE,
            height=400,
            annotations=[{
                "text": "No returns data available",
                "xref": "paper", "yref": "paper",
                "x": 0.5, "y": 0.5, "showarrow": False
            }],
        )
        return empty_ph, empty_ret

    # 3) Prepare labels and layout parameters
    def process_labels_for_display(labels: List[str]) -> List[str]:
        """Abbreviate very long labels intelligently."""
        proc: List[str] = []
        max_len = max(len(str(l)) for l in labels)
        for lbl in labels:
            s = str(lbl)
            if max_len > 40 and len(s) > 40:
                if "American Indian, Alaska Native, or Indigenous" in s:
                    proc.append("AI/AN/Indigenous")
                elif "Black, African American, or African" in s:
                    proc.append("Black/African American")
                elif "Hispanic/Latina/e/o" in s:
                    proc.append("Hispanic/Latino")
                elif "," in s:
                    proc.append(s.split(",", 1)[0] + "...")
                else:
                    proc.append(s[:35] + "...")
            else:
                proc.append(s)
        return proc

    groups = sorted(ph_df[dim_col].dropna().unique())
    display_labels = process_labels_for_display(groups)
    display_map = dict(zip(groups, display_labels))
    n = len(groups)
    max_lbl = max((len(lbl) for lbl in display_labels), default=0)
    horizontal = n > 8 or max_lbl > 30

    if horizontal:
        height = max(400, min(800, 300 + n * 35))
        left_margin = min(400, max(150, max_lbl * 7))
        layout = dict(
            height=height,
            margin=dict(l=left_margin, r=80, t=100, b=80),
            showlegend=False
        )
    else:
        height = max(450, min(700, 400 + n * 30))
        if n <= 3 and max_lbl <= 15:
            angle, bottom = 0, 100
        elif n <= 5 and max_lbl <= 20:
            angle, bottom = -30, 150
        else:
            angle = -45
            vs = max_lbl * 7 * abs(math.sin(math.radians(45)))
            bottom = max(200, int(vs + 80))
        layout = {
            "height": height,
            "margin": dict(l=60, r=60, t=100, b=bottom),
            "showlegend": False,
            "xaxis": dict(
                tickangle=angle,
                automargin=False,
                tickmode="linear",
                dtick=1,
                tickfont=dict(size=10 if max_lbl > 25 else 11),
            ),
        }

    # 4) Build PH Exit Rate chart
    ph_plot = ph_df.sort_values(dim_col).copy()
    ph_plot["display_label"] = ph_plot[dim_col].map(display_map)

    if horizontal:
        fig_ph = px.bar(
            ph_plot,
            y="display_label",
            x="PH Exit Rate",
            orientation="h",
            template=PLOT_TEMPLATE,
            title="Permanent Housing Exit Rate (%)",
            color="PH Exit Rate",
            color_continuous_scale="Blues",
        )
        fig_ph.update_layout(
            **layout,
            xaxis=dict(range=[0, max(100, ph_plot["PH Exit Rate"].max() * 1.2)])
        )
        fig_ph.update_traces(texttemplate="%{x:.1f}%", textposition="outside")
    else:
        fig_ph = px.bar(
            ph_plot,
            x="display_label",
            y="PH Exit Rate",
            template=PLOT_TEMPLATE,
            title="Permanent Housing Exit Rate (%)",
            color="PH Exit Rate",
            color_continuous_scale="Blues",
        )
        fig_ph.update_layout(
            **layout,
            yaxis=dict(
                range=[0, max(100, ph_plot["PH Exit Rate"].max() * 1.3)],
                automargin=True
            )
        )
        fig_ph.update_traces(texttemplate="%{y:.1f}%", textposition="outside")

    fig_ph.update_traces(
        customdata=ph_plot[
            [dim_col, "PH Exit Rate", "PH Exits", "Total Exits"]
        ].values,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "PH Exit Rate: %{customdata[1]:.1f}%<br>"
            "PH Exits: %{customdata[2]}<br>"
            "Total Exits: %{customdata[3]}<extra></extra>"
        ),
    )

    # 5) Build Returns Rate chart with same groups, filling zeros
    ret_df = df.copy()
    ret_df["Returns to Homelessness Rate"] = ret_df[
        "Returns to Homelessness Rate"
    ].fillna(0)
    ret_plot = ret_df[ret_df[dim_col].isin(groups)].sort_values(dim_col).copy()
    ret_plot["display_label"] = ret_plot[dim_col].map(display_map)

    if horizontal:
        fig_ret = px.bar(
            ret_plot,
            y="display_label",
            x="Returns to Homelessness Rate",
            orientation="h",
            template=PLOT_TEMPLATE,
            title=f"Returns to Homelessness within {return_window} days (%)",
            color="Returns to Homelessness Rate",
            color_continuous_scale="Reds",
        )
        fig_ret.update_layout(
            **layout,
            xaxis=dict(range=[0, max(50, ret_plot["Returns to Homelessness Rate"].max() * 1.3)])
        )
        fig_ret.update_traces(texttemplate="%{x:.1f}%", textposition="outside")
    else:
        fig_ret = px.bar(
            ret_plot,
            x="display_label",
            y="Returns to Homelessness Rate",
            template=PLOT_TEMPLATE,
            title=f"Returns to Homelessness within {return_window} days (%)",
            color="Returns to Homelessness Rate",
            color_continuous_scale="Reds",
        )
        fig_ret.update_layout(
            **layout,
            yaxis=dict(
                range=[0, max(50, ret_plot["Returns to Homelessness Rate"].max() * 1.3)],
                automargin=True
            )
        )
        fig_ret.update_traces(texttemplate="%{y:.1f}%", textposition="outside")

    fig_ret.update_traces(
        customdata=ret_plot[
            [dim_col, "Returns to Homelessness Rate", "Returns Count", "PH Exit Count"]
        ].values,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Returns (" + str(return_window) + "d): %{customdata[1]:.1f}%<br>"
            "Returns Count: %{customdata[2]}<br>"
            "PH Exits: %{customdata[3]}<extra></extra>"
        ),
    )

    return fig_ph, fig_ret

def _create_outcome_quadrant_chart(df: DataFrame, dim_col: str) -> go.Figure:
    """
    Create outcome quadrant chart comparing PH exits and returns with improved layout.
    
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
            }],
            height=600
        )
        return fig
    
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
        y0=comparison_df["Returns to Homelessness Rate"].min() - 5,
        y1=comparison_df["Returns to Homelessness Rate"].max() + 5,
        line=dict(color="white", dash="dash", width=2),
    )
    fig.add_shape(
        type="line",
        x0=comparison_df["PH Exit Rate"].min() - 5,
        x1=comparison_df["PH Exit Rate"].max() + 5,
        y0=avg_return_rate, y1=avg_return_rate,
        line=dict(color="white", dash="dash", width=2),
    )
    
    # Calculate positions for quadrant annotations to avoid overlap
    x_range = comparison_df["PH Exit Rate"].max() - comparison_df["PH Exit Rate"].min()
    y_range = comparison_df["Returns to Homelessness Rate"].max() - comparison_df["Returns to Homelessness Rate"].min()
    
    # Add quadrant annotations with better positioning
    fig.add_annotation(
        x=comparison_df["PH Exit Rate"].max() - (x_range * 0.15),
        y=comparison_df["Returns to Homelessness Rate"].min() + (y_range * 0.1),
        text="🏆 Ideal:<br>High Exits, Low Returns",
        showarrow=False,
        font=dict(size=12, color="green"),
        bgcolor="rgba(0,0,0,0.7)",
        bordercolor="green",
        borderwidth=1,
        align="center"
    )
    fig.add_annotation(
        x=comparison_df["PH Exit Rate"].min() + (x_range * 0.15),
        y=comparison_df["Returns to Homelessness Rate"].max() - (y_range * 0.1),
        text="⚠️ Concern:<br>Low Exits, High Returns",
        showarrow=False,
        font=dict(size=12, color="red"),
        bgcolor="rgba(0,0,0,0.7)",
        bordercolor="red",
        borderwidth=1,
        align="center"
    )
    
    # Highlight best-performing group
    best_groups = comparison_df.loc[
        (comparison_df["PH Exit Rate"] >= avg_exit_rate) &
        (comparison_df["Returns to Homelessness Rate"] <= avg_return_rate)
    ].sort_values("PH Exit Rate", ascending=False)
    
    if not best_groups.empty:
        best_group = best_groups.iloc[0]
        best_label = best_group[dim_col]
        
        # Position annotation to avoid overlap
        fig.add_annotation(
            x=best_group["PH Exit Rate"],
            y=best_group["Returns to Homelessness Rate"],
            text=f"🌟 Best:<br>{best_label}",
            showarrow=True,
            arrowhead=2,
            ax=40,
            ay=-40,
            bgcolor="gold",
            font=dict(color="black", size=11),
            align="center"
        )
    
    # Update layout with better margins and dynamic height
    num_groups = len(comparison_df)
    chart_height = max(700, min(900, 650 + (num_groups * 5)))
    
    fig.update_layout(
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05,
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="white",
            borderwidth=1
        ),
        margin=dict(l=80, r=180, t=120, b=100),  # Generous margins
        height=chart_height,
        xaxis=dict(
            title_standoff=25,
            range=[
                max(0, comparison_df["PH Exit Rate"].min() - 10),
                min(100, comparison_df["PH Exit Rate"].max() + 10)
            ],
            automargin=True
        ),
        yaxis=dict(
            title_standoff=25,
            range=[
                max(0, comparison_df["Returns to Homelessness Rate"].min() - 5),
                min(100, comparison_df["Returns to Homelessness Rate"].max() + 5)
            ],
            automargin=True
        )
    )
    
    return fig

def _create_flow_balance_chart(df: DataFrame, dim_col: str) -> go.Figure:
    """
    Create a chart showing net flow (inflow - outflow) by demographic.
    """
    # Sort by net flow for visual impact
    flow_df = df.sort_values("Net Flow", ascending=True)
    
    # Color bars based on positive/negative
    colors = [SUCCESS_COLOR if x >= 0 else WARNING_COLOR for x in flow_df["Net Flow"]]
    
    # Create horizontal bar chart for better label visibility
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=flow_df["Net Flow"],
        y=flow_df[dim_col],
        orientation='h',
        marker_color=colors,
        text=flow_df["Net Flow"].apply(lambda x: f"{x:+,.0f}"),
        textposition='outside',
        textfont=dict(size=12, color="white"),
        hovertemplate='<b>%{y}</b><br>Net Flow: %{x:+,}<br>Inflow: %{customdata[0]:,}<br>Outflow: %{customdata[1]:,}<extra></extra>',
        customdata=np.column_stack((flow_df["Inflow"], flow_df["Outflow"]))
    ))
    
    # Calculate height
    num_groups = len(flow_df)
    chart_height = max(400, min(800, 350 + (num_groups * 30)))
    
    # Update layout
    fig.update_layout(
        title="System Flow Balance by Group",
        template=PLOT_TEMPLATE,
        height=chart_height,
        showlegend=False,
        xaxis=dict(
            title="Net Flow (Inflow - Outflow)",
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='white',
            automargin=True
        ),
        yaxis=dict(
            automargin=True,
            title=""
        ),
        margin=dict(
            l=max(150, 50 + (flow_df[dim_col].astype(str).str.len().max() * 8)),
            r=80,
            t=80,
            b=80
        )
    )
    
    # Add annotation for context
    total_net = flow_df["Net Flow"].sum()
    fig.add_annotation(
        text=f"Total System Net Flow: {total_net:+,}",
        xref="paper", yref="paper",
        x=0.5, y=1.05,
        showarrow=False,
        font=dict(size=14, color="white", weight='bold'),
        bgcolor="rgba(0,0,0,0.7)",
        bordercolor="white",
        borderwidth=1,
        borderpad=4
    )
    
    return fig

def _create_empty_figure(message: str) -> go.Figure:
    """Create an empty figure with a message."""
    fig = go.Figure()
    fig.update_layout(
        title=message,
        template=PLOT_TEMPLATE,
        height=400,
        annotations=[{
            "text": message,
            "xref": "paper",
            "yref": "paper", 
            "x": 0.5,
            "y": 0.5,
            "showarrow": False,
            "font": {"size": 16, "color": "gray"}
        }]
    )
    return fig

@st.fragment
def render_breakdown_section(df_filt: DataFrame, full_df: Optional[DataFrame] = None) -> None:
    """
    Render demographic breakdown with enhanced visualizations.
    
    This section allows users to:
    - Analyze key metrics broken down by demographic dimensions
    - Compare PH exit rates and return rates across groups
    - Identify disparities and opportunities for intervention
    
    Parameters:
    -----------
    df_filt : DataFrame
        Filtered DataFrame
    full_df : DataFrame, optional
        Full DataFrame for returns analysis
    """
    # Ensure we have the full dataset for returns
    if full_df is None:
        full_df = st.session_state.get("df")
        if full_df is None:
            st.error("Original dataset not found. Returns analysis requires full data.")
            return

    # Initialize or retrieve section state
    state: Dict[str, Any] = init_section_state(BREAKDOWN_SECTION_KEY)

    # Check if cache is valid
    filter_ts = get_filter_timestamp()
    cache_valid = is_cache_valid(state, filter_ts)
    
    if not cache_valid:
        invalidate_cache(state, filter_ts)
        state.pop("breakdown_df", None)
        state.pop("return_window", None)
        state.pop("min_group_size", None)

    # Header with info button
    col_header, col_info = st.columns([6, 1])
    with col_header:
        st.subheader("👥 Demographic Breakdown Analysis")
    with col_info:
        with st.popover("ℹ️ Help", use_container_width=True):
            st.markdown("""
            ### Understanding Demographic Breakdowns
            
            This section analyzes key metrics broken down by demographic dimensions to identify disparities and opportunities for targeted interventions.
            
            **Available Dimensions:**
            - Race/Ethnicity, Gender, Entry Age Tier
            - Program CoC, Local CoC, Agency Name, Program Name
            - Project Type, Household Type, Head of Household
            - Veteran Status, Chronic Homelessness, Currently Fleeing DV
            
            **Metrics Calculated for Each Group:**
            
            **Population & Flow:**
            - **Served**: Unique clients active during the period (enrolled at any point)
            - **Inflow**: Clients entering who weren't in any programs the day before
            - **Outflow**: Clients exiting who aren't in any programs at period end
            - **Net Flow**: Inflow - Outflow (positive = growth, negative = reduction)
            
            **Housing Outcomes:**
            - **Total Exits**: Unique clients with any exit during the period
            - **PH Exits**: Unique clients exiting to permanent housing destinations
            - **PH Exit Rate**: (PH Exits ÷ Total Exits) × 100
            
            **Housing Stability:**
            - **Returns Count**: PH exits who returned to homelessness within tracking window
            - **Returns to Homelessness Rate**: (Returns ÷ PH Exits) × 100
            - **Return tracking**: Always system-wide, even with filters active
            
            **Important Notes:**
            - **Minimum group size filter**: Hide small groups for statistical reliability
            - **When filters are active**: Inflow/outflow only track movement within filtered programs
            - **Returns are different**: Always tracked across ALL programs regardless of filters
            - **Each client counted once**: Even with multiple enrollments
            
            **How to Interpret:**
            - **High PH Exit Rate** (>40%): Strong housing placement outcomes
            - **Low Return Rate** (<10%): Good housing stability
            - **Positive Net Flow**: Group is growing in the system
            - **Large disparities**: May indicate need for targeted interventions
            
            **Visual Components:**
            - **Client Movement Chart**: Compare served, inflow, outflow, PH exits, and returns
            - **System Flow Balance**: Visualize net flow (growth/reduction) by group
            - **Performance Metrics**: PH exit rates and return rates by category
            - **Outcome Comparison**: Quadrant chart plotting PH exits vs returns
            
            **Important Note About Client Counting:**
            - A client enrolled in multiple programs/projects is counted in each one
            - Example: A client who starts in Emergency Shelter then transitions to Rapid Re-Housing appears in both project type counts
            - This accurately reflects the number of unique clients served by each program/project/agency
            - The sum across all groups may exceed the total unique clients, which is expected and correct
            - This approach answers "How many clients did each program serve?" not "How many clients used only one program?"
                        
            **Tips for Analysis:**
            - Compare PH exit rates across groups to find disparities
            - Look for groups with high return rates needing stability support
            - Identify groups with negative net flow (more leaving than entering)
            - Use the quadrant chart to find best performers (high PH exits, low returns)
            - Consider group size when interpreting - larger groups are more reliable
            """)

    # Get time boundaries
    t0 = st.session_state.get("t0")
    t1 = st.session_state.get("t1")
    
    if not all([t0, t1]):
        st.warning("⚠️ Please set date ranges in the filter panel before viewing breakdowns.")
        return

    # Check for active filters warning
    active_filters = st.session_state.get("filters", {})
    if any(active_filters.values()):
        filter_warning_html = f"""
        <div style="background-color: rgba(255,165,0,0.1); border: 2px solid {WARNING_COLOR}; 
                    border-radius: 10px; padding: 15px; margin-bottom: 20px;">
            <strong>🔍 Filtered View Active</strong><br>
            Breakdown shows data for filtered subset only. Inflow/outflow are within this subset.
            Returns are tracked system-wide.
        </div>
        """
        st.html(filter_warning_html)

    # Settings row
    col1, col2, col3 = st.columns([3, 2, 2])
    
    with col1:
        # Choose breakdown dimension
        key_suffix = hash_data(filter_ts)
        dim_label = st.selectbox(
            "📊 Select dimension to analyze",
            [lbl for lbl, _ in DEMOGRAPHIC_DIMENSIONS],
            key=f"breakdown_dim_{key_suffix}",
            help="Choose how to segment your data"
        )
        dim_col = dict(DEMOGRAPHIC_DIMENSIONS)[dim_label]
    
    with col2:
        # Return window setting
        return_window = st.number_input(
            "Return tracking days",
            min_value=30,
            max_value=730,
            value=state.get("return_window", 180),
            step=30,
            key=f"return_window_{key_suffix}",
            help="Days after PH exit to track returns"
        )
        state["return_window"] = return_window
    
    with col3:
        # Minimum group size filter
        min_group_size = st.number_input(
            "Min group size",
            min_value=1,
            max_value=100,
            value=state.get("min_group_size", 10),
            step=5,
            key=f"min_group_{key_suffix}",
            help="Hide groups smaller than this"
        )

    # Check if any key parameters changed that require recalculation
    params_changed = (
        state.get("selected_dimension") != dim_label or
        state.get("cached_return_window") != return_window or
        state.get("min_group_size") != min_group_size
    )

    # Update state
    state["selected_dimension"] = dim_label
    state["min_group_size"] = min_group_size

    # If parameters changed, clear cache
    if params_changed:
        state.pop("breakdown_df", None)

    # Check if column exists
    if dim_col not in df_filt.columns:
        st.error(f"❌ Column '{dim_col}' not found in the dataset.")
        return

    # Group selection
    try:
        unique_groups = sorted(df_filt[dim_col].dropna().unique())
        
        if not unique_groups:
            st.warning(f"⚠️ No data available for {dim_label}.")
            return
        

        selected_groups = st.multiselect(
            f"Filter {dim_label} groups (showing all by default)",
            options=unique_groups,
            default=unique_groups,
            key=f"group_filter_{key_suffix}",
            help="Select specific groups to analyze"
        )
            
    except Exception as e:
        st.error(f"❌ Error loading categories: {e}")
        return

    if not selected_groups:
        st.info(f"ℹ️ Please select at least one {dim_label} category.")
        return

    # Compute breakdown if needed
    if "breakdown_df" not in state:
        with st.spinner(f"📊 Calculating breakdown by {dim_label}..."):
            try:
                # Calculate breakdown data
                bdf = _calculate_breakdown_data(
                    df_filt, full_df, dim_col, t0, t1, return_window
                )
                
                if bdf.empty:
                    st.info(f"ℹ️ No data available for breakdown by {dim_label}.")
                    return
                
                # Cache the FULL result before any filtering
                state["breakdown_df"] = bdf
                state["cached_return_window"] = return_window
                
            except Exception as e:
                st.error(f"❌ Error calculating breakdown: {e}")
                return
    
    # Get the cached data
    bdf = state["breakdown_df"].copy()  # Make a copy to avoid modifying cached data
    
    # Apply minimum group size filter
    bdf_filtered = bdf[bdf["Served"] >= min_group_size]
    
    if bdf_filtered.empty:
        st.info(f"ℹ️ No groups meet the minimum size threshold of {min_group_size}.")
        st.caption(f"The largest group has {bdf['Served'].max()} clients.")
        return
    
    # Apply group filter
    bdf_filtered = bdf_filtered[bdf_filtered[dim_col].isin(selected_groups)]

    if bdf_filtered.empty:
        st.info(f"ℹ️ No data available for the selected criteria.")
        return
        
    # Sort by served count for consistent display
    bdf_filtered = bdf_filtered.sort_values("Served", ascending=False)

    blue_divider()

    # Create organized layout with tabs
    tab_overview, tab_flow, tab_outcomes, tab_data = st.tabs([
        "📊 Overview", "🔄 System Flow", "🎯 Outcomes Analysis", "📋 Data Table"
    ])

    with tab_overview:
        # Summary metrics at top
        st.markdown("### Key Metrics Summary")
        
        # Calculate summary stats
        total_served = bdf_filtered["Served"].sum()
        total_inflow = bdf_filtered["Inflow"].sum()
        total_outflow = bdf_filtered["Outflow"].sum()
        total_ph_exits = bdf_filtered["PH Exits"].sum()
        total_returns = bdf_filtered["Returns Count"].sum()
        
        # Display summary metrics
        summary_cols = st.columns(5)
        summary_cols[0].metric("Total Served", fmt_int(total_served))
        summary_cols[1].metric("Total Inflow", fmt_int(total_inflow))
        summary_cols[2].metric("Total Outflow", fmt_int(total_outflow))
        summary_cols[3].metric("Total PH Exits", fmt_int(total_ph_exits))
        summary_cols[4].metric("Total Returns", fmt_int(total_returns))
        
        # Volume comparison chart
        st.markdown("### Client Count by Group")
        fig_counts = _create_counts_chart(bdf_filtered, dim_col)
        st.plotly_chart(fig_counts, use_container_width=True)

    with tab_flow:
        st.markdown("### System Flow Analysis")
        
        # Net flow visualization
        fig_flow = _create_flow_balance_chart(bdf_filtered, dim_col)
        st.plotly_chart(fig_flow, use_container_width=True)
        
        # Flow insights
        with st.expander("📊 Flow Insights", expanded=True):
            # Groups with highest growth/reduction
            growth_groups = bdf_filtered[bdf_filtered["Net Flow"] > 0].sort_values("Net Flow", ascending=False)
            reduction_groups = bdf_filtered[bdf_filtered["Net Flow"] < 0].sort_values("Net Flow", ascending=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📈 Growing Groups")
                if not growth_groups.empty:
                    for _, row in growth_groups.head(3).iterrows():
                        net_pct = (row["Net Flow"] / row["Served"] * 100) if row["Served"] > 0 else 0
                        st.markdown(f"**{row[dim_col]}**: +{row['Net Flow']} ({net_pct:.1f}% of served)")
                else:
                    st.info("No groups showing growth")
            
            with col2:
                st.markdown("#### 📉 Reducing Groups")
                if not reduction_groups.empty:
                    for _, row in reduction_groups.head(3).iterrows():
                        net_pct = (row["Net Flow"] / row["Served"] * 100) if row["Served"] > 0 else 0
                        st.markdown(f"**{row[dim_col]}**: {row['Net Flow']} ({net_pct:.1f}% of served)")
                else:
                    st.info("No groups showing reduction")

    with tab_outcomes:
        st.markdown("### Performance Metrics by Category")
        st.caption("Analyze success rates and identify high-performing groups")
        
        # Create two-column layout for rate charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig_ph, _ = _create_rates_charts(bdf_filtered, dim_col, return_window)
            st.plotly_chart(fig_ph, use_container_width=True)
            st.caption("📈 Higher rates indicate better housing placement outcomes")

        with col2:
            _, fig_ret = _create_rates_charts(bdf_filtered, dim_col, return_window)
            st.plotly_chart(fig_ret, use_container_width=True)
            st.caption("📉 Lower rates indicate better housing stability")

        # Outcome Comparison (quadrant chart)
        st.markdown("### Outcome Comparison: Finding Success Patterns")
        st.caption("Groups in the bottom-right quadrant have the best outcomes (high PH exits, low returns)")
        
        fig_outcome = _create_outcome_quadrant_chart(bdf_filtered, dim_col)
        st.plotly_chart(fig_outcome, use_container_width=True)
        
        # Add interpretation help
        with st.expander("📖 Understanding the charts", expanded=False):
            st.markdown("""
            **Rate Charts:**
            - **PH Exit Rate**: Percentage of ALL exits that went to permanent housing
            - **Return Rate**: Percentage of PH exits who returned within the tracking window
            
            **Quadrant Chart:**
            - **X-axis (PH Exit Rate)**: Higher is better - more exits to permanent housing
            - **Y-axis (Return Rate)**: Lower is better - fewer returns to homelessness
            - **Bubble size**: Represents the number of clients served
            
            The chart is divided into four quadrants by the average rates:
            - **Bottom-Right (🏆)**: High PH exits, low returns - best outcomes
            - **Top-Left (⚠️)**: Low PH exits, high returns - needs intervention
            - **Top-Right**: High PH exits but also high returns - housing stability issues
            - **Bottom-Left**: Low PH exits but also low returns - stable but limited housing placements
            """)

    with tab_data:
        st.markdown("### Detailed Data Export")
        
        # Prepare export dataframe
        export_df = bdf_filtered.copy()
        
        # Reorder columns for clarity
        column_order = [
            dim_col, "Served", "Inflow", "Outflow", "Net Flow",
            "Total Exits", "PH Exits", "PH Exit Rate",
            "PH Exit Count", "Returns Count", "Returns to Homelessness Rate"
        ]
        
        # Only include columns that exist
        column_order = [col for col in column_order if col in export_df.columns]
        export_df = export_df[column_order]
        
        # Format for display
        format_dict = {
            "Served": "{:,.0f}",
            "Inflow": "{:,.0f}",
            "Outflow": "{:,.0f}",
            "Net Flow": "{:+,.0f}",
            "Total Exits": "{:,.0f}",
            "PH Exits": "{:,.0f}",
            "PH Exit Rate": "{:.1f}%",
            "PH Exit Count": "{:,.0f}",
            "Returns Count": "{:,.0f}",
            "Returns to Homelessness Rate": "{:.1f}%",
        }
        
        # Apply formatting
        styled_export = export_df.style.format(
            {k: v for k, v in format_dict.items() if k in export_df.columns}
        )
        
        st.dataframe(styled_export, use_container_width=True, height=500)
        
        # Download options
        col1, col2 = st.columns(2)
        
        with col1:
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"breakdown_{dim_col}_{t0.strftime('%Y%m%d')}_{t1.strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Create a simple text download of key insights
            insights_text = f"""Demographic Breakdown Analysis
Date Range: {t0.strftime('%Y-%m-%d')} to {t1.strftime('%Y-%m-%d')}
Dimension: {dim_col}
Return Window: {return_window} days
Minimum Group Size: {min_group_size}

Key Metrics:
- Total Served: {export_df['Served'].sum():,}
- Total PH Exits: {export_df['PH Exits'].sum():,}
- Average PH Exit Rate: {export_df['PH Exit Rate'].mean():.1f}%
- Average Return Rate: {export_df['Returns to Homelessness Rate'].mean():.1f}%

Groups Shown: {len(export_df)} of {len(bdf)} total
Filtered Out: {len(bdf) - len(export_df)} groups with < {min_group_size} clients
"""
            
            st.download_button(
                label="📝 Download Summary",
                data=insights_text,
                file_name=f"breakdown_summary_{dim_col}_{t0.strftime('%Y%m%d')}.txt",
                mime="text/plain",
                use_container_width=True
            )