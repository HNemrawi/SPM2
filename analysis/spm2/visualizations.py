"""
SPM2 Visualization Functions
----------------------------
Specialized visualizations for SPM2 analysis with natural styling.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional, Tuple, Union, Any

from ui.styling import (
    style_metric_cards, 
    NeutralColors, 
    apply_chart_theme,
    create_info_box,
    create_styled_divider
)


# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================

@st.cache_data(show_spinner=False)
def create_flow_pivot(final_df: pd.DataFrame, exit_col: str, return_col: str) -> pd.DataFrame:
    """
    Create a crosstab pivot table for flow analysis (Exit → Return).
    
    Parameters:
        final_df (DataFrame): Final merged SPM2 DataFrame.
        exit_col (str): Column name for exit category.
        return_col (str): Column name for return category.
    
    Returns:
        pd.DataFrame: Crosstab pivot without margins.
    """
    df_copy = final_df.copy()
    df_copy[return_col] = df_copy[return_col].fillna("No Return").astype(str)
    pivot = pd.crosstab(df_copy[exit_col], df_copy[return_col], margins=False)
    return pivot


def get_top_flows_from_pivot(pivot_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Generate a table of the top flows from a crosstab pivot table,
    excluding flows involving "No Return".
    
    Parameters:
        pivot_df (DataFrame): Crosstab pivot table.
        top_n (int): Number of top flows to return.
    
    Returns:
        pd.DataFrame: Top flows with source, target, count, and percent.
    """
    df = pivot_df.copy()
    total = df.values.sum()
    rows = []
    row_labels = df.index.tolist()
    col_labels = df.columns.tolist()
    
    for rlab in row_labels:
        if rlab == "No Return":
            continue
        for clab in col_labels:
            if clab == "No Return":
                continue
            val = df.loc[rlab, clab]
            rows.append({
                "Exit Dimension": rlab,
                "Entry Dimension": clab,
                "Count": val,
                "Percent": (val / total * 100) if total else 0
            })
    
    flows_df = pd.DataFrame(rows)
    flows_df = flows_df[flows_df["Count"] > 0].sort_values("Count", ascending=False)
    return flows_df.head(top_n)


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_flow_sankey(pivot_df: pd.DataFrame, title: str = "Exit → Return Sankey Diagram") -> go.Figure:
    """
    Build a Sankey diagram to visualize the flow from exit to return categories.

    Parameters:
        pivot_df (DataFrame): Flow pivot table (rows are exit, columns are return).
        title (str): Diagram title.

    Returns:
        go.Figure: Plotly figure of the Sankey diagram.
    """
    df = pivot_df.copy()
    
    # Get exit categories (rows) and return categories (columns)
    exit_cats = df.index.tolist()   # left side nodes (Exit)
    return_cats = df.columns.tolist() # right side nodes (Return)

    # Overall nodes list: first exit, then return
    nodes = exit_cats + return_cats

    # Create custom node type data
    n_exit = len(exit_cats)
    node_types = ["Exits"] * n_exit + ["Returns"] * len(return_cats)
    
    # Define node colors based on type
    node_colors = (
        [NeutralColors.CHART_COLORS[0]] * n_exit +  # Blue for exits
        [NeutralColors.CHART_COLORS[1]] * len(return_cats)  # Green for returns
    )
    
    sources, targets, values = [], [], []
    for i, ecat in enumerate(exit_cats):
        for j, rcat in enumerate(return_cats):
            val = df.loc[ecat, rcat]
            if val > 0:
                sources.append(i)
                targets.append(n_exit + j)
                values.append(val)

    sankey = go.Sankey(
        node=dict(
            pad=25,  # More padding between nodes
            thickness=25,  # Thicker nodes for better label visibility
            line=dict(color="rgba(0, 0, 0, 0.2)", width=2),  # Black border for definition
            label=nodes,
            color=node_colors,
            customdata=node_types,
            hovertemplate='%{label}<br>%{customdata}: %{value}<extra></extra>',
            # Force label positioning
            x=[0.001] * n_exit + [0.999] * len(return_cats),  # Push nodes to edges
            y=None  # Let Plotly optimize vertical positioning
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color="rgba(128, 128, 128, 0.15)",  # Very light links
            hovertemplate='From: %{source.label}<br>To: %{target.label}<br>Count: %{value}<extra></extra>'
        ),
        textfont=dict(
            color="rgba(255, 255, 255, 0.95)",  # White text on colored nodes
            size=12,
            family="Arial, sans-serif",
            weight=600  # Bold text for better readability
        ),
        arrangement='snap',  # Snap to grid for better layout
        orientation='h'  # Horizontal orientation
    )
    
    fig = go.Figure(data=[sankey])
    
    # Calculate dynamic height with more generous spacing
    num_nodes = max(len(exit_cats), len(return_cats))
    min_height = 600
    height_per_node = 40  # More space per node
    calculated_height = max(min_height, num_nodes * height_per_node + 250)
    
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(
                size=18, 
                color="#404040",  # Dark gray that's visible in both themes
                family="Arial, sans-serif",
                weight=600
            ),
            x=0.5,
            xanchor="center",
            y=0.98,
            yanchor="top"
        ),
        font=dict(
            size=12,
            color="#404040",
            family="Arial, sans-serif"
        ),
        height=calculated_height,
        margin=dict(
            l=150,  # Large left margin for exit labels
            r=150,  # Large right margin for return labels
            t=100,  # Top margin for title
            b=100   # Bottom margin to prevent cutoff
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        hoverlabel=dict(
            bgcolor="rgba(50, 50, 50, 0.95)", 
            font=dict(
                color="white",
                size=13,
                family="Arial, sans-serif"
            ),
            bordercolor="rgba(255, 255, 255, 0.3)",
            namelength=-1  # Show full label names
        ),
        # Force the plot to use full width
        autosize=True,
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            visible=False
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            visible=False
        )
    )
    
    for i, label in enumerate(exit_cats):
        fig.add_annotation(
            x=-0.05,
            y=i/(len(exit_cats)-1) if len(exit_cats) > 1 else 0.5,
            text=label,
            showarrow=False,
            xref="paper",
            yref="paper",
            xanchor="right",
            yanchor="middle",
            font=dict(
                size=11,
                color="#404040",
                family="Arial, sans-serif"
            )
        )
    
    for i, label in enumerate(return_cats):
        fig.add_annotation(
            x=1.05,
            y=i/(len(return_cats)-1) if len(return_cats) > 1 else 0.5,
            text=label,
            showarrow=False,
            xref="paper",
            yref="paper",
            xanchor="left",
            yanchor="middle",
            font=dict(
                size=11,
                color="#404040",
                family="Arial, sans-serif"
            )
        )
    
    return fig


def plot_days_to_return_box(final_df: pd.DataFrame, return_period: int = 730) -> go.Figure:
    """
    Create a horizontal box plot for the distribution of Days-to-Return,
    ensuring that the displayed quartiles, whiskers, and mean match your summary metrics.
    
    Parameters:
        final_df (DataFrame): Merged SPM2 DataFrame.
        return_period (int): Maximum days to consider.
    
    Returns:
        go.Figure: Plotly box plot figure.
    """
    # Filter down to clients who actually returned within the chosen period
    returned_df = final_df[
        (final_df["ReturnedToHomelessness"]) &
        (final_df["DaysToReturn"].notna()) &
        (final_df["DaysToReturn"] <= return_period)
    ]

    # If no returns are found, display a simple message
    if returned_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No Returns Found Within Selected Period",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color=NeutralColors.NEUTRAL_500)
        )
        fig.update_layout(
            title="Days to Return Distribution",
            xaxis_visible=False,
            yaxis_visible=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=400
        )
        return fig

    # Calculate summary statistics
    x = returned_df["DaysToReturn"].dropna()
    p25 = x.quantile(0.25)
    median_val = x.median()
    p75 = x.quantile(0.75)
    avg_val = x.mean()
    min_val = x.min()
    max_val = x.max()
    
    # Calculate Interquartile Range (IQR)
    IQR = p75 - p25
    
    # Determine the whiskers using the 1.5 * IQR rule
    lower_whisker = max(min_val, p25 - 1.5 * IQR)
    upper_whisker = min(max_val, p75 + 1.5 * IQR)

    fig = go.Figure()

    # Create the box plot trace
    fig.add_trace(go.Box(
        q1=[p25],
        median=[median_val],
        q3=[p75],
        lowerfence=[lower_whisker],
        upperfence=[upper_whisker],
        mean=[avg_val],
        boxmean='sd',
        boxpoints='outliers',  # Show only outliers for cleaner visualization
        jitter=0.3,
        pointpos=-1.8,
        orientation='h',
        name="Days to Return",
        marker=dict(
            color=NeutralColors.PRIMARY,
            outliercolor=NeutralColors.WARNING,
            size=6
        ),
        line=dict(color=NeutralColors.PRIMARY),
        fillcolor='rgba(0, 102, 204, 0.2)'  # Light primary color fill
    ))

    # Add vertical lines to highlight the median and mean
    fig.update_layout(
        title="Days to Return Distribution",
        xaxis_title="Days to Return",
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            gridcolor='rgba(128, 128, 128, 0.2)',
            zerolinecolor='rgba(128, 128, 128, 0.2)'
        ),
        yaxis=dict(
            showticklabels=False,
            gridcolor='rgba(0,0,0,0)'
        ),
        height=400,
        margin=dict(l=20, r=20, t=40, b=60),
        shapes=[
            dict(
                type="line",
                xref="x",
                x0=median_val, x1=median_val,
                yref="paper",
                y0=0, y1=1,
                line=dict(color=NeutralColors.SUCCESS, width=2, dash="dot"),
                opacity=0.7
            ),
            dict(
                type="line",
                xref="x",
                x0=avg_val, x1=avg_val,
                yref="paper",
                y0=0, y1=1,
                line=dict(color=NeutralColors.WARNING, width=2, dash="dash"),
                opacity=0.7
            ),
        ],
        annotations=[
            dict(
                x=median_val,
                y=1.05,
                xref="x",
                yref="paper",
                text=f"Median: {median_val:.0f}",
                showarrow=False,
                font=dict(size=12, color=NeutralColors.SUCCESS),
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor=NeutralColors.SUCCESS,
                borderwidth=1
            ),
            dict(
                x=avg_val,
                y=1.05,
                xref="x",
                yref="paper",
                text=f"Mean: {avg_val:.0f}",
                showarrow=False,
                font=dict(size=12, color=NeutralColors.WARNING),
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor=NeutralColors.WARNING,
                borderwidth=1
            )
        ]
    )

    return fig


# ============================================================================
# METRIC DISPLAY FUNCTIONS
# ============================================================================

def display_spm_metrics(metrics: Dict[str, Any], return_period: int, show_total_exits: bool = True):
    """
    Display key SPM2 metrics in a card-style layout with natural theming.
    
    Parameters:
        metrics (dict): Dictionary of computed metrics.
        return_period (int): Maximum days to consider for returns
        show_total_exits (bool): Whether to display total exit count.
    """
    # Apply neutral metric card styling
    style_metric_cards(
        border_left_color=NeutralColors.PRIMARY,
        box_shadow=True
    )

    # If return_period is less than 730, override the >24 Months metric
    if return_period <= 730:
        metrics["Return > 24 Months"] = None
        metrics["% Return > 24M"] = None

    # First row of metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    if show_total_exits:
        col1.metric("Number of Relevant Exits", f"{metrics['Number of Relevant Exits']:,}")
    col1.metric("PH Exits", f"{metrics['PH Exits']:,} ({metrics['% PH Exits']:.1f}%)")
    col2.metric("Return <6M", f"{metrics['Return < 6 Months']:,} ({metrics['% Return < 6M']:.1f}%)")
    col3.metric("Return 6–12M", f"{metrics['Return 6–12 Months']:,} ({metrics['% Return 6–12M']:.1f}%)")
    col4.metric("Return 12–24M", f"{metrics['Return 12–24 Months']:,} ({metrics['% Return 12–24M']:.1f}%)")
    
    # Conditionally render >24 months
    if metrics["Return > 24 Months"] is None:
        col5.metric("Return >24M", "N/A")
    else:
        col5.metric("Return >24M", f"{metrics['Return > 24 Months']:,} ({metrics['% Return > 24M']:.1f}%)")
    
    # Add spacing
    st.html(create_styled_divider())
    
    # Second row of metrics
    colA, colB, colC = st.columns(3)
    colA.metric("Total Returns", f"{metrics['Total Return']:,} ({metrics['% Return']:.1f}%)")
    colB.metric("Median Days", f"{metrics['Median Days (<=period)']:.1f}")
    colC.metric("Avg Days", f"{metrics['Average Days (<=period)']:.1f}")
    
    # Add detailed info box with percentile information
    percentile_info = f"""
    <div style="margin-top: 20px;">
        <h4 style="margin-bottom: 10px; color: {NeutralColors.PRIMARY};">Distribution Details</h4>
        <p><strong>25th/75th Percentiles:</strong> {metrics['DaysToReturn 25th Pctl']:.0f} / {metrics['DaysToReturn 75th Pctl']:.0f} days</p>
        <p><strong>Maximum Days to Return:</strong> {metrics['DaysToReturn Max']:.0f} days</p>
        <p style="margin-top: 10px; color: var(--text-secondary);">
            25% of returns occurred within <strong>{metrics['DaysToReturn 25th Pctl']:.0f}</strong> days, 
            and 25% occurred beyond <strong>{metrics['DaysToReturn 75th Pctl']:.0f}</strong> days. 
            The longest time to return was <strong>{metrics['DaysToReturn Max']:.0f}</strong> days.
        </p>
    </div>
    """
    
    st.html(create_info_box(
        percentile_info,
        type="info",
        title="Statistical Summary",
        icon="📊"
    ))


# ============================================================================
# EXPORT ALL PUBLIC FUNCTIONS
# ============================================================================

__all__ = [
    'create_flow_pivot',
    'get_top_flows_from_pivot',
    'plot_flow_sankey',
    'plot_days_to_return_box',
    'display_spm_metrics'
]