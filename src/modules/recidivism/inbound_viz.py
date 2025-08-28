"""
Inbound Recidivism Visualizations
---------------------------------
Specialized visualizations for inbound recidivism analysis with natural styling.
"""

from typing import Any, Dict

import pandas as pd
import plotly.graph_objects as go
from pandas.api.types import is_categorical_dtype

from src.ui.factories.charts import default_chart
from src.ui.factories.components import ui
from src.ui.themes.styles import NeutralColors

# ============================================================================
# METRIC DISPLAY FUNCTIONS
# ============================================================================


def display_return_metrics_cards(metrics: Dict[str, Any]):
    """
    Display inbound recidivism metrics as cards using the new component factory.

    Parameters:
        metrics (Dict[str, Any]): Dictionary of metrics
    """
    # Use component factory for metric display
    ui.metric_row(
        {
            "Total Entries": f"{
                metrics['Total Entries']:,}",
            "New Clients": f"{
                metrics['New']:,} ({
                    metrics['New (%)']:.1f}%)",
            "Returning Clients": f"{
                        metrics['Returning']:,} ({
                            metrics['Returning (%)']:.1f}%)",
            "Returns From Housing": f"{
                                metrics['Returning From Housing']:,} ({
                                    metrics['Returning From Housing (%)']:.1f}%)",
        },
        columns=4,
    )


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================


def plot_time_to_entry_box(final_df: pd.DataFrame) -> go.Figure:
    """
    Create a box plot for the Time-to-Entry distribution (days between exit and new entry).

    Parameters:
        final_df (DataFrame): Inbound recidivism DataFrame.

    Returns:
        go.Figure: Plotly box plot figure.
    """
    # Filter to returning clients only
    returned_df = final_df[final_df["ReturnCategory"] != "New"].dropna(
        subset=["days_since_last_exit"]
    )

    if returned_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No Return Entries Found",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color=NeutralColors.NEUTRAL_500),
        )
        fig.update_layout(
            title="Time to Entry Distribution",
            xaxis_visible=False,
            yaxis_visible=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=400,
        )
        return fig

    # Calculate statistics
    x = returned_df["days_since_last_exit"].dropna()
    median_val = x.median()
    avg_val = x.mean()
    # p25 = x.quantile(0.25)
    # p75 = x.quantile(0.75)

    fig = go.Figure()

    # Add box plot with natural styling
    fig.add_trace(
        go.Box(
            x=x,
            name="Time to Entry",
            boxmean="sd",
            boxpoints="outliers",
            jitter=0.3,
            pointpos=-1.8,
            marker=dict(
                color=NeutralColors.PRIMARY,
                outliercolor=NeutralColors.WARNING,
                size=6,
            ),
            line=dict(color=NeutralColors.PRIMARY),
            fillcolor="rgba(0, 102, 204, 0.2)",  # Light primary color fill
        )
    )

    # Apply consistent layout using chart factory
    fig = default_chart.apply_layout(
        fig,
        title="Time to Entry Distribution (Days Between Exit and Return)",
        xaxis_title="Days to Entry",
        showlegend=False,
        height=400,
        margin=dict(l=20, r=20, t=60, b=60),
    )

    # Update yaxis to hide labels
    fig.update_yaxes(showticklabels=False, gridcolor="rgba(0,0,0,0)")

    # Add statistical reference lines
    fig.add_shape(
        type="line",
        xref="x",
        x0=median_val,
        x1=median_val,
        yref="paper",
        y0=0,
        y1=1,
        line=dict(color=NeutralColors.SUCCESS, width=2, dash="dot"),
        opacity=0.7,
    )
    fig.add_shape(
        type="line",
        xref="x",
        x0=avg_val,
        x1=avg_val,
        yref="paper",
        y0=0,
        y1=1,
        line=dict(color=NeutralColors.WARNING, width=2, dash="dash"),
        opacity=0.7,
    )

    # Add annotations using chart factory method
    fig = default_chart.add_annotation(
        fig,
        text=f"Median: {median_val:.0f} days",
        x=median_val,
        y=1.05,
        arrow=False,
        xref="x",
        yref="paper",
        font=dict(size=12, color=NeutralColors.SUCCESS),
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor=NeutralColors.SUCCESS,
        borderwidth=1,
    )
    fig = default_chart.add_annotation(
        fig,
        text=f"Mean: {avg_val:.0f} days",
        x=avg_val,
        y=1.05,
        arrow=False,
        xref="x",
        yref="paper",
        font=dict(size=12, color=NeutralColors.WARNING),
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor=NeutralColors.WARNING,
        borderwidth=1,
    )

    return fig


# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================


def create_flow_pivot_ra(
    final_df: pd.DataFrame, source_col: str, target_col: str
) -> pd.DataFrame:
    """
    Create a pivot table for inbound recidivism flow analysis (Exit → Entry).

    Parameters:
        final_df (pd.DataFrame): DataFrame with inbound recidivism data.
        source_col (str): Column representing the exit category.
        target_col (str): Column representing the entry category.

    Returns:
        pd.DataFrame: Crosstab pivot table of exit vs. entry counts,
                      including a 'No Data' column for missing entries.

    Raises:
        KeyError: If source_col or target_col is not in final_df.
        RuntimeError: For any other unexpected errors.
    """
    try:
        # Copy to avoid mutating original DataFrame
        df_copy = final_df.copy()

        # If target column is categorical, convert to object
        if is_categorical_dtype(df_copy[target_col]):
            df_copy[target_col] = df_copy[target_col].astype(object)

        # Fill missing entries with 'No Data' and ensure string dtype
        df_copy[target_col] = df_copy[target_col].fillna("No Data").astype(str)

        # Build the crosstab (pivot table)
        pivot = pd.crosstab(
            df_copy[source_col],
            df_copy[target_col],
            margins=False,
            dropna=False,
        )
        return pivot

    except KeyError as ke:
        raise KeyError(f"Column not found in DataFrame: {ke}") from ke
    except Exception as e:
        raise RuntimeError(
            f"Unexpected error in create_flow_pivot_ra: {e}"
        ) from e


def plot_flow_sankey_ra(
    pivot_df: pd.DataFrame, title: str = "Exit → Entry Sankey"
) -> go.Figure:
    """
    Build a Sankey diagram for inbound recidivism (Exit → Entry) with natural styling.

    Parameters:
        pivot_df (DataFrame): Flow pivot table (rows: Exit categories, columns: Entry categories).
        title (str): Diagram title.

    Returns:
        go.Figure: Plotly Sankey figure.
    """
    df = pivot_df.copy()
    exit_cats = df.index.tolist()  # Left side nodes (Exits)
    entry_cats = df.columns.tolist()  # Right side nodes (Entries)

    nodes = exit_cats + entry_cats
    n_exit = len(exit_cats)

    # Node types for hover clarity
    node_types = ["Prior Exit"] * n_exit + ["Current Entry"] * len(entry_cats)

    # Define node colors based on type
    node_colors = [NeutralColors.CHART_COLORS[0]] * n_exit + [
        NeutralColors.CHART_COLORS[1]
    ] * len(entry_cats)

    sources, targets, values = [], [], []
    for i, exit_cat in enumerate(exit_cats):
        for j, entry_cat in enumerate(entry_cats):
            val = df.loc[exit_cat, entry_cat]
            if val > 0:
                sources.append(i)
                targets.append(n_exit + j)
                values.append(val)

    sankey = go.Sankey(
        node=dict(
            pad=25,  # More padding between nodes
            thickness=25,  # Thicker nodes for better label visibility
            line=dict(
                color="rgba(0, 0, 0, 0.2)", width=2
            ),  # Black border for definition
            label=nodes,
            color=node_colors,
            customdata=node_types,
            hovertemplate="%{label}<br>%{customdata}: %{value}<extra></extra>",
            # Force label positioning
            x=[0.001] * n_exit
            + [0.999] * len(entry_cats),  # Push nodes to edges
            y=None,  # Let Plotly optimize vertical positioning
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color="rgba(128, 128, 128, 0.15)",  # Very light links
            hovertemplate="From: %{source.label}<br>To: %{target.label}<br>Count: %{value}<extra></extra>",
        ),
        textfont=dict(
            color="rgba(255, 255, 255, 0.95)",  # White text on colored nodes
            size=12,
            family="Arial, sans-serif",
            weight=600,  # Bold text for better readability
        ),
        arrangement="snap",  # Snap to grid for better layout
        orientation="h",  # Horizontal orientation
    )

    fig = go.Figure(data=[sankey])

    # Calculate dynamic height with more generous spacing
    num_nodes = max(len(exit_cats), len(entry_cats))
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
                weight=600,
            ),
            x=0.5,
            xanchor="center",
            y=0.98,
            yanchor="top",
        ),
        font=dict(size=12, color="#404040", family="Arial, sans-serif"),
        height=calculated_height,
        margin=dict(
            l=150,  # Large left margin for exit labels
            r=150,  # Large right margin for entry labels
            t=100,  # Top margin for title
            b=100,  # Bottom margin to prevent cutoff
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(
            bgcolor="rgba(50, 50, 50, 0.95)",  # Dark background
            font=dict(color="white", size=13, family="Arial, sans-serif"),
            bordercolor="rgba(255, 255, 255, 0.3)",
            namelength=-1,  # Show full label names
        ),
        # Force the plot to use full width
        autosize=True,
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
    )

    # Add text annotations for better label visibility
    # This adds labels outside the Sankey nodes for clarity
    for i, label in enumerate(exit_cats):
        fig.add_annotation(
            x=-0.05,
            y=i / (len(exit_cats) - 1) if len(exit_cats) > 1 else 0.5,
            text=f"Prior Exit: {label}",  # Clarify it's a prior exit
            showarrow=False,
            xref="paper",
            yref="paper",
            xanchor="right",
            yanchor="middle",
            font=dict(size=11, color="#404040", family="Arial, sans-serif"),
        )

    for i, label in enumerate(entry_cats):
        fig.add_annotation(
            x=1.05,
            y=i / (len(entry_cats) - 1) if len(entry_cats) > 1 else 0.5,
            text=f"Entry: {label}",  # Clarify it's a current entry
            showarrow=False,
            xref="paper",
            yref="paper",
            xanchor="left",
            yanchor="middle",
            font=dict(size=11, color="#404040", family="Arial, sans-serif"),
        )

    return fig


def get_top_flows_from_pivot(
    pivot_df: pd.DataFrame, top_n: int = 10
) -> pd.DataFrame:
    """
    Extract the top flows from a pivot table, excluding "No Data" entries.

    Parameters:
        pivot_df (pd.DataFrame): Crosstab pivot table
        top_n (int): Number of top flows to include

    Returns:
        pd.DataFrame: Top flows with counts and percentages
    """
    total = pivot_df.values.sum()

    # Convert pivot to long format for vectorized operations
    flows_long = pivot_df.stack().reset_index()
    flows_long.columns = ["Prior Exit", "Current Entry", "Count"]

    # Filter and calculate percentages vectorized
    flows = flows_long[
        (flows_long["Count"] > 0)
        & (flows_long["Prior Exit"] != "No Data")
        & (flows_long["Current Entry"] != "No Data")
    ].copy()

    flows["Count"] = flows["Count"].astype(int)
    flows["Percent"] = ((flows["Count"] / total * 100) if total else 0).round(
        1
    )

    result_df = pd.DataFrame(flows)
    if not result_df.empty:
        result_df = result_df.sort_values("Count", ascending=False).head(top_n)
        result_df["Percent"] = result_df["Percent"].apply(
            lambda x: f"{x:.1f}%"
        )

    return result_df


# ============================================================================
# ADDITIONAL DISPLAY HELPERS
# ============================================================================


def display_time_statistics(final_df: pd.DataFrame):
    """
    Display detailed time-to-entry statistics with natural styling.

    Parameters:
        final_df (DataFrame): Inbound recidivism DataFrame
    """
    returned_df = final_df[final_df["ReturnCategory"] != "New"].dropna(
        subset=["days_since_last_exit"]
    )

    if not returned_df.empty:
        days = returned_df["days_since_last_exit"]

        # Calculate statistics
        stats = {
            "count": len(days),
            "mean": days.mean(),
            "median": days.median(),
            "std": days.std(),
            "min": days.min(),
            "max": days.max(),
            "q25": days.quantile(0.25),
            "q75": days.quantile(0.75),
        }

        # Create info box with statistics
        stats_html = f"""
        <div style="margin-top: 20px;">
            <h4 style="margin-bottom: 10px; color: {NeutralColors.PRIMARY}
            ;">Time-to-Entry Statistics</h4>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;">
                <div>
                    <p><strong>Number of Returns:</strong> {stats['count']:,}</p>
                    <p><strong>Mean Days:</strong> {stats['mean']:.1f}</p>
                    <p><strong>Median Days:</strong> {stats['median']:.1f}</p>
                    <p><strong>Std Deviation:</strong> {stats['std']:.1f}</p>
                </div>
                <div>
                    <p><strong>Minimum Days:</strong> {stats['min']:.0f}</p>
                    <p><strong>Maximum Days:</strong> {stats['max']:.0f}</p>
                    <p><strong>25th Percentile:</strong> {stats['q25']:.0f}</p>
                    <p><strong>75th Percentile:</strong> {stats['q75']:.0f}</p>
                </div>
            </div>
            <p style="margin-top: 15px; color: var(--text-secondary);">
                50% of returns occurred between <strong>{stats['q25']:.0f}</strong> and
                <strong>{stats['q75']:.0f}</strong> days after exit.
            </p>
        </div>
        """

        ui.info_section(
            stats_html,
            type="info",
            title="Statistical Summary",
            icon="📊",
            expanded=True,
        )


# ============================================================================
# EXPORT ALL PUBLIC FUNCTIONS
# ============================================================================

__all__ = [
    "display_return_metrics_cards",
    "plot_time_to_entry_box",
    "create_flow_pivot_ra",
    "plot_flow_sankey_ra",
    "get_top_flows_from_pivot",
    "display_time_statistics",
]
