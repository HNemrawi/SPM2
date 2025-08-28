"""
Chart Factory Module
===================
Centralized Plotly chart configuration and generation.
Provides consistent styling and eliminates duplication across all visualizations.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.ui.themes.theme import professional_colors, theme


@dataclass
class ChartConfig:
    """Configuration for chart generation."""

    height: int = 400
    show_legend: bool = True
    legend_orientation: str = "h"  # "h" or "v"
    legend_position: str = "bottom"  # "bottom", "top", "right", "left"
    margin: Dict[str, int] = field(
        default_factory=lambda: {"l": 60, "r": 30, "t": 60, "b": 60}
    )
    hover_mode: str = "closest"
    template: Optional[str] = None
    width: str = 'stretch'
    animate: bool = False

    # Grid and axes
    show_grid: bool = True
    grid_color: str = "rgba(0, 0, 0, 0.1)"
    zero_line_color: str = "rgba(0, 0, 0, 0.2)"

    # Font settings
    font_family: Optional[str] = None
    font_size: int = 12
    title_font_size: int = 16

    # Colors
    color_sequence: Optional[List[str]] = None
    paper_bgcolor: str = "rgba(0, 0, 0, 0)"
    plot_bgcolor: str = "rgba(0, 0, 0, 0.02)"


class ChartFactory:
    """Factory class for creating consistent Plotly charts."""

    def __init__(self, config: Optional[ChartConfig] = None):
        self.config = config or ChartConfig()
        self.theme = theme
        self._setup_defaults()

    def _setup_defaults(self):
        """Setup default values from theme if not specified."""
        if not self.config.font_family:
            self.config.font_family = (
                self.theme.typography.font_family.replace('"', "")
            )
        if not self.config.color_sequence:
            # Use professional color scheme
            self.config.color_sequence = (
                professional_colors.CHART_COLORS_PRIMARY
            )

    # ============== BASE CONFIGURATION ==============

    def get_base_layout(
        self,
        title: Optional[str] = None,
        xaxis_title: Optional[str] = None,
        yaxis_title: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Get base layout configuration for any chart."""
        layout = {
            "template": self.config.template,
            "paper_bgcolor": self.config.paper_bgcolor,
            "plot_bgcolor": self.config.plot_bgcolor,
            "font": {
                "family": self.config.font_family,
                "size": self.config.font_size,
                "color": self.theme.colors.text_primary,
            },
            "margin": self.config.margin,
            "showlegend": self.config.show_legend,
            "colorway": self.config.color_sequence,
            "hovermode": self.config.hover_mode,
            "autosize": True,
            "height": self.config.height,
        }

        # Add title if provided
        if title:
            layout["title"] = {
                "text": title,
                "font": {
                    "size": self.config.title_font_size,
                    "color": self.theme.colors.text_primary,
                    "weight": 600,
                },
                "x": 0.5,
                "xanchor": "center",
            }

        # Configure legend
        if self.config.show_legend:
            layout["legend"] = self._get_legend_config()

        # Configure axes
        layout["xaxis"] = self._get_axis_config(xaxis_title)
        layout["yaxis"] = self._get_axis_config(yaxis_title)

        # Configure hover labels
        layout["hoverlabel"] = {
            "bgcolor": "rgba(255, 255, 255, 0.95)",
            "bordercolor": self.theme.colors.border,
            "font": {"color": self.theme.colors.text_primary},
        }

        # Merge with any additional kwargs
        layout.update(kwargs)

        return layout

    def _get_legend_config(self) -> Dict[str, Any]:
        """Get legend configuration based on position and orientation."""
        config = {
            "bgcolor": "rgba(255, 255, 255, 0.9)",
            "bordercolor": self.theme.colors.border,
            "borderwidth": 1,
            "font": {"color": self.theme.colors.text_secondary},
            "orientation": self.config.legend_orientation,
        }

        # Position based on orientation and position settings
        if self.config.legend_orientation == "h":
            config.update(
                {
                    "yanchor": "bottom",
                    "y": -0.15,
                    "xanchor": "center",
                    "x": 0.5,
                }
            )
        else:
            config.update(
                {"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 1.02}
            )

        return config

    def _get_axis_config(self, title: Optional[str] = None) -> Dict[str, Any]:
        """Get axis configuration."""
        config = {
            "showgrid": self.config.show_grid,
            "gridcolor": self.config.grid_color,
            "zeroline": True,
            "zerolinecolor": self.config.zero_line_color,
            "linecolor": self.theme.colors.border,
            "tickfont": {"color": self.theme.colors.text_secondary},
        }

        if title:
            config["title"] = {
                "text": title,
                "font": {"color": self.theme.colors.text_secondary},
            }

        return config

    def apply_layout(self, fig: go.Figure, **layout_kwargs) -> go.Figure:
        """Apply consistent layout to an existing figure."""
        layout = self.get_base_layout(**layout_kwargs)
        fig.update_layout(**layout)
        return fig

    # ============== CHART CREATION METHODS ==============

    def bar_chart(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        title: Optional[str] = None,
        color: Optional[str] = None,
        orientation: str = "v",
        **kwargs,
    ) -> go.Figure:
        """Create a bar chart with consistent styling."""
        fig = px.bar(
            data,
            x=x,
            y=y,
            color=color,
            orientation=orientation,
            color_discrete_sequence=self.config.color_sequence,
            **kwargs,
        )

        # Apply base layout
        self.apply_layout(
            fig,
            title=title,
            xaxis_title=x if orientation == "v" else y,
            yaxis_title=y if orientation == "v" else x,
        )

        return fig

    def line_chart(
        self,
        data: pd.DataFrame,
        x: str,
        y: Union[str, List[str]],
        title: Optional[str] = None,
        color: Optional[str] = None,
        markers: bool = True,
        **kwargs,
    ) -> go.Figure:
        """Create a line chart with consistent styling."""
        fig = px.line(
            data,
            x=x,
            y=y,
            color=color,
            markers=markers,
            color_discrete_sequence=self.config.color_sequence,
            **kwargs,
        )

        # Apply base layout
        self.apply_layout(
            fig,
            title=title,
            xaxis_title=x,
            yaxis_title=y if isinstance(y, str) else None,
        )

        return fig

    def scatter_chart(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        title: Optional[str] = None,
        color: Optional[str] = None,
        size: Optional[str] = None,
        **kwargs,
    ) -> go.Figure:
        """Create a scatter plot with consistent styling."""
        fig = px.scatter(
            data,
            x=x,
            y=y,
            color=color,
            size=size,
            color_discrete_sequence=self.config.color_sequence,
            **kwargs,
        )

        # Apply base layout
        self.apply_layout(fig, title=title, xaxis_title=x, yaxis_title=y)

        return fig

    def pie_chart(
        self,
        data: pd.DataFrame,
        values: str,
        names: str,
        title: Optional[str] = None,
        hole: float = 0,
        **kwargs,
    ) -> go.Figure:
        """Create a pie/donut chart with consistent styling."""
        fig = px.pie(
            data,
            values=values,
            names=names,
            hole=hole,
            color_discrete_sequence=self.config.color_sequence,
            **kwargs,
        )

        # Apply base layout
        self.apply_layout(fig, title=title)

        return fig

    def box_plot(
        self,
        data: Optional[pd.DataFrame] = None,
        x: Optional[Union[str, List]] = None,
        y: Optional[Union[str, List]] = None,
        title: Optional[str] = None,
        color: Optional[str] = None,
        **kwargs,
    ) -> go.Figure:
        """Create a box plot with consistent styling."""
        if data is not None:
            fig = px.box(
                data,
                x=x,
                y=y,
                color=color,
                color_discrete_sequence=self.config.color_sequence,
                **kwargs,
            )
        else:
            # Create from raw data
            fig = go.Figure()
            fig.add_trace(
                go.Box(
                    x=x,
                    y=y,
                    marker_color=self.config.color_sequence[0],
                    **kwargs,
                )
            )

        # Apply base layout
        self.apply_layout(fig, title=title)

        return fig

    def heatmap(
        self,
        data: pd.DataFrame,
        title: Optional[str] = None,
        colorscale: Optional[str] = None,
        show_values: bool = True,
        **kwargs,
    ) -> go.Figure:
        """Create a heatmap with consistent styling."""
        if colorscale is None:
            colorscale = [
                [0, self.theme.colors.primary_light],
                [1, self.theme.colors.primary],
            ]

        fig = go.Figure(
            data=go.Heatmap(
                z=data.values,
                x=data.columns,
                y=data.index,
                colorscale=colorscale,
                text=data.values if show_values else None,
                texttemplate="%{text}" if show_values else None,
                **kwargs,
            )
        )

        # Apply base layout
        self.apply_layout(fig, title=title)

        return fig

    def sankey_diagram(
        self,
        source: List[int],
        target: List[int],
        value: List[float],
        labels: List[str],
        title: Optional[str] = None,
        **kwargs,
    ) -> go.Figure:
        """Create a Sankey diagram with consistent styling."""
        # Generate colors for nodes
        node_colors = [
            self.config.color_sequence[i % len(self.config.color_sequence)]
            for i in range(len(labels))
        ]

        fig = go.Figure(
            data=[
                go.Sankey(
                    node=dict(
                        pad=15,
                        thickness=20,
                        line=dict(color=self.theme.colors.border, width=0.5),
                        label=labels,
                        color=node_colors,
                    ),
                    link=dict(
                        source=source,
                        target=target,
                        value=value,
                        color="rgba(0,0,0,0.1)",
                    ),
                    **kwargs,
                )
            ]
        )

        # Apply base layout
        self.apply_layout(fig, title=title)

        return fig

    def histogram(
        self,
        data: pd.DataFrame,
        x: str,
        title: Optional[str] = None,
        nbins: Optional[int] = None,
        color: Optional[str] = None,
        **kwargs,
    ) -> go.Figure:
        """Create a histogram with consistent styling."""
        fig = px.histogram(
            data,
            x=x,
            color=color,
            nbins=nbins,
            color_discrete_sequence=self.config.color_sequence,
            **kwargs,
        )

        # Apply base layout
        self.apply_layout(fig, title=title, xaxis_title=x, yaxis_title="Count")

        return fig

    # ============== SPECIALIZED CHARTS ==============

    def metric_gauge(
        self,
        value: float,
        title: str,
        min_value: float = 0,
        max_value: float = 100,
        target: Optional[float] = None,
        thresholds: Optional[Dict[str, float]] = None,
    ) -> go.Figure:
        """Create a gauge chart for metrics."""
        # Default thresholds
        if thresholds is None:
            thresholds = {"good": max_value * 0.7, "warning": max_value * 0.4}

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number+delta" if target else "gauge+number",
                value=value,
                title={"text": title},
                delta={"reference": target} if target else None,
                domain={"x": [0, 1], "y": [0, 1]},
                gauge={
                    "axis": {"range": [min_value, max_value]},
                    "bar": {"color": self.theme.colors.primary},
                    "steps": [
                        {
                            "range": [
                                min_value,
                                thresholds.get("warning", max_value * 0.4),
                            ],
                            "color": self.theme.colors.danger_light,
                        },
                        {
                            "range": [
                                thresholds.get("warning", max_value * 0.4),
                                thresholds.get("good", max_value * 0.7),
                            ],
                            "color": self.theme.colors.warning_light,
                        },
                        {
                            "range": [
                                thresholds.get("good", max_value * 0.7),
                                max_value,
                            ],
                            "color": self.theme.colors.success_light,
                        },
                    ],
                    "threshold": {
                        "line": {
                            "color": self.theme.colors.danger,
                            "width": 4,
                        },
                        "thickness": 0.75,
                        "value": target if target else value,
                    },
                },
            )
        )

        # Apply base layout with reduced height for gauge
        self.apply_layout(fig, title=None, height=250)

        return fig

    def waterfall_chart(
        self,
        x: List[str],
        y: List[float],
        title: Optional[str] = None,
        measure: Optional[List[str]] = None,
    ) -> go.Figure:
        """Create a waterfall chart."""
        fig = go.Figure(
            go.Waterfall(
                name="",
                orientation="v",
                measure=measure or ["relative"] * len(x),
                x=x,
                y=y,
                textposition="outside",
                text=[f"{v:+.0f}" if v != 0 else "" for v in y],
                connector={"line": {"color": self.theme.colors.border}},
                increasing={"marker": {"color": self.theme.colors.success}},
                decreasing={"marker": {"color": self.theme.colors.danger}},
                totals={"marker": {"color": self.theme.colors.primary}},
            )
        )

        # Apply base layout
        self.apply_layout(fig, title=title)

        return fig

    # ============== UTILITY METHODS ==============

    def add_threshold_line(
        self,
        fig: go.Figure,
        threshold: float,
        label: str,
        color: Optional[str] = None,
        axis: str = "y",
    ) -> go.Figure:
        """Add a threshold line to a chart."""
        color = color or self.theme.colors.danger

        if axis == "y":
            fig.add_hline(
                y=threshold,
                line_dash="dash",
                line_color=color,
                annotation_text=label,
                annotation_position="right",
            )
        else:
            fig.add_vline(
                x=threshold,
                line_dash="dash",
                line_color=color,
                annotation_text=label,
                annotation_position="top",
            )

        return fig

    def add_annotation(
        self,
        fig: go.Figure,
        text: str,
        x: float,
        y: float,
        arrow: bool = True,
        **kwargs,
    ) -> go.Figure:
        """Add an annotation to a chart."""
        # Extract font from kwargs if present, otherwise use default
        font = kwargs.pop("font", {"color": self.theme.colors.text_secondary})

        fig.add_annotation(
            text=text,
            x=x,
            y=y,
            showarrow=arrow,
            arrowhead=2 if arrow else 0,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor=self.theme.colors.text_secondary,
            font=font,
            **kwargs,
        )

        return fig

    def update_colors(
        self, fig: go.Figure, color_sequence: Optional[List[str]] = None
    ) -> go.Figure:
        """Update chart colors."""
        colors = color_sequence or self.config.color_sequence

        for i, trace in enumerate(fig.data):
            trace.marker.color = colors[i % len(colors)]

        return fig


# Create global instances with different presets
default_chart = ChartFactory()
chart_factory = default_chart  # Alias for backward compatibility

compact_chart = ChartFactory(
    ChartConfig(
        height=300,
        margin={"l": 40, "r": 20, "t": 40, "b": 40},
        show_legend=False,
    )
)

dashboard_chart = ChartFactory(
    ChartConfig(height=350, legend_position="right", legend_orientation="v")
)


# Export convenience functions
def create_bar_chart(
    data: pd.DataFrame, x: str, y: str, **kwargs
) -> go.Figure:
    """Create a bar chart with default settings."""
    return default_chart.bar_chart(data, x, y, **kwargs)


def create_line_chart(
    data: pd.DataFrame, x: str, y: Union[str, List[str]], **kwargs
) -> go.Figure:
    """Create a line chart with default settings."""
    return default_chart.line_chart(data, x, y, **kwargs)


def create_pie_chart(
    data: pd.DataFrame, values: str, names: str, **kwargs
) -> go.Figure:
    """Create a pie chart with default settings."""
    return default_chart.pie_chart(data, values, names, **kwargs)


def apply_chart_styling(fig: go.Figure, **kwargs) -> go.Figure:
    """Apply consistent styling to any figure."""
    return default_chart.apply_layout(fig, **kwargs)


__all__ = [
    "ChartFactory",
    "ChartConfig",
    "default_chart",
    "chart_factory",
    "compact_chart",
    "dashboard_chart",
    "create_bar_chart",
    "create_line_chart",
    "create_pie_chart",
    "apply_chart_styling",
]
