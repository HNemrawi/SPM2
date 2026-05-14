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
        default_factory=lambda: {"l": 60, "r": 30, "t": 50, "b": 60}
    )
    hover_mode: str = "closest"
    template: Optional[str] = None
    width: str = "stretch"
    animate: bool = False

    # Grid and axes
    show_grid: bool = True
    grid_color: str = "rgba(0, 0, 0, 0.08)"
    zero_line_color: str = "rgba(0, 0, 0, 0.15)"

    # Font settings
    font_family: Optional[str] = None
    font_size: int = 12
    title_font_size: int = 16

    # Colors
    color_sequence: Optional[List[str]] = None
    paper_bgcolor: str = "rgba(0, 0, 0, 0)"
    plot_bgcolor: str = "rgba(0, 0, 0, 0.01)"


class ChartFactory:
    """Factory class for creating consistent Plotly charts."""

    def __init__(self, config: Optional[ChartConfig] = None):
        self.config = config or ChartConfig()
        self.theme = theme
        self._setup_defaults()

    def _setup_defaults(self):
        """Setup default values from theme if not specified."""
        if not self.config.template:
            self.config.template = self.theme.chart.template
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

    def apply_sankey_layout(
        self,
        fig: go.Figure,
        title: Optional[str] = None,
        height: Optional[int] = None,
    ) -> go.Figure:
        """Apply consistent layout to a Sankey diagram.

        Sankey ignores xaxis/yaxis, so we strip them and use a generous
        margin appropriate for node labels. Font, paper/plot backgrounds,
        and hover styling are inherited from the global chart theme so
        Sankey diagrams across modules render uniformly.
        """
        layout: Dict[str, Any] = {
            "template": self.config.template,
            "paper_bgcolor": self.config.paper_bgcolor,
            "plot_bgcolor": self.config.plot_bgcolor,
            "font": {
                "family": self.config.font_family,
                "size": self.config.font_size,
                "color": self.theme.colors.text_primary,
            },
            "margin": {"l": 10, "r": 10, "t": 60, "b": 20},
            "hoverlabel": {
                "bgcolor": "rgba(255, 255, 255, 0.95)",
                "bordercolor": self.theme.colors.border,
                "font": {"color": self.theme.colors.text_primary},
            },
            "autosize": True,
        }
        if height is not None:
            layout["height"] = height
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


# ============================================================================
# SHARED FLOW SANKEY BUILDER
# ============================================================================
# Single source of truth for the three exit→entry/return Sankey diagrams used
# by SPM2, Inbound Recidivism, and Outbound Recidivism. Replaces three
# near-duplicate hand-built Sankeys that had drifted in styling, fonts, and
# hover content. Palette is the existing brand blue ramp
# (theme.colors.primary / primary_light) — no red/green semantic encoding.
# Source nodes use the darker primary, target nodes use the lighter shade,
# and link bands are primary-tinted with opacity scaled by flow share.


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a #RRGGBB string to rgba() with the given alpha."""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return hex_color
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def _pluralize_role(role: str) -> str:
    """Pluralize a flow-Sankey role label for column headers.

    Handles the common HMIS roles: Entry → Entries, Exit → Exits,
    Return → Returns, Prior Exit → Prior Exits, Current Entry → Current
    Entries. Falls back to ``role + 's'`` for unknown forms.
    """
    if not role:
        return role
    lower = role.lower()
    vowels = {"a", "e", "i", "o", "u"}
    if lower.endswith("y") and (len(lower) < 2 or lower[-2] not in vowels):
        return role[:-1] + "ies"
    if lower.endswith(("s", "x", "z")) or lower.endswith(("ch", "sh")):
        return role + "es"
    return role + "s"


def _empty_sankey_figure(title: str, message: str) -> go.Figure:
    """Themed empty-state placeholder for a Sankey figure."""
    family = theme.typography.font_family.replace('"', "")
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=16, color=theme.colors.text_muted, family=family),
    )
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(
                size=18,
                color=theme.colors.text_primary,
                family=family,
                weight=600,
            ),
            x=0.5,
            xanchor="center",
        ),
        xaxis_visible=False,
        yaxis_visible=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=400,
        margin=dict(l=40, r=40, t=80, b=40),
    )
    return fig


def build_flow_sankey(
    pivot_df: pd.DataFrame,
    title: str,
    *,
    source_role: str,
    target_role: str,
    accent_targets: Optional[frozenset] = None,
    empty_message: str = "No flows available",
) -> go.Figure:
    """Build a flow Sankey from an exit×target crosstab pivot.

    Used by all three flow Sankey charts in the app (SPM2, Inbound, Outbound).
    Replaces three near-duplicate hand-built implementations.

    Palette is the existing brand blue ramp from
    ``theme.colors.chart_colors_sequential``: source nodes use the darker
    primary blue, target nodes use the lighter ``primary_light``, links are
    primary-tinted with opacity scaled by flow share so larger flows stand
    out without smaller ones disappearing.

    Args:
        pivot_df: Crosstab. Rows = source/exit categories, columns =
            target/entry/return categories. Cell values are flow counts.
        title: Diagram title (rendered in Inter, centered).
        source_role: Human-readable role for the left-side nodes
            (e.g., "Exit", "Prior Exit"). Used in hover and column header.
        target_role: Human-readable role for the right-side nodes
            (e.g., "Return", "Current Entry"). Used in hover and column
            header.
        accent_targets: Optional set of target-side labels that should
            render in the in-theme green-teal accent
            (``chart_colors_categorical[2]``, ``#009E73`` — Wong "Bluish
            green") instead of the default ``primary_light``. Used by SPM2
            and Outbound to highlight "No Return" as the analytic positive
            sink without leaving the existing palette.
        empty_message: Text shown when ``pivot_df`` is empty or has no
            non-zero flows.

    Returns:
        A themed go.Figure ready for ``st.plotly_chart``.
    """
    if pivot_df is None or pivot_df.empty:
        return _empty_sankey_figure(title, empty_message)

    df = pivot_df.copy()
    source_cats = [str(c) for c in df.index.tolist()]
    target_cats = [str(c) for c in df.columns.tolist()]
    n_source = len(source_cats)
    n_target = len(target_cats)

    sources: List[int] = []
    targets: List[int] = []
    values: List[float] = []
    for i, scat in enumerate(source_cats):
        for j, tcat in enumerate(target_cats):
            count = df.loc[df.index[i], df.columns[j]]
            try:
                count = float(count)
            except (TypeError, ValueError):
                continue
            if count > 0:
                sources.append(i)
                targets.append(n_source + j)
                values.append(count)

    if not values:
        return _empty_sankey_figure(title, empty_message)

    total = sum(values)
    max_value = max(values)
    family = theme.typography.font_family.replace('"', "")

    # Single-hue brand palette. Source side uses the dark anchor, target
    # side uses the lighter shade — visually distinguishes the two columns
    # while staying entirely in the existing blue theme. Targets in
    # ``accent_targets`` (e.g., "No Return") get the in-theme green-teal so
    # the positive analytic sink reads at a glance.
    source_color = theme.colors.primary
    target_color = theme.colors.primary_light
    # Wong "Bluish green" — green-teal, colorblind-safe, already in theme.
    accent_color = theme.colors.chart_colors_categorical[2]  # #009E73
    accent_set = accent_targets or frozenset()

    target_node_colors = [
        accent_color if tcat in accent_set else target_color
        for tcat in target_cats
    ]
    node_colors = [source_color] * n_source + target_node_colors
    node_labels = source_cats + target_cats
    node_roles = [source_role] * n_source + [target_role] * n_target

    # Per-node share of total, for hover.
    node_totals: List[float] = [0.0] * (n_source + n_target)
    for s_idx, t_idx, v in zip(sources, targets, values):
        node_totals[s_idx] += v
        node_totals[t_idx] += v
    node_shares = [
        (nt / total * 100.0) if total else 0.0 for nt in node_totals
    ]

    # Per-source-node total, for "share of source" link hover.
    source_totals: Dict[int, float] = {}
    for s_idx, v in zip(sources, values):
        source_totals[s_idx] = source_totals.get(s_idx, 0.0) + v

    # Link colors: target-tinted, opacity scaled by flow share so big flows
    # stand out without small flows disappearing. Range [0.10, 0.35] keeps
    # the link bands clearly secondary to the node bars.
    link_colors: List[str] = []
    link_share_of_total: List[float] = []
    link_share_of_source: List[float] = []
    for t_idx, s_idx, v in zip(targets, sources, values):
        base = node_colors[t_idx]
        share = (v / max_value) if max_value else 0.0
        alpha = 0.10 + share * 0.25
        link_colors.append(_hex_to_rgba(base, alpha))
        link_share_of_total.append((v / total * 100.0) if total else 0.0)
        src_total = source_totals.get(s_idx, 0.0)
        link_share_of_source.append(
            (v / src_total * 100.0) if src_total else 0.0
        )

    # No-cutoff margins: scale to longest label per side.
    longest_left = max((len(s) for s in source_cats), default=0)
    longest_right = max((len(s) for s in target_cats), default=0)
    left_margin = max(80, min(280, longest_left * 12))
    right_margin = max(80, min(280, longest_right * 12))

    # Height scales with node count. Per-node multiplier and top/bottom
    # margins are sized so node labels never clip against the canvas edge.
    # Plotly renders Sankey node labels at the vertical center of each
    # node bar, so the bottommost/topmost nodes need extra clearance — we
    # give the figure both generous per-node spacing and large vertical
    # margins. Cap is intentionally high (page scrolls if needed).
    num_nodes = max(n_source, n_target)
    calculated_height = max(720, min(2000, num_nodes * 62 + 280))

    # Node-level customdata = [role, share_of_total].
    node_customdata = list(zip(node_roles, node_shares))
    # Link-level customdata = [share_of_total, share_of_source].
    link_customdata = list(zip(link_share_of_total, link_share_of_source))

    sankey = go.Sankey(
        arrangement="snap",
        orientation="h",
        valueformat=",.0f",
        node=dict(
            pad=30,
            thickness=22,
            line=dict(color=theme.colors.border, width=1),
            label=node_labels,
            color=node_colors,
            customdata=node_customdata,
            hovertemplate=(
                "<b>%{label}</b><br>"
                "%{customdata[0]}: %{value:,} clients<br>"
                "%{customdata[1]:.1f}% of total"
                "<extra></extra>"
            ),
            x=[0.001] * n_source + [0.999] * n_target,
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
            customdata=link_customdata,
            hovertemplate=(
                "<b>%{source.label}</b> → <b>%{target.label}</b><br>"
                "%{value:,} clients<br>"
                "%{customdata[0]:.1f}% of total · "
                "%{customdata[1]:.1f}% of source"
                "<extra></extra>"
            ),
        ),
        textfont=dict(
            color=theme.colors.text_primary,
            size=12,
            family=family,
        ),
    )

    fig = go.Figure(data=[sankey])

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(
                size=18,
                color=theme.colors.text_primary,
                family=family,
                weight=600,
            ),
            x=0.5,
            xanchor="center",
            y=0.98,
            yanchor="top",
        ),
        font=dict(
            size=12,
            color=theme.colors.text_secondary,
            family=family,
        ),
        height=calculated_height,
        margin=dict(
            l=left_margin,
            r=right_margin,
            t=130,
            b=130,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(
            bgcolor="rgba(30, 41, 59, 0.95)",
            font=dict(color="white", size=13, family=family),
            bordercolor="rgba(255, 255, 255, 0.3)",
            namelength=-1,
        ),
        autosize=True,
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        annotations=[
            dict(
                text=f"← {_pluralize_role(source_role)}",
                xref="paper",
                yref="paper",
                x=0.0,
                y=1.06,
                xanchor="left",
                yanchor="bottom",
                showarrow=False,
                font=dict(
                    size=13,
                    color=theme.colors.text_secondary,
                    family=family,
                    weight=600,
                ),
            ),
            dict(
                text=f"{_pluralize_role(target_role)} →",
                xref="paper",
                yref="paper",
                x=1.0,
                y=1.06,
                xanchor="right",
                yanchor="bottom",
                showarrow=False,
                font=dict(
                    size=13,
                    color=theme.colors.text_secondary,
                    family=family,
                    weight=600,
                ),
            ),
        ],
    )

    return fig


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
    "build_flow_sankey",
]
