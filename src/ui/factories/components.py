"""
Component Factory Module
=======================
Unified component creation system that combines HTML, charts, and Streamlit components.
This is the main interface for creating consistent UI elements across the application.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.ui.factories.charts import default_chart
from src.ui.factories.formatters import (
    fmt_change,
    fmt_currency,
    fmt_duration,
    fmt_float,
    fmt_int,
    fmt_number,
    fmt_pct,
    fmt_ratio,
)
from src.ui.factories.html import html_factory
from src.ui.themes.theme import professional_colors, theme

# ==================== HELPER FUNCTIONS ====================


def safe_background_gradient(styled, **kwargs):
    """
    Apply color gradient styling without requiring matplotlib.
    Uses custom color mapping instead of background_gradient.

    Args:
        styled: Pandas Styler object
        **kwargs: Arguments (cmap, subset, axis, vmin, vmax)

    Returns:
        Styled object with color gradient applied
    """

    def color_cell(val, vmin, vmax, cmap_name):
        """Color a cell based on its value."""
        if pd.isna(val) or vmin == vmax:
            return ''

        # Normalize value between 0 and 1
        norm_val = (val - vmin) / (vmax - vmin)
        norm_val = max(0, min(1, norm_val))  # Clamp between 0 and 1

        # Define simple color maps
        if cmap_name in ['Blues', 'blues']:
            # Light blue to dark blue
            r = int(240 - norm_val * 100)
            g = int(248 - norm_val * 80)
            b = int(255 - norm_val * 50)
        elif cmap_name in ['RdYlGn', 'rdylgn']:
            # Red to Yellow to Green
            if norm_val < 0.5:
                # Red to Yellow
                r = 255
                g = int(norm_val * 2 * 255)
                b = 0
            else:
                # Yellow to Green
                r = int((2 - norm_val * 2) * 255)
                g = 255
                b = 0
        else:
            # Default: light gray to dark gray
            gray = int(255 - norm_val * 80)
            r = g = b = gray

        return f'background-color: rgb({r}, {g}, {b})'

    # Extract parameters
    cmap = kwargs.get('cmap', 'Blues')
    subset = kwargs.get('subset', None)
    kwargs.get('axis', 0)
    vmin = kwargs.get('vmin', None)
    vmax = kwargs.get('vmax', None)

    # Get the dataframe from the styler
    df = styled.data

    # Determine columns to apply styling to
    if subset:
        cols_to_style = subset if isinstance(subset, list) else [subset]
    else:
        cols_to_style = df.select_dtypes(include=['number']).columns.tolist()

    # Apply styling to each column
    for col in cols_to_style:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            col_vmin = vmin if vmin is not None else df[col].min()
            col_vmax = vmax if vmax is not None else df[col].max()

            styled = styled.map(
                lambda val: color_cell(val, col_vmin, col_vmax, cmap),
                subset=[col],
            )

    return styled


# ==================== CONFIGURATION ====================


@dataclass
class ComponentConfig:
    """Global configuration for UI components."""

    use_containers: bool = True
    show_borders: bool = True
    enable_animations: bool = True
    responsive: bool = True


class UIComponentFactory:
    """
    Master factory for creating all UI components.
    Combines HTML, charts, and Streamlit components into a unified interface.
    """

    def __init__(self, config: Optional[ComponentConfig] = None):
        self.config = config or ComponentConfig()
        self.html = html_factory
        self.chart = default_chart
        self.theme = theme

    # ============== SECTION COMPONENTS ==============

    def section_header(
        self,
        title: str,
        subtitle: Optional[str] = None,
        icon: Optional[str] = None,
        divider: bool = True,
    ) -> None:
        """Create a section header with optional subtitle and divider."""
        icon_html = (
            f'<span style="margin-right: 0.75rem;">{icon}</span>'
            if icon
            else ""
        )
        subtitle_html = (
            f"""
        <p style="
            color: {self.theme.colors.text_secondary};
            font-size: {self.theme.typography.size_base};
            margin: 0.5rem 0 0 0;
        ">{subtitle}</p>
        """
            if subtitle
            else ""
        )

        st.html(
            f"""
        <div style="margin: 2rem 0 1rem 0;">
            <h2 style="
                color: {self.theme.colors.text_primary};
                font-size: {self.theme.typography.size_2xl};
                font-weight: {self.theme.typography.weight_semibold};
                margin: 0;
            ">
                {icon_html}{title}
            </h2>
            {subtitle_html}
        </div>
        """
        )

        if divider:
            st.html(self.html.divider(style="gradient"))

    def info_section(
        self,
        content: str,
        type: str = "info",
        title: Optional[str] = None,
        icon: Optional[str] = None,
        expanded: bool = True,
    ) -> None:
        """Create an information section with consistent styling."""
        if expanded:
            st.html(self.html.info_box(content, type, title, icon))
        else:
            with st.expander(
                f"{icon or ''} {title or 'Information'}", expanded=False
            ):
                st.markdown(content)

    def card_section(
        self,
        content: Callable,
        title: Optional[str] = None,
        icon: Optional[str] = None,
        border_color: Optional[str] = None,
    ) -> None:
        """Create a card section with content rendered inside."""
        if self.config.use_containers:
            st.html(
                f"""
            <div style="
                background: {self.theme.colors.surface};
                border: 1px solid {self.theme.colors.border};
                border-left: 4px solid {border_color or self.theme.colors.primary};
                border-radius: {self.theme.borders.radius_md};
                padding: 1.5rem;
                margin: 1rem 0;
            ">
            """
            )

            if title:
                st.html(html_factory.title(title, level=3, icon=icon))

            content()

            st.html("</div>")
        else:
            if title:
                st.html(html_factory.title(title, level=3, icon=icon))
            content()

    # ============== METRIC COMPONENTS ==============

    def metric_row(
        self,
        metrics: Dict[str, Any],
        columns: Optional[int] = None,
        colors: Optional[List[str]] = None,
        icons: Optional[List[str]] = None,
    ) -> None:
        """Create a row of metric cards."""
        if not metrics:
            return

        # Auto-determine columns if not specified
        if columns is None:
            columns = min(len(metrics), 4)

        # Default colors
        if colors is None:
            colors = [self.theme.colors.primary] * len(metrics)

        # Create columns
        cols = st.columns(columns)

        # Render metrics
        for idx, (label, value) in enumerate(list(metrics.items())[:columns]):
            with cols[idx]:
                # Format value
                if isinstance(value, dict):
                    # Complex metric with value and delta
                    main_value = value.get("value", 0)
                    delta = value.get("delta")
                else:
                    main_value = value
                    delta = None

                # Create metric card HTML
                icon = icons[idx] if icons and idx < len(icons) else None
                color = (
                    colors[idx]
                    if idx < len(colors)
                    else self.theme.colors.primary
                )

                st.html(
                    self.html.metric_card(
                        label, main_value, delta, color=color, icon=icon
                    )
                )

    def apply_metric_card_style(
        self,
        border_color: Optional[str] = None,
        background_gradient: bool = True,
        box_shadow: bool = True,
    ) -> None:
        """
        Apply professional metric card styling to the current container.
        This provides consistent styling across all metric displays.

        Args:
            border_color: Color for the left border accent
            background_gradient: Whether to apply gradient background
            box_shadow: Whether to apply shadow effect
        """
        border_color = border_color or self.theme.colors.primary

        gradient_bg = (
            "background: linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(248,250,252,0.9) 100%);"
            if background_gradient
            else ""
        )
        shadow = (
            "box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08), 0 1px 3px rgba(0, 0, 0, 0.05);"
            if box_shadow
            else ""
        )

        st.html(
            f"""
        <style>
        div[data-testid="metric-container"] {{
            {gradient_bg}
            border: 1px solid {self.theme.colors.border};
            border-left: 4px solid {border_color};
            border-radius: {self.theme.borders.radius_lg};
            padding: 1.25rem;
            margin: 0.5rem 0;
            {shadow}
            transition: all 0.3s ease;
            min-height: auto !important;
            height: auto !important;
            overflow: visible !important;
        }}

        div[data-testid="metric-container"]:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
        }}

        div[data-testid="metric-container"] label {{
            color: {self.theme.colors.text_muted};
            font-size: {self.theme.typography.size_sm};
            font-weight: {self.theme.typography.weight_medium};
            text-transform: uppercase;
            letter-spacing: 0.05em;
            white-space: normal !important;
            word-wrap: break-word !important;
            line-height: 1.4;
        }}

        div[data-testid="metric-container"] [data-testid="metric-value"] {{
            color: {self.theme.colors.text_primary};
            font-weight: {self.theme.typography.weight_bold};
            font-size: clamp(1.25rem, 2vw, {self.theme.typography.size_2xl});
            white-space: normal !important;
            word-wrap: break-word !important;
            overflow-wrap: break-word !important;
            line-height: 1.2;
        }}

        div[data-testid="metric-container"] [data-testid="metric-delta"] {{
            font-size: {self.theme.typography.size_sm};
            font-weight: {self.theme.typography.weight_medium};
            white-space: normal !important;
            word-wrap: break-word !important;
        }}

        /* Ensure metric containers in columns have equal height */
        div[data-testid="column"] > div > div[data-testid="metric-container"] {{
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }}
        </style>
        """
        )

    def metric_grid(
        self,
        metrics: List[Dict[str, Any]],
        columns: int = 3,
        group_by: Optional[str] = None,
    ) -> None:
        """Create a grid of metrics with optional grouping."""
        if group_by:
            # Group metrics by specified key
            groups = {}
            for metric in metrics:
                group = metric.get(group_by, "Other")
                if group not in groups:
                    groups[group] = []
                groups[group].append(metric)

            # Render each group
            for group_name, group_metrics in groups.items():
                st.html(html_factory.title(group_name, level=4, icon="📊"))
                self._render_metric_grid(group_metrics, columns)
                st.html(self.html.divider())
        else:
            self._render_metric_grid(metrics, columns)

    def _render_metric_grid(
        self, metrics: List[Dict[str, Any]], columns: int
    ) -> None:
        """Helper to render a grid of metrics."""
        for i in range(0, len(metrics), columns):
            cols = st.columns(columns)
            for j, col in enumerate(cols):
                if i + j < len(metrics):
                    metric = metrics[i + j]
                    with col:
                        st.metric(
                            label=metric.get("label", ""),
                            value=metric.get("value", ""),
                            delta=metric.get("delta"),
                            delta_color=metric.get("delta_color", "normal"),
                        )

    # ============== DATA DISPLAY COMPONENTS ==============

    def data_table(
        self,
        df: pd.DataFrame,
        title: Optional[str] = None,
        description: Optional[str] = None,
        highlight_columns: Optional[List[str]] = None,
        show_index: bool = False,
        height: Optional[int] = None,
        downloadable: bool = True,
        download_name: str = "data",
    ) -> None:
        """Create a formatted data table with optional features."""
        if title:
            st.html(html_factory.title(title, level=3, icon="📊"))

        if description:
            st.caption(description)

        # Display the dataframe
        if highlight_columns:
            # Apply highlighting
            def highlight_cols(s):
                return [
                    (
                        "background-color: rgba(0, 102, 204, 0.1)"
                        if s.name in highlight_columns
                        else ""
                    )
                    for _ in s
                ]

            styled_df = df.style.apply(highlight_cols, axis=0)
            st.dataframe(
                styled_df,
                width='stretch',
                hide_index=not show_index,
                height=height,
            )
        else:
            st.dataframe(
                df,
                width='stretch',
                hide_index=not show_index,
                height=height,
            )

        # Add download button if requested
        if downloadable and not df.empty:
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"{download_name}.csv",
                mime="text/csv",
                width='content',
            )

    def comparison_table(
        self,
        data: Dict[str, Dict[str, Any]],
        title: Optional[str] = None,
        highlight_best: bool = True,
    ) -> None:
        """Create a comparison table from nested dictionary data."""
        df = pd.DataFrame(data).T

        if title:
            st.html(html_factory.title(title, level=3, icon="📊"))

        if highlight_best:
            # Apply conditional formatting
            styled = df.style
            for col in df.select_dtypes(include=["number"]).columns:
                styled = safe_background_gradient(
                    styled,
                    subset=[col],
                    cmap="RdYlGn",
                    vmin=df[col].min(),
                    vmax=df[col].max(),
                )
            st.dataframe(styled, width='stretch')
        else:
            st.dataframe(df, width='stretch')

    # ============== CHART WRAPPER COMPONENTS ==============

    def chart_container(
        self,
        chart_func: Callable[[], go.Figure],
        title: Optional[str] = None,
        description: Optional[str] = None,
        help_text: Optional[str] = None,
        key: Optional[str] = None,
    ) -> None:
        """Create a chart with consistent container styling."""
        if title:
            col1, col2 = st.columns([10, 1])
            with col1:
                st.html(html_factory.title(title, level=3, icon="📊"))
            with col2:
                if help_text:
                    st.markdown("ℹ️", help=help_text)

        if description:
            st.caption(description)

        # Render the chart
        fig = chart_func()
        st.plotly_chart(fig, width='stretch', key=key)

    # ============== TAB FACTORY COMPONENTS ==============

    def analysis_tabs(
        self, tabs: List[Dict[str, str]], icons: Dict[str, str] = None
    ) -> List[Any]:
        """Create standardized analysis tabs with consistent styling.

        Args:
            tabs: List of dicts with 'key' and 'label'
            icons: Optional dict mapping tab keys to icons

        Returns:
            List of Streamlit tab objects

        Example:
            tabs = [
                {"key": "overview", "label": "Overview"},
                {"key": "details", "label": "Details"}
            ]
            icons = {"overview": "📊", "details": "📋"}
        """
        icons = icons or {}

        tab_labels = []
        for tab in tabs:
            key = tab["key"]
            label = tab["label"]
            icon = icons.get(key, "")

            # Format with icon if provided
            if icon:
                tab_labels.append(f"{icon} {label}")
            else:
                tab_labels.append(label)

        return st.tabs(tab_labels)

    def dashboard_tabs(self) -> List[Any]:
        """Create standardized dashboard tabs."""
        return self.analysis_tabs(
            tabs=[
                {"key": "overview", "label": "Overview"},
                {"key": "flow", "label": "System Flow"},
                {"key": "outcomes", "label": "Outcomes Analysis"},
                {"key": "data", "label": "Data Table"},
            ],
            icons={
                "overview": "📊",
                "flow": "🔄",
                "outcomes": "🎯",
                "data": "📋",
            },
        )

    def equity_tabs(self) -> List[Any]:
        """Create standardized equity analysis tabs."""
        return self.analysis_tabs(
            tabs=[
                {"key": "rates", "label": "Outcome Rates"},
                {"key": "disparity", "label": "Disparity Analysis"},
                {"key": "details", "label": "Data & Methodology"},
            ],
            icons={"rates": "📊", "disparity": "📈", "details": "📋"},
        )

    def los_tabs(self) -> List[Any]:
        """Create standardized length of stay tabs."""
        return self.analysis_tabs(
            tabs=[
                {"key": "overview", "label": "Overview"},
                {"key": "demographics", "label": "Demographics"},
            ],
            icons={"overview": "📊", "demographics": "👥"},
        )

    def main_dashboard_tabs(self) -> List[Any]:
        """Create standardized main dashboard tabs."""
        return self.analysis_tabs(
            tabs=[
                {"key": "overview", "label": "Overview"},
                {"key": "demographics", "label": "Demographics"},
                {"key": "trends", "label": "Trends"},
                {"key": "los", "label": "Length of Stay"},
                {"key": "equity", "label": "Equity Analysis"},
                {"key": "export", "label": "Export Data"},
            ],
            icons={
                "overview": "📊",
                "demographics": "👥",
                "trends": "📈",
                "los": "⏱️",
                "equity": "⚖️",
                "export": "💾",
            },
        )

    def chart_tabs(
        self, tabs: List[Dict[str, Any]], key: str = "chart_tabs"
    ) -> None:
        """Create tabbed charts."""
        tab_names = [tab["name"] for tab in tabs]
        tab_objects = st.tabs(tab_names)

        for tab_obj, tab_config in zip(tab_objects, tabs):
            with tab_obj:
                if "description" in tab_config:
                    st.caption(tab_config["description"])

                # Render chart or content
                if "chart" in tab_config:
                    st.plotly_chart(
                        tab_config["chart"], width='stretch'
                    )
                elif "content" in tab_config:
                    tab_config["content"]()

    # ============== FILTER COMPONENTS ==============

    def filter_sidebar(
        self, filters: List[Dict[str, Any]], reset_button: bool = True
    ) -> Dict[str, Any]:
        """Create a standardized filter sidebar."""
        results = {}

        with st.sidebar:
            st.html(html_factory.title("Filters", level=3, icon="🔍"))

            if reset_button:
                if st.button("🔄 Reset Filters", width='stretch'):
                    st.rerun()

            st.html(self.html.divider())

            for filter_config in filters:
                filter_type = filter_config.get("type", "select")
                label = filter_config.get("label", "")
                key = filter_config.get("key", label.lower().replace(" ", "_"))

                if filter_type == "select":
                    results[key] = st.selectbox(
                        label,
                        options=filter_config.get("options", []),
                        index=filter_config.get("default", 0),
                        help=filter_config.get("help"),
                    )
                elif filter_type == "multiselect":
                    results[key] = st.multiselect(
                        label,
                        options=filter_config.get("options", []),
                        default=filter_config.get("default", []),
                        help=filter_config.get("help"),
                    )
                elif filter_type == "slider":
                    results[key] = st.slider(
                        label,
                        min_value=filter_config.get("min", 0),
                        max_value=filter_config.get("max", 100),
                        value=filter_config.get("default", 50),
                        step=filter_config.get("step", 1),
                        help=filter_config.get("help"),
                    )
                elif filter_type == "date":
                    results[key] = st.date_input(
                        label,
                        value=filter_config.get("default"),
                        help=filter_config.get("help"),
                    )
                elif filter_type == "number":
                    results[key] = st.number_input(
                        label,
                        min_value=filter_config.get("min"),
                        max_value=filter_config.get("max"),
                        value=filter_config.get("default", 0),
                        step=filter_config.get("step", 1),
                        help=filter_config.get("help"),
                    )

                # Add spacing between filters
                if filter_config != filters[-1]:
                    st.html("<br>")

        return results

    # ============== LAYOUT COMPONENTS ==============

    def columns_with_gap(self, ratios: List[int], gap: str = "2rem") -> List:
        """Create columns with custom gap."""
        st.html(
            f"""
        <style>
        .row-widget.stHorizontalBlock {{
            gap: {gap};
        }}
        </style>
        """
        )

        return st.columns(ratios)

    def expandable_section(
        self,
        title: str,
        content: Callable,
        expanded: bool = False,
        icon: Optional[str] = None,
    ) -> None:
        """Create an expandable section."""
        with st.expander(f"{icon or ''} {title}", expanded=expanded):
            content()

    # ============== PROGRESS INDICATORS ==============

    def progress_indicator(
        self,
        current: float,
        total: float,
        label: Optional[str] = None,
        format_string: str = "{current}/{total} ({percentage:.1f}%)",
    ) -> None:
        """Create a progress indicator with label."""
        percentage = (current / total * 100) if total else 0

        if label:
            st.markdown(f"**{label}**")

        # Progress bar
        st.progress(percentage / 100)

        # Progress text
        st.caption(
            format_string.format(
                current=current, total=total, percentage=percentage
            )
        )

    def loading_message(
        self, message: str = "Loading...", spinner: bool = True
    ) -> None:
        """Display a loading message."""
        if spinner:
            with st.spinner(message):
                return
        else:
            st.info(message)

    # ============== NAVIGATION COMPONENTS ==============

    def breadcrumb(self, items: List[str], separator: str = "›") -> None:
        """Create a breadcrumb navigation."""
        breadcrumb_html = f' <span style="color: {self.theme.colors.text_muted};"> {separator} </span> '.join(
            [
                f'<span style="color: {self.theme.colors.text_secondary};">{item}</span>'
                for item in items
            ]
        )

        st.html(
            f"""
        <div style="
            padding: 0.5rem 0;
            margin-bottom: 1rem;
            font-size: {self.theme.typography.size_sm};
        ">
            {breadcrumb_html}
        </div>
        """
        )

    def pagination(
        self,
        total_items: int,
        items_per_page: int,
        current_page: int,
        key: str = "pagination",
    ) -> int:
        """Create pagination controls."""
        total_pages = (total_items + items_per_page - 1) // items_per_page

        col1, col2, col3 = st.columns([1, 2, 1])

        with col1:
            if current_page > 1:
                if st.button("← Previous", key=f"{key}_prev"):
                    return current_page - 1

        with col2:
            st.html(
                f"""
            <div style="text-align: center; padding: 0.5rem;">
                Page {current_page} of {total_pages}
            </div>
            """
            )

        with col3:
            if current_page < total_pages:
                if st.button("Next →", key=f"{key}_next"):
                    return current_page + 1

        return current_page

    def insight_container(
        self,
        title: str,
        content: str,
        icon: Optional[str] = None,
        color: Optional[str] = None,
        type: str = "info",
    ) -> None:
        """
        Create a styled insight container.

        Parameters:
            title: Container title
            content: Content to display
            icon: Optional emoji icon
            color: Border color (defaults based on type)
            type: Container type ('info', 'success', 'warning', 'error')
        """
        # Type configuration
        type_config = {
            "info": {
                "color": color or professional_colors.INFO,
                "bg": professional_colors.GRADIENTS["card"],
                "border": color or professional_colors.INFO,
                "icon": icon or "ℹ️",
            },
            "success": {
                "color": color or professional_colors.SUCCESS,
                "bg": professional_colors.GRADIENTS["card"],
                "border": color or professional_colors.SUCCESS,
                "icon": icon or "✅",
            },
            "warning": {
                "color": color or professional_colors.WARNING,
                "bg": professional_colors.GRADIENTS["card"],
                "border": color or professional_colors.WARNING,
                "icon": icon or "⚠️",
            },
            "error": {
                "color": color or professional_colors.ERROR,
                "bg": professional_colors.GRADIENTS["card"],
                "border": color or professional_colors.ERROR,
                "icon": icon or "❌",
            },
        }

        config = type_config.get(type, type_config["info"])
        display_icon = icon or config["icon"]

        st.html(
            f"""
        <div style="
            padding: 1.25rem;
            border-radius: 12px;
            margin-bottom: 1rem;
            border: 1px solid {config['border']};
            border-left: 4px solid {config['border']};
            background: {config['bg']};
            box-shadow: {professional_colors.SHADOWS['card']};
            transition: all 0.3s ease;
        ">
            <h4 style="
                color: {config['color']};
                margin: 0 0 0.75rem 0;
                font-size: 1.1rem;
                font-weight: 600;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            ">
                <span style="font-size: 1.2rem;">{display_icon}</span>
                {title}
            </h4>
            <div style="
                color: {self.theme.colors.text_secondary};
                font-size: 0.95rem;
                line-height: 1.6;
            ">
                {content}
            </div>
        </div>
        """
        )

    def styled_metric(
        self,
        label: str,
        value: Union[str, int, float],
        delta: Optional[Union[str, int, float]] = None,
        delta_color: str = "normal",
        help: Optional[str] = None,
    ) -> None:
        """
        Display a styled metric with professional design.

        Parameters:
            label: Metric label
            value: Metric value
            delta: Optional delta value
            delta_color: Color scheme for delta ('normal', 'inverse', 'off')
            help: Optional help text
        """
        # Format value if numeric
        if isinstance(value, (int, float)):
            if isinstance(value, int) or value.is_integer():
                formatted_value = fmt_int(value)
            else:
                formatted_value = fmt_float(value)
        else:
            formatted_value = str(value)

        # Format delta if provided
        formatted_delta = None
        if delta is not None:
            if isinstance(delta, (int, float)):
                formatted_delta = fmt_change(delta, with_sign=True)
            else:
                formatted_delta = str(delta)

        # Apply metric card styling
        self.apply_metric_card_style()

        # Use Streamlit's metric with enhanced styling
        st.metric(
            label=label,
            value=formatted_value,
            delta=formatted_delta,
            delta_color=delta_color,
            help=help,
        )

    def date_range_input(
        self,
        label: str,
        default_start: Optional[datetime] = None,
        default_end: Optional[datetime] = None,
        help_text: str = "Select date range for analysis",
        info_message: Optional[str] = None,
        on_change_callback: Optional[Callable] = None,
        df: Optional[pd.DataFrame] = None,
        validate_against_data: bool = False,
    ) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        """Reusable date range input component with standardized validation.

        Parameters:
            label: Label for the date input
            default_start: Default start date
            default_end: Default end date
            help_text: Help text for the input
            info_message: Optional info message to display
            on_change_callback: Optional callback function for changes
            df: Optional DataFrame to validate against (requires ProjectStart/Exit columns)
            validate_against_data: Whether to show data range warnings

        Returns:
            Tuple of (start_date, end_date) as pd.Timestamp objects, or (None, None) if invalid
        """
        from datetime import date, datetime

        # Set defaults if not provided
        if default_start is None:
            default_start = datetime(2024, 10, 1)
        if default_end is None:
            default_end = datetime(2025, 9, 30)

        # Render the date input
        date_range = st.date_input(
            label,
            [default_start, default_end],
            help=help_text,
            on_change=on_change_callback,
        )

        # Show info message if provided
        if info_message:
            self.info_section(
                content=info_message,
                type="info",
                icon="📌",
                expanded=True,
            )

        # Validate and process date range
        try:
            if date_range is None:
                st.error("⚠️ Please select both start and end dates.")
                return None, None
            elif isinstance(date_range, (list, tuple)):
                if len(date_range) == 2:
                    start_date = pd.to_datetime(date_range[0])
                    end_date = pd.to_datetime(date_range[1])
                elif len(date_range) == 1:
                    st.error("⚠️ Please select an end date.")
                    return None, None
                else:
                    st.error("⚠️ Please select both start and end dates.")
                    return None, None
            elif isinstance(date_range, (datetime, date)):
                st.error("⚠️ Please select an end date.")
                return None, None
            else:
                st.error("⚠️ Invalid date selection. Please select both dates.")
                return None, None

            # Validate date order
            if start_date > end_date:
                st.error(
                    "❌ Start date must be before end date. Dates have been swapped."
                )
                start_date, end_date = end_date, start_date

            # Optional data range validation
            if validate_against_data and df is not None and not df.empty:
                try:
                    from src.core.utils.helpers import (
                        show_universal_date_warning,
                    )

                    # Get data boundaries from ReportingPeriod columns (the official data range)
                    if (
                        "ReportingPeriodStartDate" in df.columns
                        and "ReportingPeriodEndDate" in df.columns
                    ):
                        data_start = pd.to_datetime(
                            df["ReportingPeriodStartDate"].iloc[0]
                        )
                        data_end = pd.to_datetime(
                            df["ReportingPeriodEndDate"].iloc[0]
                        )

                        # Check for issues
                        issues = []
                        if start_date < data_start:
                            days_before = (data_start - start_date).days
                            issues.append(
                                f"Start date is {days_before} days before reporting period"
                            )
                        if end_date > data_end:
                            days_after = (end_date - data_end).days
                            issues.append(
                                f"End date is {days_after} days after reporting period"
                            )

                        # Show warning if there are issues
                        if issues:
                            show_universal_date_warning(
                                issues,
                                start_date,
                                end_date,
                                data_start,
                                data_end,
                                df,
                            )
                except Exception:
                    # Don't break functionality if validation fails
                    pass

            return start_date, end_date

        except Exception as e:
            st.error(f"📅 Date processing error: {str(e)}")
            return None, None


# Create global instance
ui = UIComponentFactory()


# Create convenience aliases for Colors (for backward compatibility)
class Colors:
    """Backward compatibility wrapper for professional colors."""

    PRIMARY = professional_colors.PRIMARY
    PRIMARY_HOVER = professional_colors.PRIMARY_DARK
    PRIMARY_LIGHT = professional_colors.PRIMARY_LIGHT
    SUCCESS = professional_colors.SUCCESS
    SUCCESS_LIGHT = professional_colors.SUCCESS_LIGHT
    WARNING = professional_colors.WARNING
    WARNING_LIGHT = professional_colors.WARNING_LIGHT
    DANGER = professional_colors.ERROR
    DANGER_LIGHT = professional_colors.ERROR_LIGHT
    INFO = professional_colors.INFO
    INFO_LIGHT = professional_colors.INFO_LIGHT

    # Border color
    BORDER_COLOR = professional_colors.NEUTRAL_300

    # Neutral colors
    NEUTRAL_900 = professional_colors.NEUTRAL_900
    NEUTRAL_800 = professional_colors.NEUTRAL_800
    NEUTRAL_700 = professional_colors.NEUTRAL_700
    NEUTRAL_600 = professional_colors.NEUTRAL_600
    NEUTRAL_500 = professional_colors.NEUTRAL_500
    NEUTRAL_400 = professional_colors.NEUTRAL_400
    NEUTRAL_300 = professional_colors.NEUTRAL_300
    NEUTRAL_200 = professional_colors.NEUTRAL_200
    NEUTRAL_100 = professional_colors.NEUTRAL_100
    NEUTRAL_50 = professional_colors.NEUTRAL_50

    # Chart colors
    CHART_COLORS = professional_colors.CHART_COLORS_PRIMARY
    BLUE_SCALE = professional_colors.CHART_COLORS_SEQUENTIAL
    DIVERGING_SCALE = professional_colors.CHART_COLORS_DIVERGING


# Export convenience functions
def create_section_header(title: str, **kwargs) -> None:
    """Create a section header."""
    ui.section_header(title, **kwargs)


def create_metric_row(metrics: Dict[str, Any], **kwargs) -> None:
    """Create a row of metrics."""
    ui.metric_row(metrics, **kwargs)


def create_data_table(df: pd.DataFrame, **kwargs) -> None:
    """Create a data table."""
    ui.data_table(df, **kwargs)


def create_chart_container(chart_func: Callable, **kwargs) -> None:
    """Create a chart container."""
    ui.chart_container(chart_func, **kwargs)


def create_filter_sidebar(
    filters: List[Dict[str, Any]], **kwargs
) -> Dict[str, Any]:
    """Create a filter sidebar."""
    return ui.filter_sidebar(filters, **kwargs)


def create_insight_container(title: str, content: str, **kwargs) -> None:
    """Create an insight container."""
    ui.insight_container(title, content, **kwargs)


def styled_metric(label: str, value: Any, **kwargs) -> None:
    """Display a styled metric."""
    ui.styled_metric(label, value, **kwargs)


__all__ = [
    "UIComponentFactory",
    "ComponentConfig",
    "ui",
    "Colors",
    "create_section_header",
    "create_metric_row",
    "create_data_table",
    "create_chart_container",
    "create_filter_sidebar",
    "create_insight_container",
    "styled_metric",
    # Export formatting functions
    "fmt_int",
    "fmt_float",
    "fmt_pct",
    "fmt_change",
    "fmt_currency",
    "fmt_number",
    "fmt_duration",
    "fmt_ratio",
]
