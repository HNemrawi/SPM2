"""
HTML Factory Module
==================
Centralized HTML template generation for consistent UI components.
Eliminates duplication and provides a single source of truth for HTML generation.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from src.ui.themes.theme import theme


@dataclass
class HTMLConfig:
    """Configuration for HTML generation."""

    use_theme_colors: bool = True
    include_transitions: bool = True
    responsive: bool = True


class HTMLFactory:
    """Factory class for generating consistent HTML components."""

    def __init__(self, config: Optional[HTMLConfig] = None):
        self.config = config or HTMLConfig()
        self.theme = theme

    # ============== CONTAINER COMPONENTS ==============

    def container(
        self,
        content: str,
        class_name: Optional[str] = None,
        style: Optional[Dict[str, str]] = None,
        id: Optional[str] = None,
    ) -> str:
        """Create a generic container div."""
        style_str = self._dict_to_style(style) if style else ""
        class_attr = f'class="{class_name}"' if class_name else ""
        id_attr = f'id="{id}"' if id else ""

        return (
            f'<div {class_attr} {id_attr} style="{style_str}">{content}</div>'
        )

    def card(
        self,
        content: str,
        title: Optional[str] = None,
        icon: Optional[str] = None,
        border_color: Optional[str] = None,
        padding: str = "1.5rem",
        margin: str = "1rem 0",
    ) -> str:
        """Create a card component with optional title and icon."""
        border_color = border_color or self.theme.colors.primary

        title_html = ""
        if title:
            icon_html = (
                f'<span style="margin-right: 0.5rem;">{icon}</span>'
                if icon
                else ""
            )
            title_html = f"""
            <h4 style="
                color: {self.theme.colors.text_primary};
                margin: 0 0 1rem 0;
                font-size: 1.1rem;
                font-weight: 600;
                display: flex;
                align-items: center;
            ">
                {icon_html}{title}
            </h4>
            """

        return f"""
        <div style="
            background: {self.theme.colors.surface};
            border: 1px solid {
            self.theme.colors.border};
            border-left: 3px solid {border_color};
            border-radius: {self.theme.borders.radius_md};
            padding: {padding};
            margin: {margin};
            box-shadow: {self.theme.shadows.sm};
            {self._transition_style() if self.config.include_transitions else ''}
        ">
            {title_html}
            {content}
        </div>
        """

    def info_box(
        self,
        content: str,
        type: str = "info",
        title: Optional[str] = None,
        icon: Optional[str] = None,
        dismissible: bool = False,
    ) -> str:
        """Create an info/alert box with semantic styling.

        Parameters:
            content: The content to display in the box
            type: Box type (info, success, warning, danger)
            title: Optional title text
            icon: Optional icon - if not provided, uses default for type only if title is present
            dismissible: Whether to show a close button
        """
        type_styles = {
            "info": (
                self.theme.colors.info,
                self.theme.colors.info_bg_subtle,
            ),
            "success": (
                self.theme.colors.success,
                self.theme.colors.success_bg_subtle,
            ),
            "warning": (
                self.theme.colors.warning,
                self.theme.colors.warning_bg_subtle,
            ),
            "danger": (
                self.theme.colors.danger,
                self.theme.colors.danger_bg_subtle,
            ),
        }

        border_color, bg_color = type_styles.get(type, type_styles["info"])

        # Only use default icons if no icon is explicitly provided AND there's
        # a title
        if icon is None and title:
            default_icons = {
                "info": "ℹ️",
                "success": "✅",
                "warning": "⚠️",
                "danger": "❌",
            }
            icon = default_icons.get(type, "")

        # If icon is explicitly set to empty string, respect that
        display_icon = icon if icon is not None else ""

        dismiss_html = ""
        if dismissible:
            dismiss_html = """
            <button style="
                position: absolute;
                top: 0.5rem;
                right: 0.5rem;
                background: none;
                border: none;
                font-size: 1.2rem;
                cursor: pointer;
                opacity: 0.5;
            " onclick="this.parentElement.style.display='none'">×</button>
            """

        title_html = ""
        if title:
            # Only show icon if one exists
            icon_span = (
                f'<span style="font-size: 1.2rem;">{display_icon}</span>'
                if display_icon
                else ""
            )
            title_html = f"""
            <div style="
                font-weight: 600;
                font-size: 1.1rem;
                color: {self.theme.colors.text_primary};
                margin-bottom: 0.75rem;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            ">
                {icon_span}
                {title}
            </div>
            """
        elif display_icon:
            # Only add icon to content if no title and icon exists
            content = (
                f'<span style="margin-right: 0.5rem;">{display_icon}</span>'
                f"{content}"
            )

        return f"""
        <div style="
            position: relative;
            background: {bg_color};
            border: 1px solid {border_color};
            border-left: 3px solid {border_color};
            border-radius: {self.theme.borders.radius_md};
            padding: 0.875rem 1rem;
            margin: 0.75rem 0;
            color: {
            self.theme.colors.text_primary};
        ">
            {dismiss_html}
            {title_html}
            <div style="font-size: 0.875rem; line-height: 1.6; color: {self.theme.colors.text_secondary};">
                {content}
            </div>
        </div>
        """

    # ============== TYPOGRAPHY COMPONENTS ==============

    def title(
        self,
        text: str,
        level: int = 1,
        color: Optional[str] = None,
        margin: Optional[str] = None,
        icon: Optional[str] = None,
        highlight: bool = True,
    ) -> str:
        """Create a title with proper hierarchy (h1-h6) and optional background highlight."""
        level = max(1, min(6, level))  # Clamp between 1-6

        # Title styling based on level with subtle background highlights
        title_styles = {
            1: {
                "font-size": self.theme.typography.size_3xl,
                "font-weight": self.theme.typography.weight_bold,
                "margin": margin or "0 0 2rem 0",
                "background": self.theme.colors.primary_bg_subtle,
                "border_left": f"3px solid {self.theme.colors.primary}",
                "padding": "1.5rem 2rem",
                "border_radius": self.theme.borders.radius_lg,
            },
            2: {
                "font-size": self.theme.typography.size_2xl,
                "font-weight": self.theme.typography.weight_semibold,
                "margin": margin or "0 0 1.5rem 0",
                "background": self.theme.colors.primary_bg_subtle,
                "border_left": f"2px solid {self.theme.colors.primary_light}",
                "padding": "1.25rem 1.5rem",
                "border_radius": self.theme.borders.radius_md,
            },
            3: {
                "font-size": self.theme.typography.size_xl,
                "font-weight": self.theme.typography.weight_semibold,
                "margin": margin or "0 0 1rem 0",
                "background": self.theme.colors.background_secondary,
                "border_left": f"2px solid {self.theme.colors.accent}",
                "padding": "1rem 1.25rem",
                "border_radius": self.theme.borders.radius_md,
            },
            4: {
                "font-size": self.theme.typography.size_lg,
                "font-weight": self.theme.typography.weight_medium,
                "margin": margin or "0 0 0.75rem 0",
                "background": self.theme.colors.neutral_50,
                "border_left": f"2px solid {self.theme.colors.secondary}",
                "padding": "0.75rem 1rem",
                "border_radius": self.theme.borders.radius_sm,
            },
            5: {
                "font-size": self.theme.typography.size_base,
                "font-weight": self.theme.typography.weight_medium,
                "margin": margin or "0 0 0.5rem 0",
                "background": self.theme.colors.info_bg_subtle,
                "border_left": f"2px solid {self.theme.colors.info}",
                "padding": "0.5rem 0.75rem",
                "border_radius": self.theme.borders.radius_sm,
            },
            6: {
                "font-size": self.theme.typography.size_sm,
                "font-weight": self.theme.typography.weight_medium,
                "margin": margin or "0 0 0.5rem 0",
                "background": self.theme.colors.background_secondary,
                "border_left": f"1px solid {self.theme.colors.neutral_400}",
                "padding": "0.5rem 0.75rem",
                "border_radius": self.theme.borders.radius_sm,
            },
        }

        styles = title_styles[level]
        color = color or self.theme.colors.text_primary

        icon_html = (
            f'<span style="margin-right: 0.5rem; opacity: 0.8;">{icon}</span>'
            if icon
            else ""
        )

        # Build the style string
        base_styles = f"""
            color: {color};
            font-size: {styles['font-size']};
            font-weight: {styles['font-weight']};
            margin: {styles['margin']};
            line-height: {self.theme.typography.line_height_tight};
            display: flex;
            align-items: center;
        """

        if highlight:
            highlight_styles = f"""
                background: {styles['background']};
                border-left: {styles['border_left']};
                padding: {styles['padding']};
                border-radius: {styles['border_radius']};
                box-shadow: {self.theme.shadows.sm};
            """
            if self.config.include_transitions:
                highlight_styles += """
                    transition: all 0.2s ease;
                """
        else:
            highlight_styles = ""

        return f"""
        <h{level} style="{base_styles} {highlight_styles}">
            {icon_html}{text}
        </h{level}>
        """

    # ============== LAYOUT COMPONENTS ==============

    def columns(
        self, columns: List[str], ratios: Optional[List[int]] = None
    ) -> str:
        """Create a responsive column layout."""
        if ratios and len(ratios) == len(columns):
            total = sum(ratios)
            widths = [f"flex: {r}/{total};" for r in ratios]
        else:
            widths = ["flex: 1;" for _ in columns]

        column_html = ""
        for content, width in zip(columns, widths):
            column_html += f"""
            <div style="{width} padding: 0 0.5rem;">
                {content}
            </div>
            """

        return f"""
        <div style="display: flex; gap: 1rem; margin: 1rem 0;">
            {column_html}
        </div>
        """

    def divider(
        self,
        style: str = "solid",
        color: Optional[str] = None,
        margin: str = "2rem 0",
    ) -> str:
        """Create a styled divider."""
        if style == "gradient":
            color = color or self.theme.colors.primary
            return f"""
            <hr style="
                border: none;
                height: 2px;
                background: linear-gradient(to right, transparent, {color}, transparent);
                margin: {margin};
            "/>
            """
        elif style == "dots":
            return f"""
            <div style="
                text-align: center;
                margin: {margin};
                color: {self.theme.colors.text_muted};
                letter-spacing: 8px;
                font-size: 1.2rem;
            ">•••</div>
            """
        else:
            color = color or self.theme.colors.border
            return f"""
            <hr style="
                border: none;
                border-top: 1px solid {color};
                margin: {margin};
            "/>
            """

    # ============== DATA DISPLAY COMPONENTS ==============

    def metric_card(
        self,
        label: str,
        value: Union[str, int, float],
        delta: Optional[str] = None,
        delta_color: Optional[str] = None,
        icon: Optional[str] = None,
        color: Optional[str] = None,
    ) -> str:
        """Create a metric display card."""
        color = color or self.theme.colors.primary

        # Format value
        if isinstance(value, (int, float)):
            value = f"{value:,}" if isinstance(value, int) else f"{value:.1f}"

        delta_html = ""
        if delta:
            delta_color = delta_color or self.theme.colors.text_secondary
            delta_html = f"""
            <div style="
                font-size: {self.theme.typography.size_sm};
                color: {delta_color};
                margin-top: {self.theme.spacing.xs};
            ">
                {delta}
            </div>
            """

        icon_html = ""
        if icon:
            icon_html = f"""
            <div style="
                position: absolute;
                top: 1rem;
                right: 1rem;
                font-size: 1.5rem;
                opacity: 0.3;
            ">{icon}</div>
            """

        return f"""
        <div style="
            position: relative;
            background: {self.theme.colors.surface};
            border: 1px solid {
            self.theme.colors.border};
            border-left: 3px solid {color};
            border-radius: {self.theme.borders.radius_md};
            padding: {self.theme.spacing.lg};
            box-shadow: {self.theme.shadows.sm};
            {self._transition_style() if self.config.include_transitions else ''}
        ">
            {icon_html}
            <div style="
                font-size: {self.theme.typography.size_sm};
                color: {self.theme.colors.text_secondary};
                margin-bottom: {self.theme.spacing.xs};
            ">
                {label}
            </div>
            <div style="
                font-size: {self.theme.typography.size_2xl};
                font-weight: {self.theme.typography.weight_semibold};
                color: {self.theme.colors.text_primary};
            ">
                {value}
            </div>
            {delta_html}
        </div>
        """

    def progress_bar(
        self,
        value: float,
        max_value: float = 100,
        label: Optional[str] = None,
        color: Optional[str] = None,
        height: str = "20px",
        show_percentage: bool = True,
    ) -> str:
        """Create a progress bar."""
        percentage = (value / max_value * 100) if max_value else 0
        color = color or self.theme.colors.primary

        label_html = ""
        if label:
            label_html = f"""
            <div style="
                margin-bottom: {self.theme.spacing.sm};
                font-size: {self.theme.typography.size_sm};
                color: {self.theme.colors.text_secondary};
            ">{label}</div>
            """

        percentage_html = ""
        if show_percentage:
            percentage_html = f"""
            <span style="
                position: absolute;
                width: 100%;
                text-align: center;
                line-height: {height};
                color: {self.theme.colors.text_primary};
                font-size: {self.theme.typography.size_sm};
                font-weight: {self.theme.typography.weight_medium};
            ">{percentage:.1f}%</span>
            """

        return f"""
        <div style="margin: {self.theme.spacing.md} 0;">
            {label_html}
            <div style="
                position: relative;
                background: {self.theme.colors.border_light};
                border-radius: {self.theme.borders.radius_full};
                height: {height};
                overflow: hidden;
            ">
                {percentage_html}
                <div style="
                    background: {color};
                    height: 100%;
                    width: {percentage}%;
                    border-radius: {self.theme.borders.radius_full};
                    {self._transition_style(
            'width 0.3s ease') if self.config.include_transitions else ''}
                "></div>
            </div>
        </div>
        """

    def module_card(
        self,
        title: str,
        description: str,
        icon: str,
        color: Optional[str] = None,
        features: Optional[List[str]] = None,
    ) -> str:
        """Create a module card component."""
        color = color or self.theme.colors.primary

        features_html = ""
        if features:
            features_list = "".join(
                [
                    f"<li style='font-size: 0.85rem; color: {self.theme.colors.text_muted}; margin-bottom: 0.25rem;'>• {f}</li>"
                    for f in features
                ]
            )
            features_html = f"<ul style='margin: 1rem 0 0 0; padding: 0; list-style: none;'>{features_list}</ul>"

        hover_class = (
            "module-card-hover" if self.config.include_transitions else ""
        )
        hover_css = (
            f"""
        <style>
        .module-card-hover:hover {{
            transform: translateY(-2px);
            box-shadow: {self.theme.shadows.md};
        }}
        </style>
        """
            if self.config.include_transitions
            else ""
        )

        return f"""
        {hover_css}
        <div class='{hover_class}' style='
            border: 1px solid {self.theme.colors.border};
            border-left: 3px solid {color};
            border-radius: {self.theme.borders.radius_lg};
            padding: {self.theme.spacing.lg};
            margin: 0 0 {self.theme.spacing.md} 0;
            background: {self.theme.colors.surface};
            min-height: 200px;
            display: flex;
            flex-direction: column;
            box-shadow: {self.theme.shadows.sm};
            {self._transition_style('all 0.2s ease') if self.config.include_transitions else ''}
        '>
            <div style='display: flex; align-items: center; margin-bottom: {self.theme.spacing.sm};'>
                <span style='font-size: 1.5rem; margin-right: {self.theme.spacing.xs};'>{icon}</span>
                <h4 style='margin: 0; color: {self.theme.colors.text_primary}; font-size: {self.theme.typography.size_lg}; font-weight: {self.theme.typography.weight_semibold};'>{title}</h4>
            </div>
            <div style='flex: 1; display: flex; align-items: flex-start;'>
                <p style='margin: 0; color: {self.theme.colors.text_secondary}; line-height: 1.6;'>{description}</p>
            </div>
            {features_html}
        </div>
        """

    def data_status_card(
        self, filename: str, record_count: int, status: str = "active"
    ) -> str:
        """Create a data status card showing current dataset info."""
        status_color = (
            self.theme.colors.success
            if status == "active"
            else self.theme.colors.warning
        )

        return f"""
        <div style='
            background: {self.theme.colors.surface};
            border: 1px solid {status_color};
            border-radius: {self.theme.borders.radius_md};
            padding: {self.theme.spacing.sm};
            margin-bottom: {self.theme.spacing.xs};
        '>
            <div style='display: flex; align-items: center; justify-content: space-between;'>
                <div>
                    <div style='color: {status_color}
            ; font-size: {self.theme.typography.size_xs}; margin-bottom: 0.25rem;'>
                        ACTIVE DATASET
                    </div>
                    <div style='color: {
            self.theme.colors.text_primary}; font-weight: 500;'>
                        {filename}
                    </div>
                </div>
                <div style='text-align: right;'>
                    <div style='color: {self.theme.colors.text_muted}
            ; font-size: {self.theme.typography.size_xs};'>
                        {record_count:,} records
                    </div>
                </div>
            </div>
        </div>
        """

    def upload_area(
        self,
        title: str = "Drop your HMIS data file here or click to browse",
        subtitle: str = "Supports CSV format",
        icon: str = "📁",
    ) -> str:
        """Create an upload area component."""
        hover_class = "upload-hover" if self.config.include_transitions else ""
        hover_css = (
            f"""
        <style>
        .upload-hover:hover {{
            border-color: {self.theme.colors.primary} !important;
        }}
        </style>
        """
            if self.config.include_transitions
            else ""
        )

        return f"""
        {hover_css}
        <div class='{hover_class}' style='
            background: {self.theme.colors.background};
            border: 2px dashed {self.theme.colors.border};
            border-radius: {self.theme.borders.radius_lg};
            padding: {self.theme.spacing.lg};
            text-align: center;
            margin-bottom: {self.theme.spacing.md};
            {self._transition_style('border-color 0.2s ease') if self.config.include_transitions else ''}
        '>
            <div style='font-size: 2rem; margin-bottom: {self.theme.spacing.xs}; opacity: 0.7;'>{icon}</div>
            <div style='color: {self.theme.colors.text_muted}; font-size: {self.theme.typography.size_sm};'>
                {title}
            </div>
            <div style='color: {self.theme.colors.text_muted}; font-size: {self.theme.typography.size_xs}; margin-top: {self.theme.spacing.xs};'>
                {subtitle}
            </div>
        </div>
        """

    # ============== NAVIGATION COMPONENTS ==============

    def tabs(
        self,
        tabs: List[Dict[str, str]],
        active_tab: str,
        id_prefix: str = "tab",
    ) -> str:
        """Create a tab navigation component."""
        tab_nav = ""
        for tab in tabs:
            is_active = tab["id"] == active_tab
            active_style = (
                f"""
                background: {self.theme.colors.primary};
                color: white;
            """
                if is_active
                else f"""
                background: transparent;
                color: {self.theme.colors.text_secondary};
            """
            )

            tab_nav += f"""
            <button style="
                {active_style}
                border: none;
                padding: {self.theme.spacing.sm}
                {self.theme.spacing.lg};
                border-radius: {self.theme.borders.radius_sm};
                font-weight: {self.theme.typography.weight_medium};
                cursor: pointer;
                {self._transition_style() if self.config.include_transitions else ''}
            " id="{id_prefix}_{tab['id']}">
                {tab.get('icon', '')} {tab['label']}
            </button>
            """

        return f"""
        <div style="
            display: flex;
            gap: {self.theme.spacing.xs};
            background: {self.theme.colors.surface};
            padding: {self.theme.spacing.xs};
            border-radius: {self.theme.borders.radius_md};
            border: 1px solid {self.theme.colors.border};
            margin-bottom: {self.theme.spacing.md};
        ">
            {tab_nav}
        </div>
        """

    # ============== UTILITY METHODS ==============

    def _dict_to_style(self, style_dict: Dict[str, str]) -> str:
        """Convert a dictionary to CSS style string."""
        return "; ".join([f"{k}: {v}" for k, v in style_dict.items()])

    def _transition_style(self, transition: str = "all 0.2s ease") -> str:
        """Get transition CSS."""
        return f"transition: {transition};"

    def _hover_style(self, hover_css: str) -> str:
        """Generate hover CSS if transitions are enabled."""
        if self.config.include_transitions:
            return f":hover {{ {hover_css} }}"
        return ""

    def spacing(
        self,
        top: str = "0",
        right: str = "0",
        bottom: str = "0",
        left: str = "0",
    ) -> str:
        """Generate spacing style."""
        return f"margin: {top} {right} {bottom} {left};"

    def padding(
        self,
        top: str = "0",
        right: str = "0",
        bottom: str = "0",
        left: str = "0",
    ) -> str:
        """Generate padding style."""
        return f"padding: {top} {right} {bottom} {left};"

    # ==================== DASH-STYLE PRIMITIVES ====================

    def eyebrow(self, text: str) -> str:
        """Small uppercase label used above KPIs, filters, and section starts.

        Renders with the ``.eyebrow`` class defined in styles.py — muted
        color, letter-spaced, semibold. Designed to pair with a hero KPI
        or numeric callout below it.
        """
        return f'<div class="eyebrow">{text}</div>'

    def panel(
        self,
        content: str,
        *,
        lede: bool = False,
    ) -> str:
        """Wrap ``content`` in a DASH-style ``.panel`` card.

        Args:
            content: Inner HTML (already-rendered fragments OK).
            lede: When True, use the ``.panel--lede`` chapter-headlining
                variant with more generous padding.
        """
        klass = "panel panel--lede" if lede else "panel"
        return f'<div class="{klass}">{content}</div>'

    def kpi_card(
        self,
        label: str,
        value: Any,
        *,
        hero: bool = False,
        delta_pill: Optional[str] = None,
    ) -> str:
        """Render a labeled KPI card with tabular numerics.

        Args:
            label: Short uppercase-style label rendered as an eyebrow.
            value: The big number. Already-formatted string preferred.
            hero: Use the editorial hero variant (left-aligned, larger,
                gradient background) for the lead KPI of a section.
            delta_pill: Optional already-rendered ``.pill`` HTML (e.g.
                from :meth:`rate_pill`) shown beneath the value.
        """
        eyebrow_html = self.eyebrow(label)
        if hero:
            value_html = f'<div class="hero-kpi">{value}</div>'
            yoy_html = (
                f'<div class="yoy">{delta_pill}</div>' if delta_pill else ""
            )
            return (
                f'<div class="kpi-card kpi-card--hero">'
                f"{eyebrow_html}{value_html}{yoy_html}</div>"
            )
        value_html = f'<div class="num">{value}</div>'
        pill_html = (
            f'<div style="margin-top: 0.5rem;">{delta_pill}</div>'
            if delta_pill
            else ""
        )
        return (
            f'<div class="kpi-card">{eyebrow_html}{value_html}{pill_html}</div>'
        )

    def rate_pill(self, text: str, status: str = "mid") -> str:
        """Render a status pill chip.

        Args:
            text: Pill contents (usually a small number + percent or label).
            status: One of ``good`` / ``bad`` / ``mid`` / ``warn`` / ``info``.
        """
        valid = {"good", "bad", "mid", "warn", "info"}
        cls = status if status in valid else "mid"
        return f'<span class="pill pill--{cls}">{text}</span>'

    def three_things(self, items: List[Dict[str, str]]) -> str:
        """Render a three-column callout grid.

        Args:
            items: Iterable of three dicts with keys ``status`` (``good``/
                ``bad``/``mid``), ``icon`` (emoji or character), ``title``
                (heading), and ``body`` (short paragraph).
        """
        cells: List[str] = []
        for item in items:
            status = item.get("status", "mid")
            icon = item.get("icon", "")
            title = item.get("title", "")
            body = item.get("body", "")
            cells.append(
                f'<div><div class="three-things__icon three-things__icon--{status}">'
                f"{icon}</div>"
                f'<div style="font-weight: 600; color: var(--text-primary); '
                f'margin-bottom: 0.25rem;">{title}</div>'
                f'<div style="color: var(--text-secondary); font-size: 0.875rem; '
                f'line-height: 1.5;">{body}</div></div>'
            )
        return f'<div class="three-things">{"".join(cells)}</div>'

    def page_header(
        self,
        title: str,
        subtitle: Optional[str] = None,
        badge: Optional[str] = None,
    ) -> str:
        """Render the editorial page header used at the top of main.py.

        Replaces the inline-style block that previously lived in ``main.py``.
        Pulls all colors and gradients from the theme so this is the only
        place those values need to be tuned.
        """
        bg = (
            f"linear-gradient(180deg, {self.theme.colors.background_secondary} "
            f"0%, {self.theme.colors.background} 100%)"
        )
        subtitle_html = (
            f'<p style="color: {self.theme.colors.text_secondary}; '
            f'font-size: 1.125rem; margin: 0; font-weight: 400;">{subtitle}</p>'
            if subtitle
            else ""
        )
        badge_html = (
            f'<div style="margin-bottom: 0.5rem;">{self.rate_pill(badge, "info")}</div>'
            if badge
            else ""
        )
        return (
            f'<div style="text-align: center; padding: 2rem 0; '
            f'background: {bg}; '
            f"border-bottom: 1px solid {self.theme.colors.border}; "
            f'margin: -1rem -1rem 2rem -1rem;">'
            f"{badge_html}"
            f'<h1 style="color: {self.theme.colors.primary}; '
            f"font-size: clamp(1.75rem, 1.4rem + 1.5vw, 2.5rem); font-weight: 700; "
            f'margin-bottom: 0.5rem; letter-spacing: -0.025em;">{title}</h1>'
            f"{subtitle_html}"
            f"</div>"
        )

    def welcome_banner(
        self,
        title: str,
        subtitle: str,
        icon: Optional[str] = None,
    ) -> str:
        """Render the welcome card shown when no data is loaded.

        Replaces the inline-style block previously in ``main.render_welcome_screen``.
        """
        icon_html = (
            f'<div style="font-size: clamp(3rem, 2rem + 3vw, 4.5rem); '
            f'margin-bottom: 1.5rem; filter: drop-shadow(0 4px 6px rgba(0, 0, 0, 0.1));">'
            f"{icon}</div>"
            if icon
            else ""
        )
        return (
            f'<div style="text-align: center; padding: 4rem 3rem; '
            f"background: linear-gradient(135deg, {self.theme.colors.primary_bg} 0%, "
            f"{self.theme.colors.background} 50%, "
            f"{self.theme.colors.background_secondary} 100%); "
            f"border: 1px solid {self.theme.colors.border}; "
            f"border-radius: 1.25rem; margin: 3rem auto; max-width: 700px; "
            f"box-shadow: {self.theme.shadows.lg}; position: relative; overflow: hidden;\">"
            f"{icon_html}"
            f'<h2 class="heading-fluid-2" style="color: {self.theme.colors.text_primary}; '
            f'margin-bottom: 1rem; font-weight: 700; letter-spacing: -0.025em;">{title}</h2>'
            f'<p class="lede" style="margin: 0 auto 2.5rem auto; max-width: 500px;">'
            f"{subtitle}</p>"
            f"</div>"
        )


# Create global instance
html_factory = HTMLFactory()


# Export convenience functions
def create_card(content: str, **kwargs) -> str:
    """Create a card component."""
    return html_factory.card(content, **kwargs)


def create_info_box(content: str, **kwargs) -> str:
    """Create an info box."""
    return html_factory.info_box(content, **kwargs)


def create_metric_card(label: str, value: Any, **kwargs) -> str:
    """Create a metric card."""
    return html_factory.metric_card(label, value, **kwargs)


def create_divider(style: str = "solid", **kwargs) -> str:
    """Create a divider."""
    return html_factory.divider(style, **kwargs)


def create_progress_bar(value: float, **kwargs) -> str:
    """Create a progress bar."""
    return html_factory.progress_bar(value, **kwargs)


def create_title(text: str, level: int = 1, **kwargs) -> str:
    """Create a title with hierarchy and optional background highlight."""
    return html_factory.title(text, level, **kwargs)


__all__ = [
    "HTMLFactory",
    "HTMLConfig",
    "html_factory",
    "create_card",
    "create_info_box",
    "create_metric_card",
    "create_divider",
    "create_progress_bar",
    "create_title",
]
