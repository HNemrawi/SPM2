"""
Unified Theme Configuration
============================
Single source of truth for all theme-related settings including colors, typography,
spacing, borders, and chart configurations.
"""

from dataclasses import dataclass, field


@dataclass
class ThemeColors:
    """Unified color palette for the entire application."""

    # Primary brand colors
    primary = "#00629b"  # Deep professional blue
    primary_dark = "#004d7a"  # Darker variant
    primary_light = "#4d94c1"  # Light blue
    primary_hover = "#0074b3"  # Hover state
    primary_bg = "#e6f2f8"  # Lightest blue background
    primary_bg_subtle = "rgba(0, 98, 155, 0.04)"  # Very subtle blue

    # Secondary colors
    secondary = "#7C3AED"  # Purple
    secondary_light = "#A78BFA"  # Light Purple
    secondary_dark = "#5B21B6"  # Dark Purple

    # Accent colors
    accent = "#F59E0B"  # Amber
    accent_light = "#FCD34D"  # Light Amber
    accent_dark = "#D97706"  # Dark Amber

    # Semantic colors for status and feedback
    success = "#10B981"  # Emerald green
    success_light = "#34D399"
    success_dark = "#059669"
    success_bg = "#ECFDF5"  # Softer green background
    success_bg_subtle = "rgba(16, 185, 129, 0.04)"  # Very subtle green

    warning = "#F59E0B"  # Orange/Amber
    warning_light = "#FCD34D"
    warning_dark = "#D97706"
    warning_bg = "#FFFBEB"  # Softer amber background
    warning_bg_subtle = "rgba(245, 158, 11, 0.04)"  # Very subtle amber

    danger = "#EF4444"  # Red
    danger_light = "#F87171"
    danger_dark = "#DC2626"
    danger_bg = "#FEF2F2"  # Softer red background
    danger_bg_subtle = "rgba(239, 68, 68, 0.04)"  # Very subtle red

    error = danger  # Alias for danger
    error_light = danger_light
    error_dark = danger_dark
    error_bg = "#FEF2F2"  # Softer red background
    error_bg_subtle = danger_bg_subtle

    info = "#3B82F6"  # Blue/Cyan
    info_light = "#60A5FA"
    info_dark = "#2563EB"
    info_bg = "#EFF6FF"  # Softer blue background
    info_bg_subtle = "rgba(59, 130, 246, 0.04)"  # Very subtle blue

    # Text hierarchy
    text_primary = "#1E293B"  # Dark slate
    text_secondary = "#475569"  # Medium slate
    text_muted = "#94A3B8"  # Light slate
    text_disabled = "#CBD5E1"  # Very light

    # Background layers
    background = "#FFFFFF"  # Main background
    background_secondary = "#F8FAFC"  # Secondary areas
    surface = "#F1F5F9"  # Card surfaces
    surface_hover = "#E2E8F0"  # Hover state

    # Border colors
    border = "#E2E8F0"  # Default border
    border_light = "#F1F5F9"  # Subtle borders
    border_dark = "#CBD5E1"  # Emphasized borders

    # Neutral scale
    neutral_50 = "#FAFAFA"
    neutral_100 = "#F4F4F5"
    neutral_200 = "#E4E4E7"
    neutral_300 = "#D4D4D8"
    neutral_400 = "#A1A1AA"
    neutral_500 = "#71717A"
    neutral_600 = "#52525B"
    neutral_700 = "#3F3F46"
    neutral_800 = "#27272A"
    neutral_900 = "#18181B"

    # Chart color sequences
    chart_colors = [
        "#00629b",  # Deep Blue (Primary)
        "#10B981",  # Emerald
        "#F59E0B",  # Amber
        "#EF4444",  # Red
        "#7C3AED",  # Purple
        "#06B6D4",  # Cyan
        "#EC4899",  # Pink
        "#8B5CF6",  # Violet
        "#64748B",  # Slate
        "#065F46",  # Dark Green
    ]

    chart_colors_categorical = [
        "#3B82F6",  # Blue
        "#10B981",  # Green
        "#F59E0B",  # Amber
        "#EF4444",  # Red
        "#8B5CF6",  # Purple
        "#14B8A6",  # Teal
        "#F97316",  # Orange
        "#EC4899",  # Pink
        "#6366F1",  # Indigo
        "#84CC16",  # Lime
    ]

    chart_colors_sequential = [
        "#e6f2f8",  # Lightest blue
        "#cce5f1",  # Blue 100
        "#b3d9ea",  # Blue 200
        "#99cce3",  # Blue 300
        "#80bfdc",  # Blue 400
        "#4d94c1",  # Blue 500
        "#337ba8",  # Blue 600
        "#1a628f",  # Blue 700
        "#00629b",  # Blue 800 (Primary)
        "#004d7a",  # Blue 900 (Dark)
    ]

    chart_colors_diverging = [
        "#DC2626",  # Red 700
        "#EF4444",  # Red 500
        "#F87171",  # Red 400
        "#FCA5A5",  # Red 300
        "#FECACA",  # Red 200
        "#F3F4F6",  # Gray 100 (neutral)
        "#DBEAFE",  # Blue 200
        "#BFDBFE",  # Blue 300
        "#93C5FD",  # Blue 400
        "#3B82F6",  # Blue 500
        "#1E40AF",  # Blue 800
    ]


@dataclass
class ThemeTypography:
    """Typography system for professional appearance."""

    # Font families
    font_family = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
    font_mono = "'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, monospace"

    # Font sizes (rem-based for responsiveness)
    size_xs = "0.75rem"  # 12px
    size_sm = "0.875rem"  # 14px
    size_base = "1rem"  # 16px
    size_lg = "1.125rem"  # 18px
    size_xl = "1.25rem"  # 20px
    size_2xl = "1.5rem"  # 24px
    size_3xl = "1.875rem"  # 30px
    size_4xl = "2.25rem"  # 36px
    size_5xl = "3rem"  # 48px

    # Font weights
    weight_normal = "400"
    weight_medium = "500"
    weight_semibold = "600"
    weight_bold = "700"
    weight_extrabold = "800"

    # Line heights
    line_height_tight = "1.25"
    line_height_normal = "1.5"
    line_height_relaxed = "1.625"
    line_height_loose = "2"

    # Letter spacing
    letter_spacing_tighter = "-0.05em"
    letter_spacing_tight = "-0.025em"
    letter_spacing_normal = "0"
    letter_spacing_wide = "0.025em"
    letter_spacing_wider = "0.05em"
    letter_spacing_widest = "0.1em"


@dataclass
class ThemeSpacing:
    """Spacing system using consistent scale."""

    # Base spacing unit (4px)
    unit = 4

    # Spacing scale
    xs = "0.25rem"  # 4px
    sm = "0.5rem"  # 8px
    md = "1rem"  # 16px
    lg = "1.5rem"  # 24px
    xl = "2rem"  # 32px
    xxl = "3rem"  # 48px
    xxxl = "4rem"  # 64px

    # Common spacing patterns
    card_padding = "1.5rem"  # 24px
    section_gap = "2rem"  # 32px
    container_padding = "1rem"  # 16px
    title_margin = "0.75rem"  # 12px
    element_gap = "1rem"  # 16px between elements


@dataclass
class ThemeBorders:
    """Border configuration for consistency."""

    # Border widths
    width_thin = "1px"
    width_default = "1px"
    width_medium = "2px"
    width_thick = "4px"

    # Border radii
    radius_none = "0"
    radius_sm = "0.25rem"  # 4px
    radius_md = "0.5rem"  # 8px
    radius_lg = "0.75rem"  # 12px
    radius_xl = "1rem"  # 16px
    radius_2xl = "1.5rem"  # 24px
    radius_full = "9999px"

    # Border styles
    style_solid = "solid"
    style_dashed = "dashed"
    style_dotted = "dotted"


@dataclass
class ThemeShadows:
    """Shadow definitions for depth and elevation."""

    # Shadow levels
    none = "none"
    sm = "0 1px 2px 0 rgba(0, 0, 0, 0.05)"
    default = "0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)"
    md = (
        "0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)"
    )
    lg = "0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)"
    xl = "0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)"
    xxl = "0 25px 50px -12px rgba(0, 0, 0, 0.25)"
    inner = "inset 0 2px 4px 0 rgba(0, 0, 0, 0.06)"

    # Special shadows
    card = "0 2px 8px rgba(0, 0, 0, 0.08), 0 1px 3px rgba(0, 0, 0, 0.05)"
    hover = "0 4px 12px rgba(0, 0, 0, 0.12)"


@dataclass
class ChartTheme:
    """Chart-specific theme settings."""

    # Plotly template
    template = "plotly_white"

    # Default layout settings
    layout = {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {
            "family": "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
            "size": 12,
            "color": "#475569",
        },
        "title": {
            "font": {
                "size": 18,
                "color": "#1E293B",
            }
        },
        "margin": {"l": 40, "r": 20, "t": 40, "b": 40},
        "hoverlabel": {
            "bgcolor": "white",
            "font_size": 12,
            "font_family": "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
        },
        "xaxis": {
            "gridcolor": "#E2E8F0",
            "linecolor": "#CBD5E1",
            "tickfont": {"color": "#475569"},
        },
        "yaxis": {
            "gridcolor": "#E2E8F0",
            "linecolor": "#CBD5E1",
            "tickfont": {"color": "#475569"},
        },
    }


@dataclass
class Theme:
    """Main theme configuration class."""

    colors: ThemeColors = field(default_factory=ThemeColors)
    typography: ThemeTypography = field(default_factory=ThemeTypography)
    spacing: ThemeSpacing = field(default_factory=ThemeSpacing)
    borders: ThemeBorders = field(default_factory=ThemeBorders)
    shadows: ThemeShadows = field(default_factory=ThemeShadows)
    chart: ChartTheme = field(default_factory=ChartTheme)

    def get_gradient(self, type: str = "primary") -> str:
        """Get CSS gradient string."""
        gradients = {
            "primary": f"linear-gradient(135deg, {self.colors.primary_light} 0%, {self.colors.primary} 100%)",
            "success": f"linear-gradient(135deg, {self.colors.success_light} 0%, {self.colors.success} 100%)",
            "warning": f"linear-gradient(135deg, {self.colors.warning_light} 0%, {self.colors.warning} 100%)",
            "danger": f"linear-gradient(135deg, {self.colors.danger_light} 0%, {self.colors.danger} 100%)",
            "info": f"linear-gradient(135deg, {self.colors.info_light} 0%, {self.colors.info} 100%)",
            "card": "linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(248,250,252,0.9) 100%)",
            "subtle": "linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 100%)",
        }
        return gradients.get(type, gradients["primary"])


# Create singleton instance
theme = Theme()


# ==================== BACKWARD COMPATIBILITY ====================
# Legacy color constants for modules still using old names

MAIN_COLOR = theme.colors.primary
SECONDARY_COLOR = theme.colors.warning
NEUTRAL_COLOR = theme.colors.neutral_500
SUCCESS_COLOR = theme.colors.success
WARNING_COLOR = theme.colors.warning
DANGER_COLOR = theme.colors.danger
INFO_COLOR = theme.colors.info

# Color sequences and scales
CUSTOM_COLOR_SEQUENCE = theme.colors.chart_colors
BLUE_SCALE = theme.colors.chart_colors_sequential
RED_SCALE = [
    "#FEE2E2",
    "#FECACA",
    "#FCA5A5",
    "#F87171",
    "#EF4444",
    "#DC2626",
    "#B91C1C",
    "#991B1B",
]
DIVERGING_SCALE = theme.colors.chart_colors_diverging
DISPARITY_COLOR_SCALE = ["#F4B183", "#D9D9D9", "#5B9BD5"]

# Plot template
PLOT_TEMPLATE = theme.chart.template


# Legacy professional_colors compatibility
class professional_colors:
    """Backward compatibility wrapper for professional_colors imports."""

    PRIMARY = theme.colors.primary
    PRIMARY_LIGHT = theme.colors.primary_light
    PRIMARY_DARK = theme.colors.primary_dark

    SECONDARY = theme.colors.secondary
    SECONDARY_LIGHT = theme.colors.secondary_light
    SECONDARY_DARK = theme.colors.secondary_dark

    SUCCESS = theme.colors.success
    SUCCESS_LIGHT = theme.colors.success_light
    SUCCESS_DARK = theme.colors.success_dark

    WARNING = theme.colors.warning
    WARNING_LIGHT = theme.colors.warning_light
    WARNING_DARK = theme.colors.warning_dark

    ERROR = theme.colors.error
    ERROR_LIGHT = theme.colors.error_light
    ERROR_DARK = theme.colors.error_dark

    INFO = theme.colors.info
    INFO_LIGHT = theme.colors.info_light
    INFO_DARK = theme.colors.info_dark

    # Neutral colors
    NEUTRAL_50 = theme.colors.neutral_50
    NEUTRAL_100 = theme.colors.neutral_100
    NEUTRAL_200 = theme.colors.neutral_200
    NEUTRAL_300 = theme.colors.neutral_300
    NEUTRAL_400 = theme.colors.neutral_400
    NEUTRAL_500 = theme.colors.neutral_500
    NEUTRAL_600 = theme.colors.neutral_600
    NEUTRAL_700 = theme.colors.neutral_700
    NEUTRAL_800 = theme.colors.neutral_800
    NEUTRAL_900 = theme.colors.neutral_900

    # Chart colors
    CHART_COLORS_PRIMARY = theme.colors.chart_colors
    CHART_COLORS_CATEGORICAL = theme.colors.chart_colors_categorical
    CHART_COLORS_SEQUENTIAL = theme.colors.chart_colors_sequential
    CHART_COLORS_DIVERGING = theme.colors.chart_colors_diverging

    # Gradients and shadows (from original professional_colors)
    GRADIENTS = {
        "primary": theme.get_gradient("primary"),
        "success": theme.get_gradient("success"),
        "warning": theme.get_gradient("warning"),
        "danger": theme.get_gradient("danger"),
        "info": theme.get_gradient("info"),
        "card": theme.get_gradient("card"),
        "subtle": theme.get_gradient("subtle"),
    }

    SHADOWS = {
        "sm": theme.shadows.sm,
        "default": theme.shadows.default,
        "md": theme.shadows.md,
        "lg": theme.shadows.lg,
        "xl": theme.shadows.xl,
        "card": theme.shadows.card,
        "hover": theme.shadows.hover,
    }


# Helper functions for backward compatibility
def blue_divider():
    """Create a blue gradient divider."""
    import streamlit as st

    from src.ui.factories.html import html_factory

    st.html(html_factory.divider(style="gradient", color=theme.colors.primary))


def apply_chart_style(fig, **kwargs):
    """Apply chart styling for backward compatibility."""
    from src.ui.factories.charts import chart_factory

    return chart_factory.apply_layout(fig, **kwargs)


# Export all
__all__ = [
    # Main theme instance
    "theme",
    "Theme",
    "ThemeColors",
    "ThemeTypography",
    "ThemeSpacing",
    "ThemeBorders",
    "ThemeShadows",
    "ChartTheme",
    # Backward compatibility
    "professional_colors",
    "MAIN_COLOR",
    "SECONDARY_COLOR",
    "NEUTRAL_COLOR",
    "SUCCESS_COLOR",
    "WARNING_COLOR",
    "DANGER_COLOR",
    "INFO_COLOR",
    "CUSTOM_COLOR_SEQUENCE",
    "BLUE_SCALE",
    "RED_SCALE",
    "DIVERGING_SCALE",
    "DISPARITY_COLOR_SCALE",
    "PLOT_TEMPLATE",
    "blue_divider",
    "apply_chart_style",
]
