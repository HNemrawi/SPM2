"""
Plotly and Streamlit theme configuration
"""

import plotly.graph_objects as go
import plotly.io as pio

def setup_plotly_theme():
    """
    Configure a custom dark theme for Plotly charts.
    """
    custom_theme = go.layout.Template(
        layout=go.Layout(
            font=dict(
                family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif", 
                size=12, 
                color="rgba(0, 0, 0, 0.87)"  # High contrast text that works in both modes
            ),
            paper_bgcolor="rgba(0, 0, 0, 0)",  # Transparent background
            plot_bgcolor="rgba(0, 0, 0, 0.02)",  # Very subtle background
            colorway=[
                "#0066CC",  # Professional blue
                "#059862",  # Success green
                "#D97706",  # Warning amber
                "#DC2626",  # Danger red
                "#7C3AED",  # Purple
                "#0891B2",  # Cyan
                "#EC4899",  # Pink
                "#6366F1",  # Indigo
                "#84CC16",  # Lime
                "#F59E0B",  # Orange
            ],
            xaxis=dict(
                gridcolor="rgba(0, 0, 0, 0.1)",  # Subtle grid lines
                zerolinecolor="rgba(0, 0, 0, 0.2)",
                linecolor="rgba(0, 0, 0, 0.2)",
                tickfont=dict(color="rgba(0, 0, 0, 0.7)")
            ),
            yaxis=dict(
                gridcolor="rgba(0, 0, 0, 0.1)",  # Subtle grid lines
                zerolinecolor="rgba(0, 0, 0, 0.2)",
                linecolor="rgba(0, 0, 0, 0.2)",
                tickfont=dict(color="rgba(0, 0, 0, 0.7)")
            ),
            title=dict(
                font=dict(
                    color="rgba(0, 0, 0, 0.87)", 
                    size=16,
                    weight=600
                ),
                x=0.5,
                xanchor="center"
            ),
            legend=dict(
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="rgba(0, 0, 0, 0.15)",
                borderwidth=1,
                font=dict(color="rgba(0, 0, 0, 0.7)")
            ),
            hoverlabel=dict(
                bgcolor="rgba(255, 255, 255, 0.95)",
                bordercolor="rgba(0, 0, 0, 0.1)",
                font=dict(color="rgba(0, 0, 0, 0.87)")
            ),
            # Additional theme settings for better adaptation
            hovermode="closest",
            margin=dict(l=60, r=30, t=60, b=60),
            showlegend=True,
            autosize=True
        )
    )
    pio.templates["custom_neutral"] = custom_theme
    pio.templates.default = "custom_neutral"

def setup_pandas_options():
    """
    Configure pandas display options.
    """
    import pandas as pd
    pd.set_option("styler.render.max_elements", 1_000_000)