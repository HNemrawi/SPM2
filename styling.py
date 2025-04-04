"""
Custom CSS & Styling
--------------------
Contains custom dark CSS, styling helpers for cards, metrics, etc.
"""

import streamlit as st

CUSTOM_CSS = """
<style>
html, body, [class*="css"] {
    font-family: 'Open Sans', sans-serif;
    background-color: #2E2E2E;
    color: #FFFFFF;
}
.block-container {
    padding: 1rem 2rem !important;
    background-color: #2E2E2E;
}
h1, h2, h3, h4 {
    color: #FFFFFF;
}
.dataframe thead {
    background-color: #444444 !important;
    color: #FFFFFF !important;
}
.dataframe tbody tr:nth-child(even) {
    background-color: #3E3E3E !important;
}
.dataframe tbody tr:nth-child(odd) {
    background-color: #2E2E2E !important;
}
div[data-testid="stMetric"],
div[data-testid="metric-container"] {
    background-color: #2E2E2E;
    border: 1px solid #444444;
    padding: 1rem;
    border-radius: 8px;
    border-left: 0.5rem solid #1f77b4 !important;
}
</style>
"""

def apply_custom_css():
    """
    Inject the custom CSS into the Streamlit app for consistent dark styling.
    """
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def style_metric_cards(
    background_color: str = "#2E2E2E",
    border_size_px: int = 1,
    border_color: str = "#444444",
    border_radius_px: int = 5,
    border_left_color: str = "#1f77b4",
    box_shadow: bool = True
) -> None:
    """
    Apply additional styling for metric cards with customizable options.
    
    Parameters:
        background_color (str): Background color of the cards.
        border_size_px (int): Border width in pixels.
        border_color (str): Border color.
        border_radius_px (int): Border radius in pixels.
        border_left_color (str): Left border accent color.
        box_shadow (bool): Whether to apply a shadow effect.
    """
    box_shadow_str = (
        "box-shadow: 0 0.15rem 1.75rem 0 rgba(0, 0, 0, 0.3) !important;"
        if box_shadow else "box-shadow: none !important;"
    )
    st.markdown(f"""
        <style>
            div[data-testid="stMetric"],
            div[data-testid="metric-container"] {{
                background-color: {background_color};
                border: {border_size_px}px solid {border_color};
                padding: 5% 5% 5% 10%;
                border-radius: {border_radius_px}px;
                border-left: 0.5rem solid {border_left_color} !important;
                {box_shadow_str}
            }}
        </style>
        """, unsafe_allow_html=True)
