"""
Setup & Configuration
---------------------
Handles initial page config, Plotly theme setup, and any global configurations.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

def setup_page():
    """
    Configure the Streamlit page and set up a custom dark Plotly theme.
    """
    st.set_page_config(
        page_title="Homelessness Analysis Suite",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    # Define and set a custom dark theme for Plotly charts.
    custom_theme = go.layout.Template(
        layout=go.Layout(
            font=dict(family="Open Sans", size=12, color="#FFFFFF"),
            paper_bgcolor="#2E2E2E",
            plot_bgcolor="#2E2E2E",
            colorway=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
            xaxis=dict(gridcolor="#444444"),
            yaxis=dict(gridcolor="#444444"),
            title=dict(font=dict(color="#FFFFFF", size=16))
        )
    )
    pio.templates["custom_dark"] = custom_theme
    pio.templates.default = "custom_dark"
    pd.set_option("styler.render.max_elements", 1_000_000)
