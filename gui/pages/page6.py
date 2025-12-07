import streamlit as st
import sys, os

# --------------------------------------------------------------------------------------------------------
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) # Build the absolute path to the parent directory
sys.path.append(parent_dir) # Add to Python module search path
# --------------------------------------------------------------------------------------------------------
from gui.utilities import load_css  # Import a function to load custom CSS
load_css(parent_dir)
# --------------------------------------------------------------------------------------------------------
from gui.sidebar import add_sidebar  # Import the function to add the sidebar
add_sidebar(parent_dir)
# --------------------------------------------------------------------------------------------------------
# Set the page configuration for the Streamlit app -------------------------------------------------------
st.set_page_config(
    page_title="Insight Extraction",
    page_icon=":book:",
    # layout="wide",  # Use 'wide' layout for more space
    initial_sidebar_state="expanded"  # Start with the sidebar expanded
)
# Page title
st.header("📈 Insight Extraction")
st.info("This page is under development. Stay tuned for updates!")
# st.header
# --------------------------------------------------------------------------------------------------------