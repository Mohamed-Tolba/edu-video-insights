"""
Module: streamlit_app.py
Author: Mohamed Tolba
Date Created: 13-07-2025
Last Updated: 15-07-2025

Description:
    Streamlit-based graphical user interface (GUI) for the educational video analysis project.
    Allows users to update the local GitHub repository, view and run available Python scripts,
    and display their outputs interactively. Designed to support the workflow of metadata 
    processing, retention analysis, and machine learning integration.
"""
# Streamlit is a tool that lets you turn your Python scripts into interactive web apps — easily, quickly, and with very little code.
# streamlit run streamlit_app.py

import streamlit as st
import sys
import os

# Build the absolute path to the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add to Python module search path
sys.path.append(parent_dir)

# Load core modules
from core.keys_manager import load_api_key  # Import the function to load the API key
from core.metadata_core import MetadataExtractor  # Import the MetadataExtractor class from the core module

# Load tab modules
from tabs.tab1 import render_tab1  # Import the function to render the first tab
from tabs.tab2 import render_tab2  # Import the function to render the second tab

API_KEY = load_api_key(parent_dir + '/' + "keys/youtube_data_API_key.txt")  # Load the YouTube Data API key from the specified file
MetadataExtractor_obj = MetadataExtractor(API_KEY)

# App title
st.title("🎓 EduVideo Insights: GUI for Video Metadata and Retention Analysis")

# Introductory markdown statement
st.markdown("""
Welcome to the **EduVideo Insights** tool — an interactive Streamlit app designed to support the
open-source educational video analysis project.

With this app, you can:
- Upload your video data 📁
- Run Python scripts for metadata processing, retention extraction, or analysis 🧠
- View outputs directly in your browser 🌐

This GUI helps educators and researchers investigate how video design impacts student engagement and retention.
""")

# Create four tabs
if 'tab_index' not in st.session_state:
    st.session_state.tab_index = 0

tabs = st.tabs(["📤 Upload user data", "🛠️ Extract video metadata", "🔍 Extract video characteristics", "📊 ML insights"])
tab1, tab2, tab3, tab4 = tabs
current_tab = tabs[st.session_state.tab_index]

## ---- Tab 1: Upload CSV and Prepare the submission file ----
with tab1:
    st.title("📤 Upload User Data")
    st.header("Uploading and Preparing New Video Data")
    render_tab1(MetadataExtractor_obj, parent_dir)  # Render the first tab for uploading CSV files and preparing submission data

# ---- Tab 2: Extract Videos Metadata and Retention Metrics ----
with tab2:
    st.header("🛠️ Extract Video Metadata")
    render_tab2(parent_dir)  # Render the second tab for extracting video metadata

# ---- Tab 3: ??? ----
with tab3:
    st.header("🔍 Extract Video Characteristics")
    st.info("Coming soon: Select and execute scripts here.")

# ---- Tab 4: ??? ----
with tab4:
    st.header("📊 Machine Learning Insights")
    st.info("Coming soon: Visualise ML model results here.")