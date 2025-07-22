"""
Module: streamlit_app.py
Author: Mohamed Tolba
Date Created: 13-07-2025
Last Updated: 21-07-2025

Description:
    Streamlit-based graphical user interface (GUI) for the educational video analysis project.
    Allows users to update the local GitHub repository, view and run available Python scripts,
    and display their outputs interactively. Designed to support the workflow of metadata 
    processing, retention analysis, and machine learning integration.

To do:
    - Consider allowing the user to upload their submission file.
    Consider giving the user the option to select which metadata to extract.
    - Consider giving the user the option to select which metrics to extract/use.
    - Consider giving the user the option to select which dataset to use for analysis.
"""
# Streamlit is a tool that lets you turn your Python scripts into interactive web apps — easily, quickly, and with very little code.
# source /home/mohamedtolba/Documents/edu-video-insights/.venv/bin/activate
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
from gui.tabs.tab1_1 import render_tab1_1  # Import the function to render the first tab
from gui.tabs.tab1_2 import render_tab1_2  # Import the function to render the second tab

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

# Creating a user ID:
id_file_path = parent_dir + '/' + 'temp/user_id.txt'
user_id_string = ''
# Read from file
with open(id_file_path, 'r') as f:
    user_id_string = f.read()
user_id = user_id_string

# Initialize stat
if 'tab1' not in st.session_state:
    st.session_state['tab1'] = 0

# Initialize stat
if 'button1_1' not in st.session_state:
    st.session_state['button1_1'] = 0

# Initialize stat
if 'tab1_1' not in st.session_state:
    st.session_state['tab1_1'] = 0

# tab1, tab2, tab3, tab4, tab5 = st.tabs(["Youtube", "Dailymotion", "Panopto", "Echo 360", "Kaltura"])
st.subheader("Select the video-hosting platform from the tabs below:")
tab1, tab2, tab3, tab4 = st.tabs(["Youtube", "Dailymotion", "Panopto", "Echo 360"])
# tabs = st.tabs(["📤 Import user data", "🛠️ Extract video metadata", "🔍 Extract video characteristics", "📊 ML insights"])
# tab1, tab2, tab3, tab4 = tabs
# current_tab = tabs[st.session_state.tab_index]

## ---- Tab 1: Upload CSV and Prepare the submission file ----
with tab1:
    tab1_1, tab1_2, tab1_3, tab1_4 = st.tabs(["📤 Import user data", "🛠️ Extract video metadata", "🔍 Extract video characteristics", "📊 ML insights"])
    with tab1_1:
        if st.button("Upload New Data"):
            st.session_state['button1_1'] = 1
            user_id_int = int(user_id_string)
            user_id_int = user_id_int + 1 
            user_id_string = f"{user_id_int:04d}"  # Output: '000012'
            # Write to file
            with open(id_file_path, 'w') as f:
                f.write(user_id_string)
            user_id = user_id_string

        if st.session_state['button1_1']==1:
            render_tab1_1(MetadataExtractor_obj, parent_dir, user_id)  # Render the first tab for uploading CSV files and preparing submission data
    
    with tab1_2:
        render_tab1_2(parent_dir, API_KEY, user_id)  # Render the first tab for uploading CSV files and preparing submission data
    # st.header("What do you want to do today?")
    # subtab = st.radio('Video-hosting Platform', ['Youtube', 'Other'])
    
    # if st.session_state.step == 1:
    #     if st.button("Youtube"):
    #         st.session_state.step = 2
# 
    # elif st.session_state.step == 2:
    #     if st.button("Upload new data"):
    #         st.session_state.step = 3
    #         user_id_int = int(user_id_string)
    #         user_id_int = user_id_int + 1 
    #         user_id_string = f"{user_id_int:06d}"  # Output: '000012'
    #         # Write to file
    #         with open(id_file_path, 'w') as f:
    #             f.write(user_id_string)
    #         user_id = user_id_string
    # 
    # elif st.session_state.step == 3:
    #     if st.button("Print Hello"):
               # st.info("Hello")
        

# ---- Tab 2: Extract Videos Metadata and Retention Metrics ----
with tab2:
    st.info("🚧 Under Construction 🚧")

# ---- Tab 3: ??? ----
with tab3:
    st.info("🚧 Under Construction 🚧")

# ---- Tab 4: ??? ----
with tab4:
    st.info("🚧 Under Construction 🚧")
