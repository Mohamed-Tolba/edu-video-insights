"""
Module: streamlit_app.py
Author: Mohamed Tolba
Date Created: 13-07-2025
Last Updated: 30-07-2025

Description:
    Streamlit-based graphical user interface (GUI) for the educational video analysis project.
    Allows users to update the local GitHub repository, view and run available Python scripts,
    and display their outputs interactively. Designed to support the workflow model for analysing educational videos.

To do:
    - Consider allowing the user to upload their submission file.
      Consider giving the user the option to select which metadata to extract.
    - Consider giving the user the option to select which metrics to extract/use.
    - Consider giving the user the option to select which dataset to use for analysis.
    - The app does not push any new data to github?!
"""
# Streamlit is a tool that lets you turn your Python scripts into interactive web apps — easily, quickly, and with very little code.
# streamlit run Home.py
# https://edu-video-insights.streamlit.app/
# An app with a UI design that I like: https://dezoomcamp.streamlit.app/
# An app for streamlit cheatsheet: https://docs.streamlit.io/develop/quick-reference/cheat-sheet

## Import necessary libraries & modules -----------------------------------------------------------------------
import streamlit as st
import sys, os

# --------------------------------------------------------------------------------------------------------
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # Build the absolute path to the parent directory
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
    page_title="EduVideo Insights",
    page_icon=":book:",
    layout="centered",  # Use 'wide' layout for more space 
    initial_sidebar_state="expanded"  # Start with the sidebar expanded
)
# Page title
st.title("🎓 EduVideo Insights") 

st.markdown("""<br>""", unsafe_allow_html=True)  # Add a line break for spacing

st.markdown("""            
<p style="font-family:sans serif; font-size:25px; color:#FFFFFF;">
👨‍🔧 Educational Video Analysis Tool by <a href="https://www.monash.edu/engineering/mohamedtolba" style="color:#FFFFFF;">Mohamed Tolba</a>.<br>
</p>""", unsafe_allow_html=True)

st.markdown("""            
<p style="font-family:sans serif; font-size:20px; color:#FFFFFF;">
Welcome to the EduVideo Insights tool — an interactive Streamlit app designed to support the open-source educational video analysis project.</p>""", unsafe_allow_html=True)

cover_image_path = parent_dir + '/' + 'gui/cover_image.png'  # Path to the cover image 
st.image(cover_image_path) # caption="Optional caption", use_container_width = True

st.info("""Original Project Repository on [Github](https://github.com/Mohamed-Tolba/edu-video-insights.git)""")

st.markdown("""            
<p style="font-family:sans serif; font-size:20px; color:#FFFFFF;">
This application helps educators and researchers explore the impact of different educational video characteristics on student retention and engagement.
</p>""", unsafe_allow_html=True)

welcome_message = """
<div style="font-family:sans serif; font-size:20px; color:#FFFFFF;">
With this app, you can apply the following seven-staged workflow model to your video data: <br>
<li> Stage 1: Planning and Video Selection 📹 </li>
<li> Stage 2: Data Collection 📊  </li>
<li> Stage 3: Dataset Construction 🏗   </li>
<li> Stage 4: Exploratory Data Analysis (EDA) 🔍 </li>
<li> Stage 5: Model Training and Validation 🤖 </li>
<li> Stage 6: Insight Extraction 📈 </li>
<li> Stage 7: Educaional Design Feedback 💡 </li>
<li> Reporting and Contributing 📑 </li>
</div>
"""
st.markdown(welcome_message, unsafe_allow_html=True)
# -------------------------------------------------------------------------------------------------------- 

