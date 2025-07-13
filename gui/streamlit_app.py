"""
Module: streamlit_app.py
Author: Mohamed Tolba
Date Created: 13-07-2025
Last Updated: 13-07-2025

Description:
    Streamlit-based graphical user interface (GUI) for the educational video analysis project.
    Allows users to update the local GitHub repository, view and run available Python scripts,
    and display their outputs interactively. Designed to support the workflow of metadata 
    processing, retention analysis, and machine learning integration.
"""


# Streamlit is a tool that lets you turn your Python scripts into interactive web apps — easily, quickly, and with very little code.
# streamlit run streamlit_app.py

import streamlit as st
import os
import subprocess

# App title
st.title("🎓 EduVideo Insights: GUI for Video Metadata and Retention Analysis")

# Introductory markdown statement
st.markdown("""
Welcome to the **EduVideo Insights** tool — an interactive Streamlit app designed to support the
open-source educational video analysis project.

With this app, you can:
- Pull updates from the linked GitHub repository 📁
- Run Python scripts for metadata processing, retention extraction, or analysis 🧠
- View outputs directly in your browser 🌐

This GUI helps educators and researchers investigate how video design impacts student engagement and retention.
""")

# py_files = [f for f in os.listdir('.') if f.endswith('.py') and f != 'app.py']
# selected = st.selectbox("Choose a script to run:", py_files)
# 
# if st.button("Run it!"):
#     result = subprocess.run(['python3', selected], capture_output=True, text=True)
#     st.text_area("Output", result.stdout + "\n" + result.stderr)
# 
# 
# if st.button("Update Repo from GitHub"):
#     os.system("git pull")
#     st.success("Repository updated!")
# 


