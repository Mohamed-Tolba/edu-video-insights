# sidebar.py
import streamlit as st

def add_sidebar(parent_dir):
    st.sidebar.title("🎓 EduVideo Insights")
    st.sidebar.markdown("Use the links below to navigate through the app. <br> ________________________________________", unsafe_allow_html=True)
    
    home_path = parent_dir + '/' + 'gui/Home.py'
    st.sidebar.page_link(home_path, label="Home")

    page1_path =  parent_dir + '/' + 'gui/pages/page1.py'
    st.sidebar.page_link(page1_path, label = "Stage 1: Planning and Video Selection")

    page1_path =  parent_dir + '/' + 'gui/pages/page2.py'
    st.sidebar.page_link(page1_path, label="Stage 2: Data Collection")

    page1_path =  parent_dir + '/' + 'gui/pages/page3.py'
    st.sidebar.page_link(page1_path, label="Stage 3: Dataset Construction")

    page1_path =  parent_dir + '/' + 'gui/pages/page4.py'
    st.sidebar.page_link(page1_path, label="Stage 4: Exploratory Data Analysis (EDA)")

    page1_path =  parent_dir + '/' + 'gui/pages/page5.py'
    st.sidebar.page_link(page1_path, label="Stage 5: Model Training and Validation")

    page1_path =  parent_dir + '/' + 'gui/pages/page6.py'
    st.sidebar.page_link(page1_path, label="Stage 6: Insight Extraction")

    page1_path =  parent_dir + '/' + 'gui/pages/page7.py'
    st.sidebar.page_link(page1_path, label="Stage 7: Educaional Design Feedback")

    page1_path =  parent_dir + '/' + 'gui/pages/page8.py'
    st.sidebar.page_link(page1_path, label="Reporting and Contributing")

