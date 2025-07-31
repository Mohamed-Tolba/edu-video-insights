import streamlit as st
import sys, os
from utilities import load_css
load_css()
# --------------------------------------------------------------------------------------------------------
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) # Build the absolute path to the parent directory
sys.path.append(parent_dir) # Add to Python module search path

# Set the page configuration for the Streamlit app -------------------------------------------------------
st.set_page_config(
    page_title="Planning and Video Selection",
    page_icon=":book:",
    layout="wide",  # Use 'wide' layout for more space
    initial_sidebar_state="expanded"  # Start with the sidebar expanded
)
# Page title
st.header("🎬 Planning and Video Selection")
# --------------------------------------------------------------------------------------------------------
from gui.sidebar import add_sidebar  # Import the function to add the sidebar
add_sidebar(parent_dir)
# -------------------------------------------------------------------------------------------------------- 


st.subheader("Fill in your information:")
study_owner_name = st.text_input("Your name (e.g., M_Tolba)", "M_Tolba")
study_owner_institution = st.text_input("Your institution (e.g., Monash)", "Monash")
study_owner_position = st.selectbox("You are doing this study as a", ["Researcher", "Educator", "Both", "Other"])

st.divider()

st.subheader("Fill in your study details:")
study_purpose = st.text_area(" State below the purpose and context of the study", "State here the purpose and context of the analysis", height=75)
video_type = st.selectbox("Video Type", ["Lecture", "Tutorial", "Lab", "Seminar", "Other"])
video_platform = st.selectbox("Video-Hosting Platform", ["YouTube", "Dailymotion", "Panopto", "Echo 360", "Kaltura"])
if video_platform != "YouTube":
    st.warning(f"{video_platform} is not yet supported. This project is primarily focused on YouTube data collection and analysis. Please check back later for updates.")
institution_name = st.text_input("Institution Name (e.g., Monash)", "Monash")
speaker_name = st.text_input("Speaker Name (e.g., M_Tolba)", "M_Tolba")
course_code = st.text_input("Course Code (e.g., TRC3200)", "TRC3200")
course_name = st.text_input("Course Name (e.g., Dynamical Systems)", "Dynamical Systems")
unit_level = st.text_input("Unit Level (e.g., Year_3)", "Year_3")
academic_year = st.text_input("Academic Year (e.g., 2025)", "2025")
subject_area = st.text_input("Subject Area (e.g., Mechanical Engineering)", "Mechanical Engineering")

study_info = {
    "study_owner_name": study_owner_name,
    "study_owner_institution": study_owner_institution,
    "study_owner_position": study_owner_position,
    "study_purpose": study_purpose,
    "video_type": video_type,
    "video_platform": video_platform,
    "institution_name": institution_name,
    "speaker_name": speaker_name,
    "course_code": course_code,
    "course_name": course_name,
    "unit_level": unit_level,
    "academic_year": academic_year,
    "subject_area": subject_area
}

study_info_text = f"""
**Study Owner Name:** {study_owner_name} \n 
**Institution:** {study_owner_institution} \n
**Position:** {study_owner_position} \n
**Purpose of Study:** {study_purpose} \n
**Video Type:** {video_type} \n
**Video Platform:** {video_platform} \n
**Institution Name:** {institution_name} \n
**Speaker Name:** {speaker_name} \n
**Course Code:** {course_code} \n
**Course Name:** {course_name} \n
**Unit Level:** {unit_level} \n
**Academic Year:** {academic_year} \n
**Subject Area:** {subject_area}
"""
# st.divider()
# st.markdown("###  Study information as it will appear in the final report:")
# st.markdown(study_info_text, unsafe_allow_html=True)




# submission_data = {
#     "institution_name": institution_name,
#     "speaker_name": speaker_name,
#     "course_code": course_code,
#     "course_name": course_name,
#     "unit_level": unit_level,
#     "year": academic_year,
#     "video_type": video_type,
#     "subject_area": subject_area}
# 