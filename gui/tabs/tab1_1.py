import streamlit as st
import pandas as pd

# Add the parent directory (project/) to the Python path
from scripts.extract_metadata import *  # Import the function to populate new metadata file
from scripts.create_temp_data_files import *

def render_tab1_1(MetadataExtractor_obj, parent_dir, user_id):
    """
    Render the first tab for uploading CSV files and preparing submission data.
    """
    st.subheader("📊 Step 1: Download the video IDs and required metrics from your YouTube channel")

    # Input YouTube URL
    youtube_url = st.text_input("Paste a URL for any video in your channel (e.g., https://www.youtube.com/channel/UCxxxx...), then press Enter:")

    # Extract channel ID
    channel_id = None
    if youtube_url:
        # Extract video ID and channel ID from the provided URL
        video_id = MetadataExtractor_obj.extract_video_id(youtube_url)
        channel_id = MetadataExtractor_obj.get_video_channel_id(video_id)
        if channel_id:
            st.success(f"✅ Channel ID extracted: `{channel_id}`")
        else:
            st.error("❌ Failed to extract Channel ID. Please check the URL format.")

    # Display YouTube Studio link and instructions if channel ID is valid
    if channel_id:
        studio_url = f"https://studio.youtube.com/channel/{channel_id}/analytics"

        st.markdown(
            f"""
            <p>
            🔗 <a href="{studio_url}" target="_blank">
            <strong>Open YouTube Studio Analytics for Your Channel</strong></a>
            </p>
            <p style='color:gray; font-size: 0.9em;'>
            Once the page opens:
            <ol>
                <li>Click on <strong>Advanced Mode</strong></li>
                <li>Change the metric to <strong>Average Percentage Viewed</strong></li>
                <li>Set time period as required</li>
                <li>Click <strong>Export CSV</strong> (top-right corner)</li>
            </ol>
            </p>
            """,
            unsafe_allow_html=True
        )

    st.subheader("📤 Step 2: Upload your CSV file")
    # File uploader widget (CSV only)
    uploaded_file = st.file_uploader("Choose a CSV file to upload", type=["csv"])

    # If a file is uploaded
    if uploaded_file is not None:
        st.success(f"✅ Uploaded file: `{uploaded_file.name}`")

        try:
            # Read CSV using pandas
            df = pd.read_csv(uploaded_file) 
            user_data_file_path = parent_dir + '/' + f'temp/user_data_{user_id}.csv'
            df.to_csv(user_data_file_path, index=False)  # Save to a local file for further processing

            # Change index to start from 1 instead of 0
            df.index = pd.Index(range(1, len(df) + 1))

            # Display dataframe
            st.markdown("🧾 The content of the uploaded CSV file can be seen below:")
            st.dataframe(df)
            
            # Submission metadata inputs
            # st.markdown("### 📝 Preparing a submission file for the data extraction and analysis")
            # st.markdown("### Fill in the following details for the uploaded data")
            st.subheader("Step 3: Fill in the following details for the uploaded data")
            institution_name = st.text_input("Institution Name (e.g., Monash)")
            speaker_name = st.text_input("Speaker Name (e.g., M_Tolba)")
            course_code = st.text_input("Course Code (e.g., TRC3200)")
            course_name = st.text_input("Course Name (e.g., Dynamical Systems)")
            unit_level = st.text_input("Unit Level (e.g., Year_3)")
            academic_year = st.text_input("Academic Year (e.g., 2025)")
            video_type = st.selectbox("Video Type", ["Lecture", "Tutorial", "Lab", "Seminar", "Other"])
            subject_area = st.text_input("Subject Area (e.g., Mechanical_Engineering)")
            submission_data = {
                "institution_name": institution_name,
                "speaker_name": speaker_name,
                "course_code": course_code,
                "course_name": course_name,
                "unit_level": unit_level,
                "year": academic_year,
                "video_type": video_type,
                "subject_area": subject_area}

             # Add a button to extract metadata from the uploaded CSV
            st.subheader("Step 4: Do a final check for all the data before proceeding")
            if st.button("Prepare and check the final submission file"):
                missing_fields = [k for k, v in submission_data.items() if v is None or v == ""]
                if missing_fields:
                    st.error(f"❌ The following fields are missing: {', '.join(missing_fields)}")
                else:
                    video_submission_file_path = parent_dir + '/' + f'temp/video_submission_{user_id}.csv'
                    create_video_submission_csv(video_submission_file_path)
                    populate_video_submission_file(submission_data, user_data_file_path, video_submission_file_path)
                    st.success("✅ Submission file prepared successfully!")
                    # Read CSV using pandas
                    df = pd.read_csv(video_submission_file_path)

                    # Change index to start from 1 instead of 0
                    df.index = pd.Index(range(1, len(df) + 1))

                    # Display dataframe
                    st.markdown("""🧾 Check the contents of the prepared CSV file below. If any of the data is incorrect,
                    please download the CSV file, fix it and then upload it again in the next tab.""")
                    st.dataframe(df)

                    # Offer download button
                    with open(video_submission_file_path, "rb") as f:
                        st.download_button(
                            label="📥 Download Submission File",
                            data=f,
                            file_name=f"video_submission_{user_id}.csv",
                            mime="text/csv"
                        )

        except Exception as e:
            st.error(f"❌ Failed to read the CSV file. Error: {e}")