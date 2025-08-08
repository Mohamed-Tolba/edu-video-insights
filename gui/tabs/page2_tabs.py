import streamlit as st
import pandas as pd
import os

# Add the parent directory (project/) to the Python path
from scripts.extract_metadata import *  # Import the function to populate new metadata file
from scripts.create_temp_data_files import *
from scripts.extract_metrics import *  # Import the function to extract metrics 
from scripts.extract_characteristics import *  # Import the function to extract characteristics

upload_dir = parent_dir + '/' + 'temp/user_data'
user_data_file_path = upload_dir + '/' + 'user_data.csv'
video_submission_file_path = upload_dir + '/' + 'video_submission.csv'
new_metadata_file_path = upload_dir + '/' + 'new_metadata.csv'
new_metrics_file_path = upload_dir + '/' + 'new_metrics.csv'
new_characs_file_path = parent_dir + '/' + 'new_characs.csv'  # Path to the new metadata file

def create_new_video_submission_file(MetadataExtractor_obj):
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
    uploaded_file = st.file_uploader("Choose a CSV file to upload", type=["csv"], key="user_data_file_csv_uploader")

    # If a file is uploaded
    if uploaded_file is not None:
        st.success(f"✅ Uploaded file: `{uploaded_file.name}`")

        try:
            # Read CSV using pandas
            df = pd.read_csv(uploaded_file) 
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
            institution_name = st.text_input("Institution Name", "Institution")
            speaker_name = st.text_input("Speaker Name (e.g., FirstName_LastName)", "Speaker")
            course_code = st.text_input("Course Code (e.g., MECXXXX)", "UnitCode")
            course_name = st.text_input("Course Name (e.g., Engineering Materials)", "Engineering_Materials")
            unit_level = st.text_input("Unit Level (e.g., Year_1)", "Year_1")
            academic_year = st.text_input("Academic Year (e.g., 2025)", "2020")
            video_type = st.selectbox("Video Type", ["Lecture", "Tutorial", "Lab", "Seminar", "Other"])
            subject_area = st.text_input("Subject Area (e.g., Mechanical_Engineering)", "Mechanical_Engineering")
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
                            label = "📥 Download Submission File",
                            data = f,
                            file_name = f"video_submission.csv",
                            mime = "text/csv"
                        )

        except Exception as e:
            st.error(f"❌ Failed to read the CSV file. Error: {e}")

def upload_video_submission_file():
    # File uploader widget (CSV only)
    uploaded_submission_file = st.file_uploader("Choose a CSV file to upload", type=["csv"], key="video_submission_file_csv_uploader")
    # If a file is uploaded
    if uploaded_submission_file is not None:
        st.success(f"✅ Uploaded file: `{uploaded_submission_file.name}`")
        try:
            # Read CSV using pandas
            df = pd.read_csv(uploaded_submission_file) 
            df.to_csv(video_submission_file_path, index=False)  # Save to a local file for further processing

            # Change index to start from 1 instead of 0
            df.index = pd.Index(range(1, len(df) + 1))

            # Display dataframe
            st.markdown("🧾 The content of the uploaded CSV file can be seen below:")
            st.dataframe(df)
        except Exception as e:
            st.error(f"❌ Failed to read the CSV file. Error: {e}")

def extract_metadata(API_KEY):
    # Call the function to populate new metadata file
    try:
        create_new_metadata_csv(new_metadata_file_path)
        populate_new_metadata_file(API_KEY, video_submission_file_path, new_metadata_file_path)
        st.success("Metadata extraction completed successfully!")
        # Read CSV using pandas
        df = pd.read_csv(new_metadata_file_path)
        # Change index to start from 1 instead of 0
        df.index = pd.Index(range(1, len(df) + 1))
        # Display dataframe
        st.subheader("🧾 Check the contents of the metadata file below")
        st.dataframe(df)
        # Offer download button
        with open(new_metadata_file_path, "rb") as f:
            st.download_button(
                label="📥 Download Metadata File",
                data=f,
                file_name=f"new_metadata.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"An error occurred while extracting metadata: {e}")

def extract_metrics():
    # Call the function to populate new metadata file
    try:
        create_new_metrics_csv(new_metrics_file_path)
        populate_new_metrics_file(video_submission_file_path, new_metrics_file_path)
        st.success("Metrics extraction completed successfully!")
        # Read CSV using pandas
        df = pd.read_csv(new_metrics_file_path)

        # Change index to start from 1 instead of 0
        df.index = pd.Index(range(1, len(df) + 1))

        # Display the DataFrame
        st.subheader("📊 Check the contents of the metrics file below")
        st.dataframe(df)

        # Offer download button
        with open(new_metrics_file_path, "rb") as f:
            st.download_button(
                label="📥 Download Metrics File",
                data=f,
                file_name=f"new_metrics.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"An error occurred while extracting metrics: {e}")

def upload_videos():
    # Upload multiple video files
    uploaded_videos = st.file_uploader(
        "Upload MP4 videos",
        type=["mp4"],
        accept_multiple_files=True,
        key="multi_video_uploader"
    )

    # If files are uploaded, save them to the upload directory
    if uploaded_videos:
        for uploaded_video in uploaded_videos:
            save_path = os.path.join(upload_dir, uploaded_video.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_video.getbuffer())
            # st.success(f"Saved: {uploaded_video.name}")

    
    # Display the uploaded videos
    if st.button("View/Unview Uploaded Videos"):
        if st.session_state['button2_4_1'] == 0:
            st.session_state['button2_4_1'] = 1
        else:
            st.session_state['button2_4_1'] = 0

    # Set the directory you want to manage
    if st.session_state['button2_4_1'] == 1:
        target_dir = upload_dir 
        st.subheader("📂 Files Manager")

        # Get list of files
        video_list = [f for f in os.listdir(target_dir) if os.path.isfile(os.path.join(target_dir, f))]

        if not video_list:
            st.info("No files found in the directory.")
        else:
            for video_name in video_list:
                video_path = os.path.join(target_dir, video_name)
                video_ext = os.path.splitext(video_name)[1].lower()

                if video_ext not in [".mp4"]: # , ".csv", ".txt"
                    continue
    
                col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 1])

                # Display file name
                with col1:
                    st.write(video_name)

                # Display file type
                with col2:
                    st.write(f"Type: {video_ext.upper()}")

                # Display file size
                with col3:
                    file_size = os.path.getsize(video_path)
                    st.write(f"Size: {file_size / 1024:.2f} KB")

                # Open file button
                with col4:
                    if st.button("📂 Open", key=f"open_video_{video_name}"):
                        st.session_state['button2_4_2'] = 1
                if st.session_state['button2_4_2'] == 1:
                    st.session_state['button2_4_2'] = 0 
                    st.video(video_path)

                # Download button
                with open(video_path, "rb") as f:
                    file_bytes = f.read()
                    with col5:
                        st.download_button(
                            label="⬇️ Download",
                            data=file_bytes,
                            file_name=video_name,
                            mime="application/octet-stream",
                            key=f"download_video_{video_name}"
                        )

                # Delete button
                with col6:
                    if st.button("🗑️ Delete", key=f"delete_video_{video_name}"):
                        os.remove(video_path)
                        st.success(f"Deleted: {video_name}")
                        st.rerun()
                
def extract_characteristics():
    if st.button("Extract Characteristics"):
        try:
            st.info("Extracting characteristics from the uploaded videos...")
            create_new_characs_csv(new_characs_file_path)
            populate_new_characs_file(video_submission_file_path, new_characs_file_path, upload_dir)  # Call the function to populate the video submission file
            st.success("Characteristics extraction completed successfully!")
            # Read CSV using pandas
            df = pd.read_csv(new_characs_file_path)
            # Change index to start from 1 instead of 0
            df.index = pd.Index(range(1, len(df) + 1))
            # Display the DataFrame
            st.subheader("🔍 Check the contents of the characteristics file below")
            st.dataframe(df)
            # Offer download button
            with open(new_characs_file_path, "rb") as f:
                st.download_button(
                    label="📥 Download Characteristics File",
                    data=f,
                    file_name=f"new_characs.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"An error occurred while extracting characteristics: {e}")   

def show_all_files():
    # Referesh the page to show all files
    if st.button("Referesh Files"):
        st.rerun()

    # Upload multiple files
    uploaded_files = st.file_uploader(
        "Upload new files",
        type=["mp4", "csv"],
        accept_multiple_files = True,
        key="multi_file_uploader"
    )

    # If files are uploaded, save them to the upload directory
    if uploaded_files:
        for uploaded_file in uploaded_files:
            save_path = os.path.join(upload_dir, uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

    target_dir = upload_dir
    all_files = [];    files_csv = [];    files_mp4 = []
    for f in sorted(os.listdir(target_dir)):
        p = os.path.join(target_dir, f)
        if os.path.isfile(p):
            file_ext = os.path.splitext(f)[1].lower()
            if file_ext == ".csv":
                files_csv.append({"Name": f, "Type": "CSV", "Size (KB)": round(os.path.getsize(p)/(1024),2)})
            elif file_ext == ".mp4":
                files_mp4.append({"Name": f, "Type": "MP4", "Size (KB)": round(os.path.getsize(p)/(1024),2)})
            # if file_ext in [".mp4", ".csv"]:
            #     files.append({"Name": f, "Type": file_ext.upper(), "Size (KB)": round(os.path.getsize(p)/(1024),2)})
    
    all_files = files_csv + files_mp4  # Combine both lists

    df = pd.DataFrame(all_files)

    st.subheader("All Files")
    st.dataframe(df, use_container_width=True, height=380)  # <- fixed height + scroll

    # Pick a file to act on
    choices = [row["Name"] for row in all_files]
    selected = st.selectbox("Select a file", choices) if choices else None

    if selected:
        file_path = os.path.join(target_dir, selected)
        file_ext = os.path.splitext(selected)[1].lower()
        col1, col2, col3, col4, _ = st.columns([1, 0.9, 0.7, 1, 4])
        with col1:
            if st.button("📂 View/Unview", key=f"view_file_{selected}"):
                if st.session_state['button2_5_2'] == 0:
                    st.session_state['button2_5_2'] = selected
                else:
                    st.session_state['button2_5_2'] = 0
        if st.session_state['button2_5_2'] == selected:
            if file_ext == ".mp4":
                st.video(file_path)
            elif file_ext == ".csv":
                # Read CSV using pandas
                df = pd.read_csv(file_path)
                # Change index to start from 1 instead of 0
                df.index = pd.Index(range(1, len(df) + 1))
                # Display the DataFrame
                st.dataframe(df)
            else:
                st.code(open(file_path).read())

        with col2:
            with open(file_path, "rb") as f:
                st.download_button("⬇️ Download", f, file_name=selected,
                                   mime="application/octet-stream")

        with col3:
            if st.button("🗑️ Delete"):
                os.remove(file_path)
                st.success(f"Deleted: {selected}")
                st.rerun()
        
        with col4:
            if st.button("🗑️ Delete All"):
                for filename in os.listdir(target_dir):
                    file_path = os.path.join(target_dir, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                st.success(f"Deleted: {selected}")
                st.rerun()

# def show_all_files2():
#     # Referesh the page to show all files
#     if st.button("Referesh Files"):
#         st.rerun()
# 
#     target_dir = upload_dir 
#     st.info("📂 Files Manager")
#     # Get list of files
#     file_list = [f for f in os.listdir(target_dir) if os.path.isfile(os.path.join(target_dir, f))]
#     if not file_list:
#         st.info("No files found in the directory.")
#     else:
#         for file_name in file_list:
#             file_path = os.path.join(target_dir, file_name)
#             file_ext = os.path.splitext(file_name)[1].lower()
#             if file_ext not in [".mp4", ".csv"]: # , ".csv", ".txt"
#                 continue
#             col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 1])
#             # Display file name
#             with col1:
#                 st.write(file_name)
#             # Display file type
#             with col2:
#                 st.write(f"Type: {file_ext.upper()}")
#             # Display file size
#             with col3:
#                 file_size = os.path.getsize(file_path)
#                 st.write(f"Size: {file_size / 1024:.2f} KB")
#             # View file button
#             with col4:
#                 if st.button("📂 View", key=f"view_file_{file_name}"):
#                     st.session_state['button2_5_2'] = 1
#             if st.session_state['button2_5_2'] == 1:
#                     st.session_state['button2_5_2'] = 0 
#                     if file_ext == ".mp4":
#                         st.video(file_path)
#                     elif file_ext == ".csv":
#                         # Read CSV using pandas
#                         df = pd.read_csv(file_path)
#                         # Change index to start from 1 instead of 0
#                         df.index = pd.Index(range(1, len(df) + 1))
#                         # Display the DataFrame
#                         st.dataframe(df)
#                     else:
#                         st.code(open(file_path).read())
#             # Download button
#             with open(file_path, "rb") as f:
#                 file_bytes = f.read()
#                 with col5:
#                     st.download_button(
#                         label="⬇️ Download",
#                         data=file_bytes,
#                         file_name=file_name,
#                         mime="application/octet-stream",
#                         key=f"download_file_{file_name}"
#                     )
#             # Delete button
#             with col6:
#                 if st.button("🗑️ Delete", key=f"delete_file_{file_name}"):
#                     os.remove(file_path)
#                     st.success(f"Deleted: {file_name}")
#                     st.rerun()