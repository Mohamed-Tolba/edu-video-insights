import streamlit as st
import pandas as pd

# Add the parent directory (project/) to the Python path
from scripts.extract_metadata import *  # Import the function to populate new metadata file
from scripts.extract_metrics import extract_metrics  # Import the function to extract metrics
from scripts.create_temp_data_files import *

def render_tab1_2(parent_dir, API_KEY, user_id):
    if st.button("Extract Metadata"):
        st.info("Extracting metadata for each video...")

        # Call the function to populate new metadata file
        try:
            video_submission_file_path = parent_dir + '/' + f'temp/video_submission_{user_id}.csv'  # Path to the video submission
            new_metadata_file_path = parent_dir + '/' + f'temp/new_metadata_{user_id}.csv'  # Path to the new metadata file
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
                    file_name="new_metadata.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"An error occurred while extracting metadata: {e}")

    if st.button("Extract Metrics"):
        st.info("Extracting metrics for each video...")

        # Call the function to populate new metadata file
        try:
            video_submission_file_path = parent_dir + '/' + f'temp/video_submission_{user_id}.csv'  # Path to the video submission
            new_metrics_file_path = parent_dir + '/' + f'data/new_metrics_{user_id}.csv'  # Path to the new metrics file
            extract_metrics(video_submission_file_path, new_metrics_file_path)
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
                    file_name="new_metrics.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"An error occurred while extracting metrics: {e}")