"""
Module: extract_metadata.py
Author: Mohamed Tolba
Date Created: 24-06-2025
Last Updated: 08-07-2025

Description:
    Script that loads video IDs from video_submission.csv and uses the MetadataExtractor
    (from core/metadata_core.py) to fetch and write metadata for each video into new_metadata.csv.

To test/develop:
- I think there is a smarter way to copy data from video_submission.csv to new_metadata.csv
"""

import sys
import os

# Add the parent directory (project/) to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.metadata_core import MetadataExtractor  # Import the MetadataExtractor class from the core module
from core.csv_utils import CSVHandler  # Import the CSVHandler class for managing CSV files

if __name__ == "__main__":
    API_KEY = "AIzaSyBb1gDmCnge668A6FG3cppBlEib3CsQ4zc"  # Mohamed's YouTube Data API key - generated for a project named "Educational Videos Project 1".
    # 'AIzaSyAItxG2Mye_NlmofGrmpX50pB-g6txm3Kw' is an API key provided by Alex

    MetadataExtractor_obj = MetadataExtractor(API_KEY)

    video_submission_file_path = '../data/video_submission.csv'
    video_submission_file_handler = CSVHandler(video_submission_file_path)

    new_metadata_file_path = '../data/new_metadata.csv'
    new_metadata_file_handler = CSVHandler(new_metadata_file_path)

    new_metadata_file_handler.clear_all_rows(msg = "Any data in the new_metadata_file has been deleted")  # Clear all data in the new metadata file, keeping only the header
    new_metadata_file_handler.clean_csv() 
    video_ids = video_submission_file_handler.df['video_id'].tolist()  # Fetch all video IDs from the video submission file
    for video_id in video_ids:  # Loop through each video ID
        if video_id:  # Check if the video ID is not empty
            video_metadata = {
                "video_id": video_id,  # Store the video ID
                "dataset_tag": video_submission_file_handler.get_cell_value_by_match("video_id", video_id, "dataset_tag"), # Get the subject area of the video from the video submission file
                "institution_name": video_submission_file_handler.get_cell_value_by_match("video_id", video_id, "institution_name"), # Get the subject area of the video from the video submission file
                "speaker_name": video_submission_file_handler.get_cell_value_by_match("video_id", video_id, "speaker_name"), # Get the subject area of the video from the video submission file
                "course_code": video_submission_file_handler.get_cell_value_by_match("video_id", video_id, "course_code"), # Get the subject area of the video from the video submission file
                "unit_level": video_submission_file_handler.get_cell_value_by_match("video_id", video_id, "unit_level"), # Get the subject area of the video from the video submission file
                "week_number": video_submission_file_handler.get_cell_value_by_match("video_id", video_id, "week_number"), # Get the subject area of the video from the video submission file
                "year": video_submission_file_handler.get_cell_value_by_match("video_id", video_id, "year"), # Get the subject area of the video from the video submission file
                "video_type": video_submission_file_handler.get_cell_value_by_match("video_id", video_id, "video_type"), # Get the subject area of the video from the video submission file
                "subject_area": video_submission_file_handler.get_cell_value_by_match("video_id", video_id, "subject_area"), # Get the subject area of the video from the video submission file            
                "duration_sec": MetadataExtractor_obj.get_video_duration_sec(video_id),  # Get the video duration
                "video_url": MetadataExtractor_obj.construct_video_url(video_id),  # Construct the full video URL from the video ID
                "title": MetadataExtractor_obj.get_video_title(video_id),  # Get the video title using the video ID
                "channel_id": MetadataExtractor_obj.get_video_channel_id(video_id),  # Get the channel ID of the video
                "channel_name": MetadataExtractor_obj.get_video_channel_name(video_id),  # Get the channel title of the video
                "published_at": MetadataExtractor_obj.get_video_published_at(video_id),  # Get the published date and time of the video
            }
            new_metadata_file_handler.add_new_row(video_metadata)  # Populate the row with the video metadata

    new_metadata_file_handler.clean_csv() # Clean the new metadata file by removing invalid rows and duplicates, and extra unnamed columns
    print(new_metadata_file_handler.df)  # Print the DataFrame containing the new metadata