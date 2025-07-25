"""
Module: extract_characteristics.py
Author: Mohamed Tolba
Date Created: 24-06-2025
Last Updated: 22-07-2025

Description:
    Orchestrates the extraction of video characteristics by calling the CharacsExtractor
    class defined in core/characteristics_core.py. The output is written to new_characs.csv.
"""

import sys
import os

# Build the absolute path to the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add to Python module search path
sys.path.append(parent_dir)

# from core.keys_manager import load_api_key  # Import the function to load the API key
# from core.metadata_core import MetadataExtractor  # Import the MetadataExtractor class from the core module

from core.characteristics_core import CharacsExtractor
from core.csv_utils import CSVHandler  # Import the CSVHandler class for managing CSV files
from scripts.create_temp_data_files import *

CharacsExtractor_obj = CharacsExtractor()

def donwload_all_videos(video_submission_file_path: str = 'temp/video_submission.csv', temp_save_dir = 'temp') -> None:
    video_submission_file_handler = CSVHandler(video_submission_file_path)
    video_ids = video_submission_file_handler.df['video_id'].tolist()  # Fetch all video IDs from the video submission file
    counter = 1
    for video_id in video_ids:  # Loop through each video ID
       if video_id:  # Check if the video ID is not empty
           print(f"Downloading video {counter}/{len(video_ids)}... video id: {video_id}, video URL: https://www.youtube.com/watch?v={video_id}")
           CharacsExtractor_obj.download_youtube_video_audio(video_id, temp_save_dir)  # Download the video
           counter = counter + 1


def populate_new_characs_file(video_submission_file_path: str = 'temp/video_submission.csv', new_characs_file_path: str = 'temp/new_characs.csv', temp_save_dir = 'temp') -> None:
    """
    Populates the new_characs.csv file with video characteristics.
    This function reads video IDs from video_submission.csv and creates a new new_characs.csv file.
    """
    video_submission_file_handler = CSVHandler(video_submission_file_path)
    new_characs_file_handler = CSVHandler(new_characs_file_path)

    new_characs_file_handler.clear_all_rows(msg = "Any data in the new_characs_file has been deleted")  # Clear all data in the new metadata file, keeping only the header
    new_characs_file_handler.clean_csv()
    print("Creating new characs file...")
    # Loop through each video ID in the video submission file and fetch metadata 
    counter = 1
    video_ids = video_submission_file_handler.df['video_id'].tolist()  # Fetch all video IDs from the video submission file
    for video_id in video_ids:  # Loop through each video ID
        if video_id:  # Check if the video ID is not empty
            print(f"Analysing video {counter}/{len(video_ids)}... video id: {video_id}, video URL: https://www.youtube.com/watch?v={video_id}")
            video_path = temp_save_dir + '/' + f"{video_id}.mp4"
            video_characs = {
                "video_id": video_id,  # Store the video ID
                "dataset_tag": video_submission_file_handler.get_cell_value_by_match("video_id", video_id, "dataset_tag"),
                "duration_min": CharacsExtractor_obj.extract_video_duration(video_path, 'min'),
                "speaking_words_count": CharacsExtractor_obj.count_words(video_path),
                "avg_speaking_speed_wpm": CharacsExtractor_obj.extract_word_per_minute(video_path),
                "scenes_count": CharacsExtractor_obj.count_scene_changes(video_path, 30),
                "avg_scene_change_per_min": CharacsExtractor_obj.extract_scene_change_per_min(video_path, 30),
            }
            new_characs_file_handler.add_new_row(video_characs)  # Populate the row with the video characs
            CharacsExtractor_obj.delete_file(video_path)
            counter = counter + 1
    new_characs_file_handler.clean_csv() # Clean the new metadata file by removing invalid rows and duplicates, and extra unnamed columns

def delete_all_videos(video_submission_file_path: str = 'temp/video_submission.csv', temp_save_dir = 'temp') -> None:
    video_submission_file_handler = CSVHandler(video_submission_file_path)
    video_ids = video_submission_file_handler.df['video_id'].tolist()  # Fetch all video IDs from the video submission file
    counter = 1
    for video_id in video_ids:  # Loop through each video ID
       if video_id:  # Check if the video ID is not empty
           print(f"Deleting video {counter}/{len(video_ids)}... video id: {video_id}, video URL: https://www.youtube.com/watch?v={video_id}")
           video_path = temp_save_dir + '/' + f"{video_id}.mp4"
           # CharacsExtractor_obj.delete_file(video_path)
           counter = counter + 1

if __name__ == "__main__":
    temp_save_dir = parent_dir + '/temp'
    video_submission_file_path = parent_dir + '/' + 'temp/MECH1750_Lectures_video_submission.csv'  # Path to the video submission
    new_characs_file_path = parent_dir + '/' + 'temp/MECH1750_Lectures_characs.csv'  # Path to the new metadata file
    
    create_new_characs_csv(new_characs_file_path)

    donwload_all_videos(video_submission_file_path, temp_save_dir)
    populate_new_characs_file(video_submission_file_path, new_characs_file_path, temp_save_dir)  # Call the function to populate the video submission file
    # delete_all_videos(video_submission_file_path, temp_save_dir)
    
    #print("Video characs file populated successfully.")