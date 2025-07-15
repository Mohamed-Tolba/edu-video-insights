"""
Module: clean_user_data.py
Author: Mohamed Tolba
Date Created: 15-07-2025
Last Updated: 15-07-2025

Description:
    Script that cleans the user data CSV file by removing invalid rows, duplicates, and extra unnamed columns.
"""

import sys
import os

# Build the absolute path to the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Print the path
# print("📂 Parent directory path being added to sys.path:", parent_dir)

# Add to Python module search path
sys.path.append(parent_dir)

from core.csv_utils import CSVHandler  # Import the CSVHandler class for managing CSV files

def compare_and_clean(user_data_file_path: str = 'data/user_data.csv', ref_data_file_path: str = 'data/ref_data.csv') -> None:
    """
    Compares the user data file with a reference data file and cleans the user data file.
    Removes rows from the user data file that do not exist in the reference data file.
    """
    user_data_file_path = parent_dir + '/' + user_data_file_path  # Build the absolute path to the user data file
    ref_data_file_path = parent_dir + '/' + ref_data_file_path  # Build the

    user_data_file_handler = CSVHandler(user_data_file_path)  # Create a CSVHandler instance for the user data file
    ref_data_file_handler = CSVHandler(ref_data_file_path)  # Create a CSVHandler instance for the reference data file

    # Loop through each video ID in the user data file and populate the video submission file
    ref_video_ids = ref_data_file_handler.df['video_id'].tolist()  # Fetch all video IDs from the video uref data file
    print(ref_data_file_handler.df)
    video_ids = user_data_file_handler.df['Content'].tolist()[1:]  # Fetch all video IDs from the video user data file
    for video_id in video_ids:  # Loop through each video ID
        if video_id in ref_video_ids:  # Check if the video ID is in the reference data file
            pass
        else:
            user_data_file_handler.remove_row_by_field("Content", video_id)  # Remove the row with the video ID from the user data file
    

if __name__ == "__main__":
    compare_and_clean()  # Call the function to clean the user data file
    print("User data file cleaned successfully.")
    # print(user_data_file_handler.df)  # Print the DataFrame containing the cleaned user