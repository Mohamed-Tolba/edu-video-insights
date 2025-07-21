"""
Module: clean_user_data.py
Author: Mohamed Tolba
Date Created: 15-07-2025
Last Updated: 21-07-2025

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

def compare_and_clean(user_data_file_path: str = '.', ref_data_file_path_1: str = '.', ref_data_file_path_2: str = '.', type = '') -> None:
    """
    Compares the user data file with a reference data file and cleans the user data file.
    Removes rows from the user data file that do not exist in the reference data file.
    """

    user_data_file_handler = CSVHandler(user_data_file_path)  # Create a CSVHandler instance for the user data file
    ref_data_file_handler_1 = CSVHandler(ref_data_file_path_1)  # Create a CSVHandler instance for the reference data file
    ref_data_file_handler_2 = CSVHandler(ref_data_file_path_2)  # Create a CSVHandler instance for the reference data file

    # Loop through each video ID in the ref data file 1, then retrive info from ref data file 2. Aftrwards, populate the user data file
    ref_video_ids_1 = ref_data_file_handler_1.df['Video'].tolist()  # Fetch all video IDs from the video uref data file
    for video_id in ref_video_ids_1:  # Loop through each video ID
        video_type = ref_data_file_handler_1.get_cell_value_by_match('Video', video_id, 'Type')
        if video_type == type: 
            average_percentage_viewed = ref_data_file_handler_2.get_cell_value_by_match('Content', video_id, "Average percentage viewed (%)")
            video_user_inputs = {
                "video_id": video_id,  # Store the video ID
                "video_type": video_type,
                "average_percentage_viewed": average_percentage_viewed # Get the average_percentage_viewed from the user data file
            }
            user_data_file_handler.add_new_row(video_user_inputs)  # Populate the row with the video user inputs
    user_data_file_handler.clean_csv()  # Clean the video submission file by removing invalid rows and duplicates, and extra unnamed columns
       
if __name__ == "__main__":
    user_data_file_path = parent_dir + '/' + 'temp/Test_Data/raw/MECH1750_Lectures.csv'  # Build the absolute path to the user data file
    ref_data_file_path_1 = parent_dir + '/' + 'temp/Test_Data/raw/MECH1750_1.csv'
    ref_data_file_path_2 = parent_dir + '/' + 'temp/Test_Data/raw/MECH1750_2.csv'
    compare_and_clean(user_data_file_path, ref_data_file_path_1, ref_data_file_path_2, 'Lecture')  # Call the function to clean the user data file
    print("User data file cleaned successfully.")
    # print(user_data_file_handler.df)  # Print the DataFrame containing the cleaned user