"""
Module: merge_new_data.py
Author: Mohamed Tolba
Date Created: 24-06-2025
Last Updated: 04-08-2025

Description:
    Reads validated entries from the new_*.csv files and appends them to their corresponding
    all_*.csv datasets while avoiding duplicate video IDs and maintaining clean formatting.
"""

import os, sys
# Build the absolute path to the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add to Python module search path
sys.path.append(parent_dir)

from core.csv_utils import CSVHandler  # Import the CSVHandler class for managing CSV files
from scripts.create_temp_data_files import *

all_characs_file_path = parent_dir + '/' + 'data/all_characs.csv'
all_metadata_file_path = parent_dir + '/' + 'data/all_metadata.csv'
all_metrics_file_path = parent_dir + '/' + 'data/all_metrics.csv'
all_train_dataset_file_path = parent_dir + '/' + 'data/all_train_dataset.csv'

if (os.path.exists(all_characs_file_path) == False):
    create_new_characs_csv(all_characs_file_path)  # Create the all characs file

if (os.path.exists(all_metadata_file_path) == False):
    create_new_metadata_csv(all_metadata_file_path)  # Create the all metadata file

if (os.path.exists(all_metrics_file_path) == False):
    create_new_metrics_csv(all_metrics_file_path)  # Create the all metrics file 

if (os.path.exists(all_train_dataset_file_path) == False):
    create_train_dataset_csv(all_train_dataset_file_path)  # Create the all train dataset file


def merge_new_data(all_file_path, new_file_path):
    """
    Merges new data from new_file_path into all_file_path.
    Avoids duplicates based on video_id and ensures clean formatting.
    """
    all_file_handler = CSVHandler(all_file_path)
    new_file_handler = CSVHandler(new_file_path)
    
    # Append new data to the all file
    for row in new_file_handler.df.iterrows():
        all_file_handler.add_new_row(row[1].to_dict())  # Convert the row to a dictionary and add it to the all file
        # print()


        # if row['video_id'] not in all_file_handler.df['video_id'].values:
        #     row['dataset_tag'] = dataset_tag  # Set the dataset tag for the new entry
    
    # Clean the all file by removing invalid rows and duplicates
    all_file_handler.clean_csv()
    all_file_handler.sort_by_field("dataset_tag", ascending = False)  # Sort by the field in ascending or descending order

    print(f"Merged new data into {all_file_path} successfully.")

if __name__ == "__main__":
    # Define the new data file paths
    Unit_1 = "MECH1750_Lectures"
    Unit_2 = "ENGG2240_Lectures"

    Unit = Unit_1

    new_characs_file_path = parent_dir + '/' + 'temp/' + Unit + '_characs.csv'
    new_metadata_file_path = parent_dir + '/' + 'temp/' + Unit + '_metadata.csv'
    new_metrics_file_path = parent_dir + '/' +  'temp/' + Unit + '_metrics.csv'
    new_train_dataset_file_path = parent_dir + '/' + 'temp/' + Unit + '_train_dataset.csv'

    # Merge new data into all datasets
    merge_new_data(all_characs_file_path, new_characs_file_path)
    merge_new_data(all_metadata_file_path, new_metadata_file_path)
    merge_new_data(all_metrics_file_path, new_metrics_file_path)
    merge_new_data(all_train_dataset_file_path, new_train_dataset_file_path)