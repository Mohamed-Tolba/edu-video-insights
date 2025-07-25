"""
Module: extract_metrics.py
Author: Mohamed Tolba
Date Created: 24-06-2025
Last Updated: 15-07-2025

Description:
    Calls the MetricsExtractor from core/metrics_core.py to collect and write
    video engagement metrics into new_metrics.csv. This includes both public API
    values and optional user-supplied private metrics.
"""

import sys
import os

# Build the absolute path to the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add to Python module search path
sys.path.append(parent_dir)

from core.csv_utils import CSVHandler  # Import the CSVHandler class for managing CSV files
from scripts.create_temp_data_files import *

def populate_new_metrics_file(video_submission_file_path: str = 'temp/video_submission.csv', new_metrics_file_path: str = 'temp/new_metrics.csv') -> None:
    """
    Extracts video metrics from the video submission file and writes them to a new metrics file.
    """

    video_submission_file_handler = CSVHandler(video_submission_file_path)
    new_metrics_file_handler = CSVHandler(new_metrics_file_path)

    new_metrics_file_handler.clear_all_rows(msg = "Any data in the new_metrics_file has been deleted")  # Clear all data in the new metrics file, keeping only the header
    video_ids = video_submission_file_handler.df['video_id'].tolist()  # Fetch all video IDs from the video submission file
    for video_id in video_ids:  # Loop through each video ID
        if video_id:  # Check if the video ID is not empty
            video_metrics = {
                "video_id": video_id,  # Store the video ID
                "dataset_tag": video_submission_file_handler.get_cell_value_by_match("video_id", video_id, "dataset_tag"), # Get the dataset tag of the video from the video submission file
                "average_percentage_viewed": video_submission_file_handler.get_cell_value_by_match("video_id", video_id, "average_percentage_viewed"), # Get the average_percentage_viewed from the video submission file
            }
            new_metrics_file_handler.add_new_row(video_metrics)  # Populate the row with the video metadata
    
    new_metrics_file_handler.clean_csv() # Clean the new metadata file by removing invalid rows and duplicates, and extra unnamed columns

if __name__ == "__main__":
    video_submission_file_path = parent_dir + '/' + 'temp/ENGG2240_Lectures_video_submission.csv'
    new_metrics_file_path = parent_dir + '/' + 'temp/ENGG2240_Lectures_metrics.csv'
    create_new_metrics_csv(new_metrics_file_path)
    populate_new_metrics_file(video_submission_file_path, new_metrics_file_path)  # Call the function to extract metrics from the video submission file
    print("Metrics extraction completed successfully.")


"""
You may consider adding more metrics to the new_metrics.csv file, such as:
- "dislikes": MetadataExtractor_obj.get_video_dislikes(video_id)  # Get the number of dislikes for the video
- "shares": MetadataExtractor_obj.get_video_shares(video_id)  # Get the number of shares for the video
- "average_watch_time": MetadataExtractor_obj.get_average_watch_time(video_id)  # Get the average watch time for the video
- "engagement_rate": MetadataExtractor_obj.get_engagement_rate(video_id)  # Calculate the engagement rate for the video
- "subscriber_change": MetadataExtractor_obj.get_subscriber_change(video_id)
- "video_quality": MetadataExtractor_obj.get_video_quality(video_id)  # Get the quality of the video (e.g., 720p, 1080p)
- "video_category": MetadataExtractor_obj.get_video_category(video_id)  # Get the category of the video
- "video_tags": MetadataExtractor_obj.get_video_tags(video_id)  # Get the tags associated with the video
- "video_language": MetadataExtractor_obj.get_video_language(video_id)  # Get the language of the video
- "video_thumbnail_url": MetadataExtractor_obj.get_video_thumbnail_url(video_id)  # Get the URL of the video thumbnail
- "video_transcript": MetadataExtractor_obj.get_video_transcript(video_id)  # Get the transcript of the video
- "video_comments_count": MetadataExtractor_obj.get_video_comments_count(video_id)  # Get the number of comments for the video
- "video_comments_sentiment": MetadataExtractor_obj.get_video_comments_sentiment(video_id)  # Get the sentiment of the comments for the video
- "video_engagement_score": MetadataExtractor_obj.get_video_engagement_score(video_id)  # Calculate the engagement score for the
.
.
.
"""