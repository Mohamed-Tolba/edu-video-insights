"""
Module: extract_metrics.py
Author: Mohamed Tolba
Date Created: 24-06-2025
Last Updated: 8-07-2025

Description:
    Calls the MetricsExtractor from core/metrics_core.py to collect and write
    video engagement metrics into new_metrics.csv. This includes both public API
    values and optional user-supplied private metrics.
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
    
    new_metrics_file_path = '../data/new_metrics.csv'
    new_metrics_file_handler = CSVHandler(new_metrics_file_path)
    
    new_metrics_file_handler.clear_all_rows(msg = "Any data in the new_metrics_file has been deleted")  # Clear all data in the new metrics file, keeping only the header
    video_ids = video_submission_file_handler.df['video_id'].tolist()  # Fetch all video IDs from the video submission file
    for video_id in video_ids:  # Loop through each video ID
        if video_id:  # Check if the video ID is not empty
            video_metrics = {
                "video_id": video_id,  # Store the video ID
                "dataset_tag": video_submission_file_handler.get_cell_value_by_match("video_id", video_id, "dataset_tag"), # Get the subject area of the video from the video submission file
                "retention_rate": video_submission_file_handler.get_cell_value_by_match("video_id", video_id, "retention_rate"), # Get the subject area of the video from the video submission file
                "avg_view_duration": video_submission_file_handler.get_cell_value_by_match("video_id", video_id, "avg_view_duration"), # Get the subject area of the video from the video submission file
                "total_views": MetadataExtractor_obj.get_video_views(video_id),  # Get the published date and time of the video
                "likes": MetadataExtractor_obj.get_video_likes(video_id),  # Get the published date and time of the video
                "comments": MetadataExtractor_obj.get_video_comments(video_id) # Get the published date and time of the video
            }
            new_metrics_file_handler.add_new_row(video_metrics)  # Populate the row with the video metadata
    
    new_metrics_file_handler.clean_csv() # Clean the new metadata file by removing invalid rows and duplicates, and extra unnamed columns
    print(new_metrics_file_handler.df)  # Print the DataFrame containing the new metadata


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