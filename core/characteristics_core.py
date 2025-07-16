"""
Module: characteristics_core.py
Author: Mohamed Tolba
Date Created: 24-06-2025
Last Updated: 16-07-2025

Description:
    Defines the CharacteristicsExtractor class that performs audio-visual analysis
    on videos to extract features such as speaking speed, slide change frequency,
    caption availability, audio clarity, and tone variability.

    Extracted features are written to new_characs.csv.
    Delete this line # source .venv/bin/activate

To Validate/Tune:
    1. The scene change detection threshold can be adjusted in the ContentDetector.
    We need to define what constitutes a significant scene change.
"""
import sys
import os
import re # Regular expression module for text processing

from scenedetect import open_video, SceneManager 
# open_video: Recommended function to load video files for analysis in PySceneDetect.
# SceneManager: Orchestrates detection logic and stores detected scene boundaries.
from scenedetect.detectors import ContentDetector 
# ContentDetector: A detection algorithm that identifies significant visual content changes between frames (based on pixel difference)

# Build the absolute path to the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add to Python module search path
sys.path.append(parent_dir)

import yt_dlp
from core.keys_manager import load_api_key  # Import the function to load the API key
from core.metadata_core import MetadataExtractor

class CharacsExtractor:
    def __init__(self, API_KEY: str):
        self.MetadataExtractor_obj = MetadataExtractor(API_KEY)
        
    def get_video_duration_min(self, video_id: str) -> float:
        """
        Get the duration of the video in minutes.
        """
        duration_sec = self.MetadataExtractor_obj.get_video_duration_sec(video_id)
        if duration_sec is None:
            return None  # Return None if duration is not available
        return round(duration_sec / 60.0, 2)  # Convert seconds to minutes
    
    def count_words(self, video_id: str) -> int:
        """
        Count the number of words in the video's transcript.
        """
        video_transcript = self.MetadataExtractor_obj.get_video_transcript(video_id)
        if not video_transcript:
            return None  # Return None if no transcript is available
        
        # Remove bracketed words like [music], [laughter], [applause], etc.
        cleaned_transcript = re.sub(r'\[.*?\]', '', video_transcript)

        # Assuming the transcript is a string of words, we can split it to count words
        words = cleaned_transcript.split()
        return len(words)  # Return the number of words in the transcript
    
    def extract_word_per_minute(self, video_id: str) -> float:
        """
        Calculate the speaking speed in words per minute.
        """
        duration_sec = self.MetadataExtractor_obj.get_video_duration_sec(video_id)
        word_count = self.count_words(video_id)
        
        if duration_sec > 0 and word_count is not None:
            word_count_per_minute = word_count / (duration_sec / 60.0)  # Convert seconds to minutes
            return round(word_count_per_minute, 2)  # Round to 2 decimal places
        else:
            return None
        
    def download_youtube_video(self, video_id, save_path="."):
        """Download a YouTube video using yt-dlp.   
        Args:
            video_id (str): The YouTube video ID.
            save_path (str): The directory where the video will be saved.
        """
        video_url = self.MetadataExtractor_obj.construct_video_url(video_id)
        if not video_url:
            print(f"Could not retrieve video URL for video ID: {video_id}")
            return
        ydl_opts = {
            'outtmpl': f"{save_path}/%(id)s.%(ext)s", # Save the video with its ID as the filename
            'noplaylist': True,  # Download only the single video, not the playlist
            'quiet': True,  # Suppress output messages
            'format': 'mp4',
        }
        ydl = yt_dlp.YoutubeDL(ydl_opts)
        ydl.download([video_url])
    
    def count_scene_changes(self, video_id, video_path=".", threshold=10.0):
        """Detect and count scene changes in a video using PySceneDetect.
        Args:
            video_path (str): The path to the video file.
            video_id (str, optional): The YouTube video ID. Defaults to None.
        Returns:
            int: The total number of detected scene changes.
        """

        # Open video using the recommended method
        video = open_video(video_path) # This function is used to load the video file for analysis.

        # Create SceneManager and add ContentDetector
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold = threshold))
        # Analyses consecutive frames. If pixel-wise content difference exceeds threshold, a scene/slide change is detected.
        # threshold: The threshold value for detecting scene changes. A higher value means fewer scene changes will be detected.

        # Perform scene detection
        scene_manager.detect_scenes(video)
        # This method processes the video (frame-by-frame) and detects scene changes based on the added detectors.
        # The detected scenes are stored in the SceneManager instance.

        # Get list of detected scenes (scene cuts)
        scene_list = scene_manager.get_scene_list()
        # This method retrieves the list of detected scenes, which are represented as tuples of start and end frames.
        # Each tuple contains the start and end frame indices of a detected scene change.
        # Example: [(start_frame1, end_frame1), (start_frame2, end_frame2), ...]

        total_duration_sec = self.MetadataExtractor_obj.get_video_duration_sec(video_id)

        # Filter scenes to exclude first & last seconds
        filtered_scenes = []
        skip_start_sec = 1  # Skip the first second
        skip_end_sec = 2    # Skip the last second
        for start_time, end_time in scene_list:
            start_sec = start_time.get_seconds()
            end_sec = end_time.get_seconds()

            if start_sec >= skip_start_sec and end_sec <= (total_duration_sec - skip_end_sec):
                filtered_scenes.append((start_time, end_time))

        # print(filtered_scenes)
        # Return total number of detected scene changes
        return len(filtered_scenes)
    
    def extract_scene_change_freq(self, video_id, video_path=".", threshold=10.0) -> float:
        """
        Calculate the frequency of scene changes in a video.
        """
        total_duration_sec = self.MetadataExtractor_obj.get_video_duration_sec(video_id)
        total_scene_changes = self.count_scene_changes(video_id, video_path, threshold)
        if total_duration_sec > 0 and total_scene_changes > 0:
            scene_change_freq = total_scene_changes / (total_duration_sec / 60.0)  # Convert seconds to minutes
            return round(scene_change_freq, 2)  # Round to 2 decimal places
        else:
            return None
        

if __name__ == "__main__":
    API_KEY = load_api_key(parent_dir + '/' + "keys/youtube_data_API_key.txt")  # Load the YouTube Data API key from the specified file
    CharacsExtractor_obj = CharacsExtractor(API_KEY)  # Create an instance of the CharacsExtractor class with the API key
    video_id = "mqCdqQg3dKU"  # Replace with an actual video ID for testing
    
    # MetadataExtractor_obj = MetadataExtractor(API_KEY)
    # duration_sec = MetadataExtractor_obj.get_video_duration_sec(video_id)  # Get the video duration in seconds
    # if duration_sec is not None:
    #     print(f"Video duration for {video_id}: {duration_sec} seconds")
    # else:
    #     print(f"Could not retrieve duration for video {video_id}. Check if the video ID is correct or if the video exists.")
    # 
    # duration_min = CharacsExtractor_obj.get_video_duration_min(video_id)  # Get the video duration
    # print(f"Video duration for {video_id}: {duration_min} minutes")  # Print the video duration
# 
    # word_count = CharacsExtractor_obj.count_words(video_id)  # Count the words in the video's transcript
    # if word_count is not None:
    #     print(f"Word count for video {video_id}: {word_count}")
    # else:
    #     print(f"No transcript available for video {video_id}.")
# 
    # speaking_speed = CharacsExtractor_obj.extract_word_per_minute(video_id)  # Calculate speaking speed
    # if speaking_speed is not None:
    #     print(f"Speaking speed for video {video_id}: {speaking_speed} words per minute")
    # else:
    #     print(f"Could not calculate speaking speed for video {video_id}. Check duration and word count.")


    save_dir = parent_dir + '/temp'
    CharacsExtractor_obj.download_youtube_video(video_id, save_dir)  # Download the video
    video_path = os.path.join(save_dir, f"{video_id}.mp4")  # Construct the path to the downloaded video
    if os.path.exists(video_path):
        print(f"Video downloaded successfully to {video_path}")
    else:
        print(f"Failed to download video {video_id}. Check if the video ID is correct or if the video exists.")
    # Detect scene changes in the downloaded video
    total_scene_changes = CharacsExtractor_obj.count_scene_changes(video_id, video_path, threshold = 10.0)  # Set the threshold for scene change detection
    print(f"Total scene changes detected in {video_path}: {total_scene_changes}")
    scene_change_freq = CharacsExtractor_obj.extract_scene_change_freq(video_id, video_path, threshold = 10.0)  # Calculate scene change frequency
    if scene_change_freq is not None:
        print(f"Scene change frequency for video {video_id}: {scene_change_freq} changes per minute")   
# 
    # 
    # print(video_url)
    # # Download the video using pytube
    # try: 
    #     yt = YouTube(video_url)
    #     print(yt.title)
    #     print(yt.description)
    #     print(yt.length)  # Video length in seconds
    #     print(yt.author)
# 
    #     ys = yt.streams.get_highest_resolution()
# 
    #     print(ys)
    #     ys.download()
    # except Exception as e:
    #     print(f"An error occurred: {e}")
    # Example usage
    # video_path = "example_video.mp4"
    # extractor = CharacteristicsExtractor(video_path)
    # extractor.extract_features()
    # extractor.save_to_csv("new_characs.csv")
    # print("Characteristics extracted and saved to new_characs.csv.")