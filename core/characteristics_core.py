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
"""
import sys
import os
import re # Regular expression module for text processing

# Build the absolute path to the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add to Python module search path
sys.path.append(parent_dir)

from pytube import YouTube  # Importing YouTube class from pytube for video handling
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
        
    def download_youtube_video(self, video_id: str, save_dir: str = ".", resolution: str = "720p") -> str:
        """
        Downloads a YouTube video by video ID as an MP4 file.

        Args:
            video_id (str): YouTube video ID.
            save_dir (str): Directory to save the downloaded video.
            resolution (str): Desired resolution (default '720p').

        Returns:
            str: Full file path of the downloaded video.
        """
        video_url = self.MetadataExtractor_obj.construct_video_url(video_id)
        yt = YouTube(video_url)  # Create a YouTube object with the video URL
        # Select the highest available resolution not exceeding the requested one
        stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by(resolution).desc().first()
        
        if not stream:
            raise ValueError(f"No suitable video stream found for {video_url}")

        file_path = stream.download(output_path=save_dir, filename=f"{video_id}.mp4")

        print(f"✅ Downloaded video to: {file_path}")
        return file_path

if __name__ == "__main__":
    API_KEY = load_api_key(parent_dir + '/' + "keys/youtube_data_API_key.txt")  # Load the YouTube Data API key from the specified file
    CharacsExtractor_obj = CharacsExtractor(API_KEY)  # Create an instance of the CharacsExtractor class with the API key
    video_id = "NU_b_14rdL0"  # Replace with an actual video ID for testing
    
    MetadataExtractor_obj = MetadataExtractor(API_KEY)
    duration_sec = MetadataExtractor_obj.get_video_duration_sec(video_id)  # Get the video duration in seconds
    if duration_sec is not None:
        print(f"Video duration for {video_id}: {duration_sec} seconds")
    else:
        print(f"Could not retrieve duration for video {video_id}. Check if the video ID is correct or if the video exists.")
    
    duration_min = CharacsExtractor_obj.get_video_duration_min(video_id)  # Get the video duration
    print(f"Video duration for {video_id}: {duration_min} minutes")  # Print the video duration

    word_count = CharacsExtractor_obj.count_words(video_id)  # Count the words in the video's transcript
    if word_count is not None:
        print(f"Word count for video {video_id}: {word_count}")
    else:
        print(f"No transcript available for video {video_id}.")

    speaking_speed = CharacsExtractor_obj.extract_word_per_minute(video_id)  # Calculate speaking speed
    if speaking_speed is not None:
        print(f"Speaking speed for video {video_id}: {speaking_speed} words per minute")
    else:
        print(f"Could not calculate speaking speed for video {video_id}. Check duration and word count.")

    CharacsExtractor_obj.download_youtube_video(video_id, save_dir=".", resolution="720p")  # Download the video
    
    # Example usage
    # video_path = "example_video.mp4"
    # extractor = CharacteristicsExtractor(video_path)
    # extractor.extract_features()
    # extractor.save_to_csv("new_characs.csv")
    # print("Characteristics extracted and saved to new_characs.csv.")