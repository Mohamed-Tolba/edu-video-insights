"""
Module: metadata_core.py
Author: Mohamed Tolba
Date Created: 24-06-2025
Last Updated: 24-06-2025

Description:
    Defines the MetadataExtractor class responsible for retrieving metadata from the
    YouTube Data API, including video title, duration, channel information, and upload date.

    This class is used to process entries from video_submission.csv and populate
    new_metadata.csv with structured metadata for each video.
"""

from isodate import parse_duration  # To parse ISO 8601 durations
from googleapiclient.discovery import build  # Import the Google API client library
from youtube_transcript_api import YouTubeTranscriptApi # Library to retrieve captions (transcripts) from YouTube videos (manual or auto-generated).
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable, CouldNotRetrieveTranscript

import re  # Import regular expressions module for parsing the video ID
# The re module in Python stands for "regular expressions". 
# It provides powerful tools for searching, matching, extracting, or replacing text patterns in strings.

class MetadataExtractor:
    """
    A class to represent a VBL_Data (Video-based Learning_Data) object.
    This is a placeholder class that can be extended with specific attributes and methods.
    """
    def __init__(self, api_key):
        """
        Initialize the VBL_Data object with a YouTube Data API key.
        Parameters:
        api_key (str): The YouTube Data API key to authenticate requests.
        """
        self.youtube_data_api_key = api_key
        self.youtube_client = build("youtube", "v3", developerKey = api_key)
        # Build the YouTube service client using the API key
        # The build function creates a resource object for interacting with the YouTube Data API.
        # It takes the API name ("youtube"), version ("v3"), and the developer key (API key) as parameters.
        # The developerKey parameter is used to authenticate requests to the YouTube Data API.
        # The YouTube Data API v3 allows developers to access public YouTube content, including videos, playlists, and channels.
        # The youtube variable now holds a client that can be used to make API requests.
    
    # Function to extract the video ID from a YouTube URL
    def extract_video_id(self, url: str) -> str:
        """
        Extract the video ID from a YouTube URL.
        Parameters:
        url (str): The YouTube video URL.
        Returns:
        str: The extracted video ID or None if not found.
        """
        pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
        # This regex (short for regular expression) matches the video ID in standard YouTube URLs
        # The prefix r before a string in Python — like r"some text" — stands for raw string.
        # It tells Python not to treat backslashes \ as escape characters.
        # The regex pattern looks for 'v=' or a slash followed by 11 alphanumeric characters, underscores, or hyphens.
        # The video ID is typically 11 characters long and can include letters, numbers, underscores, and hyphens.

        # The pattern is designed to handle various YouTube URL formats, including:
        # - Standard URLs: https://www.youtube.com/watch?v=VIDEO_ID
        # - Shortened URLs: https://youtu.be/VIDEO_ID
        # - Embedded URLs: https://www.youtube.com/embed/VIDEO_ID
        # - Playlist URLs: https://www.youtube.com/playlist?list=PLAYLIST_ID&v=VIDEO_ID

        match = re.search(pattern, url)
        # The re.search function searches the url string for the first location where the regular expression pattern produces a match.
        # If a match is found, it returns a match object; otherwise, it returns None.
        # The match object contains information about the search result, including the matched text.

        return match.group(1) if match else None
        # The group(1) method retrieves the first captured group from the match, which is the video ID.
        # If no match is found, it returns None.

    def construct_video_url(self, video_id):
        """
        Constructs a full YouTube video URL from a given video ID.

        Parameters:
            video_id (str): The 11-character YouTube video ID.

        Returns:
            str: Full YouTube video URL.
        """
        return f"https://www.youtube.com/watch?v={video_id}"

    # Functions to get video information using the YouTube Data API
    def get_video_data(self, video_id: str) -> dict: # Service function
        """
        Get video data using the YouTube Data API.
        Parameters:
        video_id (str): The ID of the YouTube video.
        Returns:
        dict: A dictionary containing video data such as snippet, content details, and statistics.
        """
        request = self.youtube_client.videos().list(
        part = "snippet, contentDetails, statistics",  # Specify what data to retrieve
        id = video_id  # Provide the video ID
        )
        response = request.execute()  # Execute the API call
        return response
    
    def get_video_published_at(self, video_id: str) -> str:
        """
        Get the published date and time of the video using its ID.
        """
        response = self.get_video_data(video_id)
        if not response["items"]:
            return "Video not found."
        return response["items"][0]["snippet"]["publishedAt"]
    
    def get_video_channel_id(self, video_id: str) -> str:
        """
        Get the channel ID of the video using its ID.
        """
        response = self.get_video_data(video_id)
        if not response["items"]:
            return "Video not found."
        return response["items"][0]["snippet"]["channelId"]
    
    def get_video_title(self, video_id: str) -> str:
        """
        Get the title of the video using its ID.
        """
        response = self.get_video_data(video_id)
        if not response["items"]:
            return "Video not found."
        return response["items"][0]["snippet"]["title"]
    
    def get_video_description(self, video_id: str) -> str:
        """
        Get the description of the video using its ID.
        """
        response = self.get_video_data(video_id)
        if not response["items"]:
            return "Video not found."
        return response["items"][0]["snippet"]["description"]
    
    def get_video_channel_name(self, video_id: str) -> str:
        """
        Get the channel title of the video using its ID.
        """
        response = self.get_video_data(video_id)
        if not response["items"]:
            return "Video not found."
        return response["items"][0]["snippet"]["channelTitle"]
    
    def get_video_duration_sec(self, video_id: str) -> str:
        """
        Get the duration of the video using its ID.
        """
        response = self.get_video_data(video_id)
        if not response["items"]:
            return "Video not found."
        duration_iso = response["items"][0]["contentDetails"]["duration"] # Video duration (ISO 8601 format)
        
        # Convert ISO 8601 duration to a timedelta object
        duration = parse_duration(duration_iso)
        total_seconds = duration.total_seconds()
        # The total_seconds() method returns the total duration in seconds as a float.
        return total_seconds
    
    def get_video_caption_status(self, video_id: str) -> float:
        """
        Check if captions are available for the video using its ID.
        """
        response = self.get_video_data(video_id)
        if not response["items"]:
            return "Video not found."
        return response["items"][0]["contentDetails"]["caption"]
    
    def get_video_views(self, video_id: str) -> str:
        """
        Get the view count of the video using its ID.
        """
        response = self.get_video_data(video_id)
        if not response["items"]:
            return "Video not found."
        return response["items"][0]["statistics"].get("viewCount", "No view count available")
    
    def get_video_likes(self, video_id: str) -> str:
        """
        Get the like count of the video using its ID.
        """
        response = self.get_video_data(video_id)
        if not response["items"]:
            return "Video not found."
        return response["items"][0]["statistics"].get("likeCount", "No like count available")
    
    def get_video_comments(self, video_id: str) -> str:
        """
        Get the comment count of the video using its ID.
        """
        response = self.get_video_data(video_id)
        if not response["items"]:
            return "Video not found."
        return response["items"][0]["statistics"].get("commentCount", "No comment count available")
    
    def get_channel_subscription_count(self, channel_id: str) -> str:
        """
        Get the subscription count of the channel using its ID
        """
        request = self.youtube_client.channels().list(
            part="statistics",
            id=channel_id
        )
        response = request.execute()
        if not response["items"]:
            return "Channel not found."
        return response["items"][0]["statistics"].get("subscriberCount", "No subscriber count available")

    # Functions to get the timstamped video transcript
    def format_timestamp(self, total_seconds): # Service function
        # This function is to serve the get_video_transcript function
        minutes = int(total_seconds // 60)
        sec = int(total_seconds % 60)
        return f"[{minutes:02}:{sec:02}]"
    
    def get_video_transcript(self, video_id: str):
        try:
            transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'en-US'])
            full_text = ''
            for entry in transcript_data:
                timestamp = self.format_timestamp(entry['start'])
                text = entry['text']
                full_text += f"{timestamp} {text}\n"
            return full_text
        except TranscriptsDisabled:
            print("❌ Captions are disabled for this video.")
            return
        except NoTranscriptFound:
            print("❌ No English transcript found.")
            return
        except VideoUnavailable:
            print("❌ This video is unavailable or private.")
            return
        except CouldNotRetrieveTranscript:
            print("❌ Could not retrieve transcript (possibly geo-blocked or network error).")
            return
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return

    def new_fun(self):
        """
        A placeholder function that can be extended with specific functionality.
        """
        pass

if __name__ == "__main__":
    # Example usage of the VBL_Data class
    API_KEY = "AIzaSyAItxG2Mye_NlmofGrmpX50pB-g6txm3Kw"  # Alex's YouTube Data API key
    vbl_data_obj = MetadataExtractor(API_KEY)

    video_url = input("Enter YouTube video URL: ")  # Ask user to enter a YouTube video URL
    video_id = vbl_data_obj.extract_video_id(video_url)  # Call the function to fetch and display video info
    video_url_constructed = vbl_data_obj.construct_video_url(video_id)  # Construct the full video URL from the video ID
    video_title = vbl_data_obj.get_video_title(video_id)  # Get the video title using the video ID
    video_published_at = vbl_data_obj.get_video_published_at(video_id)  # Get the published date and time of the video
    video_channel_id = vbl_data_obj.get_video_channel_id(video_id)  # Get the channel ID of the video
    video_description = vbl_data_obj.get_video_description(video_id)  # Get the video description
    video_channel_name = vbl_data_obj.get_video_channel_name(video_id)  # Get the channel title of the video
    video_duration = vbl_data_obj.get_video_duration_sec(video_id)  # Get the video duration
    video_caption_status = vbl_data_obj.get_video_caption_status(video_id)  # Check if captions are available
    video_views = vbl_data_obj.get_video_views(video_id)  # Get the view count of the video
    video_likes = vbl_data_obj.get_video_likes(video_id)  # Get the like count of the video
    video_comments = vbl_data_obj.get_video_comments(video_id)  # Get the comment count of the video
    channel_subscription_count = vbl_data_obj.get_channel_subscription_count(video_channel_id)
    video_transcript = vbl_data_obj.get_video_transcript(video_id)
    
    print(f"Video URL: {video_url_constructed}")  # Print the full video URL
    print(f"Video ID: {video_id}")  # Print the video ID
    print(f"Video Title: {video_title}")  # Print the video title
    print(f"Video Published At: {video_published_at}")  # Print the published date and time of the video
    print(f"Video Channel ID: {video_channel_id}")  # Print the channel ID of the video
    # print(f"Video Description: {video_description}")  # Print the video description
    print(f"Video Channel Title: {video_channel_name}")  # Print the channel title of the video
    print(f"Video Duration: {video_duration} s")  # Print the video duration
    print(f"Video Captions Available: {'Yes' if video_caption_status else 'No'}")  # Print caption availability
    print(f"Video Views: {video_views}")  # Print the view count of the video
    print(f"Video Likes: {video_likes}")  # Print the like count of the video
    print(f"Video Comments: {video_comments}")  # Print the comment count of the video
    print(f"Channel Subscription Count: {channel_subscription_count} subscribers") # Print the subscribers count of the channel
    # print(f"Video Transcript: {video_transcript}")