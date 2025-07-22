"""
Module: characteristics_core.py
Author: Mohamed Tolba
Date Created: 24-06-2025
Last Updated: 20-07-2025

Description:
    Defines the CharacteristicsExtractor class that performs audio-visual analysis
    on videos to extract features such as speaking speed, slide change frequency,
    caption availability, audio clarity, and tone variability.

    Extracted features are written to new_characs.csv.
    Delete this line # source .venv/bin/activate

To Validate/Tune:
    1. The scene change detection threshold can be adjusted in the ContentDetector.
       We need to define what constitutes a significant scene change.
    2. The tone variability function is not working properly - needs investigation.

To do:
    1. Compare the results for three videos for validation
    2. Consider making the transcript function accepts audio rather than video (more robust and efficient)
       But the challenge will be how to download the video directly if the video-hosting platform is not Youtube
    3. Clean the main function
"""
import sys
import os
import re # Regular expression module for text processing
import cv2
import whisper
import warnings
import yt_dlp

# import moviepy as mp        # For extracting audio from video
# import parselmouth                 # For pitch analysis using Praat
# import numpy as np                 # For numerical operations

from scenedetect import open_video, SceneManager 
# open_video: Recommended function to load video files for analysis in PySceneDetect.
# SceneManager: Orchestrates detection logic and stores detected scene boundaries.
from scenedetect.detectors import ContentDetector 
# ContentDetector: A detection algorithm that identifies significant visual content changes between frames (based on pixel difference)

# Build the absolute path to the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add to Python module search path
sys.path.append(parent_dir)

from core.keys_manager import load_api_key  # Import the function to load the API key
from core.metadata_core import MetadataExtractor

class CharacsExtractor:
    def __init__(self):
        pass

    def download_youtube_video_audio(self, video_id, save_dir="."):
        """
        Download a YouTube video and audio using yt-dlp.   
        """
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        if not video_url:
            print(f"Could not retrieve video URL for video ID: {video_id}")
            return
        try:
            ydl_opts = {
                'outtmpl': f"{save_dir}/%(id)s.%(ext)s", # Save the video with its ID as the filename
                'noplaylist': True,  # Download only the single video, not the playlist
                'quiet': True,  # Suppress output messages
                'format': 'mp4',
            }
            ydl = yt_dlp.YoutubeDL(ydl_opts)
            ydl.download([video_url])
            video_path = save_dir + '/' + f"{video_id}.mp4" # Construct the video path after downloading
            return video_path
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False
    
    def extract_video_duration(self, video_path, format = 'sec'): # Uses OpenCV package. It counts the frames and divides the count by the fps
        # More accurate than getting the video duration from youtube
        cap = cv2.VideoCapture(video_path) # Open the video file using OpenCV's VideoCapture
        fps = cap.get(cv2.CAP_PROP_FPS) # Retrieve the frame rate (frames per second) of the video
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) # Retrieve the total number of frames in the video
        cap.release()  # Close the video file to free up resources
        duration_sec = round(frame_count / fps, 2)
        duration_min = round(duration_sec / 60.0, 2)  # Convert seconds to minutes
        if format == 'sec': return duration_sec
        elif format == 'min': return duration_min
    
    def extract_transcript(self, video_path):
        warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")
        model = whisper.load_model("base")      # Load the small Whisper model
        # Load the pre-trained Whisper ASR (Automatic Speech Recognition) model.
        # "base" specifies the model size – options include "tiny", "base", "small", "medium", and "large".
        # Smaller models are faster but less accurate; "base" balances speed and accuracy for general use.
        # "large" is the most accurate but also the slowest
        # The model will automatically download if not previously cached locally.

        result = model.transcribe(video_path)   # Transcribe audio from video file
        transcript = result['text']             # Extract the text portion of the result
        return transcript                       # Return the extracted transcript
    
    def count_words(self, video_path: str) -> int:
        """
        Count the number of words in the video's transcript.
        """
        video_transcript = self.extract_transcript(video_path)
        if False:
            return None  # Return None if no transcript is available
        
        # Remove bracketed words like [music], [laughter], [applause], etc.
        cleaned_transcript = re.sub(r'\[.*?\]', '', video_transcript)

        # Assuming the transcript is a string of words, we can split it to count words
        words = cleaned_transcript.split()
        return len(words)  # Return the number of words in the transcript
    
    def extract_word_per_minute(self, video_path: str) -> float:
        """
        Calculate the speaking speed in words per minute.
        """
        duration_sec = self.extract_video_duration(video_path, 'sec')
        word_count = self.count_words(video_path)
        
        if duration_sec > 0 and word_count is not None:
            word_count_per_minute = word_count / (duration_sec / 60.0)  # Convert seconds to minutes
            return round(word_count_per_minute, 2)  # Round to 2 decimal places
        else:
            return None        
    
    def count_scene_changes(self, video_path=".", threshold=10.0, offsets = [1,2]):
        """
        Detect and count scene changes in a video using PySceneDetect.
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

        total_duration_sec = self.extract_video_duration(video_path, 'sec')
        # total_duration_sec = self.get_video_duration_sec(video_path)

        # Filter scenes to exclude first & last seconds
        filtered_scenes = []
        skip_start_sec = offsets[0]  # Skipping seconds at the beginning of the video
        skip_end_sec = offsets[1]    # Skipping seconds at the end of the video
        for start_time, end_time in scene_list:
            start_sec = start_time.get_seconds()
            end_sec = end_time.get_seconds()

            if start_sec >= skip_start_sec and end_sec <= (total_duration_sec - skip_end_sec):
                filtered_scenes.append((start_time, end_time))

        # print(filtered_scenes)
        # Return total number of detected scene changes
        return len(filtered_scenes)
    
    def extract_scene_change_per_min(self, video_path=".", threshold=10.0, offsets = [1,2]) -> float:
        """
        Calculate the frequency of scene changes in a video.
        """
        total_duration_sec = self.extract_video_duration(video_path, 'sec')
        total_scene_changes = self.count_scene_changes(video_path, threshold, offsets)
        if total_duration_sec > 0 and total_scene_changes > 0:
            scene_change_freq = total_scene_changes / (total_duration_sec / 60.0)  # Convert seconds to minutes
            return round(scene_change_freq, 2)  # Round to 2 decimal places
        else:
            return None

    # def extract_pitch_features(self, video_path):
    #     video = mp.VideoFileClip(video_path)                      # Load video file
    #     audio = video.audio                                       # Extract audio from video
    #     audio.write_audiofile("temp_audio.wav")                   # Save audio as temporary WAV
 
    #     sound = parselmouth.Sound("temp_audio.wav")               # Load audio file in Praat format
    #     pitch = sound.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=600) # Extract pitch object
    #     # Typical human pitch range:
    #     # Male voice: ~85 Hz to 180 Hz
    #     # Female voice: ~165 Hz to 255 Hz
    #     # Children: can go above 300 Hz
 
    #     pitch_values = pitch.selected_array['frequency']          # Get pitch values (Hz)
    #     pitch_values = pitch_values[pitch_values > 0]             # Remove unvoiced parts (0 Hz)
 
    #     min_pitch = np.min(pitch_values) if len(pitch_values) > 0 else 0            # Minimum pitch (Hz)
    #     max_pitch = np.max(pitch_values) if len(pitch_values) > 0 else 0            # Maximum pitch (Hz)
    #     pitch_variance = np.var(pitch_values) if len(pitch_values) > 0 else 0       # Pitch variance (Hz^2)
 
    #     return {
    #         'min_pitch': min_pitch,
    #         'max_pitch': max_pitch,
    #         'pitch_variance': pitch_variance
    #     }    # Return pitch-related features
 
    def delete_file(self, file_path):
        """
        Deletes a file if it exists.
        """
        if os.path.isfile(file_path):
            os.remove(file_path)
            return True  # File successfully deleted
        else:
            return False  # File did not exist    

if __name__ == "__main__":
    API_KEY = load_api_key(parent_dir + '/' + "keys/youtube_data_API_key.txt")  # Load the YouTube Data API key from the specified file
    CharacsExtractor_obj = CharacsExtractor()  # Create an instance of the CharacsExtractor class with the API key
    MetadataExtractor_obj = MetadataExtractor(API_KEY)

    video_id = "sEE33_YIgys" 
    # video_id = " "
    save_dir = parent_dir + '/temp'
    video_path = CharacsExtractor_obj.download_youtube_video_audio(video_id, save_dir)  # Download the video
    
    video_path = os.path.join(save_dir, f"{video_id}.mp4")  # Construct the path to the downloaded video
    if os.path.exists(video_path):
        print(f"Video downloaded successfully to {video_path}")
    else:
        print(f"Failed to download video {video_id}. Check if the video ID is correct or if the video exists.")
    
    video_url = MetadataExtractor_obj.construct_video_url(video_id)
    video_duration_min = CharacsExtractor_obj.extract_video_duration(video_path, 'min')
    video_words_count = CharacsExtractor_obj.count_words(video_path)
    video_speaking_word_per_minute = CharacsExtractor_obj.extract_word_per_minute(video_path)
    video_scences_count = CharacsExtractor_obj.count_scene_changes(video_path, 30)
    video_scene_change_per_min = CharacsExtractor_obj.extract_scene_change_per_min(video_path, 30)
    
    print(f"video url: {video_url}")
    print(f"video duration min: {video_duration_min}")
    print(f"video words count: {video_words_count}")
    print(f"video speaking rate (wpm): {video_speaking_word_per_minute}")
    print(f"video scenes count: {video_scences_count}")
    print(f"video scene change rate (per min): {video_scene_change_per_min}")

    # print(CharacsExtractor_obj.extract_pitch_features(video_path))

    # transcript = CharacsExtractor_obj.extract_transcript(video_path)
    # print(transcript)

    CharacsExtractor_obj.delete_file(video_path)