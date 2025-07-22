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
"""
import sys
import os
import re # Regular expression module for text processing
import cv2
# import moviepy as mp
# import numpy as np
# import scipy.io.wavfile as wav
# from scipy.signal import medfilt
# import matplotlib.pyplot as plt

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

    def get_video_duration_sec(self, video_path): # Uses OpenCV package. It counts the frames and divides the count by the fps
        # More accurate than extracting the video duration from youtube
        cap = cv2.VideoCapture(video_path) # Open the video file using OpenCV's VideoCapture
        fps = cap.get(cv2.CAP_PROP_FPS) # Retrieve the frame rate (frames per second) of the video
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) # Retrieve the total number of frames in the video
        cap.release()  # Close the video file to free up resources
        duration_sec = frame_count / fps
        return duration_sec

    def download_youtube_video_audio(self, video_id, save_path="."):
        """Download a YouTube video and audio using yt-dlp.   
        Args:
            video_id (str): The YouTube video ID.
            save_path (str): The directory where the video will be saved.
        """
        video_url = self.MetadataExtractor_obj.construct_video_url(video_id)
        if not video_url:
            print(f"Could not retrieve video URL for video ID: {video_id}")
            return
        try:
            ydl_opts = {
                'outtmpl': f"{save_path}/%(id)s.%(ext)s", # Save the video with its ID as the filename
                'noplaylist': True,  # Download only the single video, not the playlist
                'quiet': True,  # Suppress output messages
                'format': 'mp4',
            }
            ydl = yt_dlp.YoutubeDL(ydl_opts)
            ydl.download([video_url])
            return True
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False

    def count_scene_changes(self, video_id, video_path=".", threshold=10.0):
        """Detect and count scene changes in a video using PySceneDetect.
        Args:
            video_path (str): The path to the video file.
            threshold: determines the degree of sensitivity to the change in frames pixels.
                       The lower the threshold is, the more sensitive the detection becomes.
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
        # total_duration_sec = self.get_video_duration_sec(video_path)

        # Filter scenes to exclude first & last seconds
        filtered_scenes = []
        skip_start_sec = 1  # Skip the first second
        skip_end_sec = 2    # Skip the last second
        for start_time, end_time in scene_list:
            start_sec = start_time.get_seconds()
            end_sec = end_time.get_seconds()

            if start_sec >= skip_start_sec and end_sec <= (total_duration_sec - skip_end_sec):
                filtered_scenes.append((start_time, end_time))

        print(filtered_scenes)
        # Return total number of detected scene changes
        return len(filtered_scenes)
    
    def extract_scene_change_freq(self, video_id, video_path=".", threshold=10.0) -> float:
        """
        Calculate the frequency of scene changes in a video.
        """
        total_duration_sec = self.MetadataExtractor_obj.get_video_duration_sec(video_id)
        # total_duration_sec = self.get_video_duration_sec(video_path)
        total_scene_changes = self.count_scene_changes(video_id, video_path, threshold)
        if total_duration_sec > 0 and total_scene_changes > 0:
            scene_change_freq = total_scene_changes / (total_duration_sec / 60.0)  # Convert seconds to minutes
            return round(scene_change_freq, 2)  # Round to 2 decimal places
        else:
            return None

    def delete_file(self, file_path):
        """
        Deletes a file if it exists.

        Parameters:
            file_path (str): Full path to the file you want to delete.

        Returns:
            bool: True if file was deleted, False if file did not exist.
        """
        if os.path.isfile(file_path):
            os.remove(file_path)
            return True  # File successfully deleted
        else:
            return False  # File did not exist    

if __name__ == "__main__":
    API_KEY = load_api_key(parent_dir + '/' + "keys/youtube_data_API_key.txt")  # Load the YouTube Data API key from the specified file
    CharacsExtractor_obj = CharacsExtractor(API_KEY)  # Create an instance of the CharacsExtractor class with the API key
    video_id = "sEE33_YIgys"  # Replace with an actual video ID for testing
    # video_id = "sEE33_YIgys" # This video has some scene transitions
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
    CharacsExtractor_obj.download_youtube_video_audio(video_id, save_dir)  # Download the video
    # CharacsExtractor_obj.download_video_only(video_id, save_dir)
    # CharacsExtractor_obj.download_audio_only(video_id, save_dir)
    video_path = os.path.join(save_dir, f"{video_id}.mp4")  # Construct the path to the downloaded video
    temp_audio_path = os.path.join(save_dir, f"{video_id}.wav")
    if os.path.exists(video_path):
        print(f"Video downloaded successfully to {video_path}")
    else:
        print(f"Failed to download video {video_id}. Check if the video ID is correct or if the video exists.")
    
    # CharacsExtractor_obj.delete_file(video_path)
    
   
    # Detect scene changes in the downloaded video
    # total_scene_changes = CharacsExtractor_obj.count_scene_changes(video_id, video_path, threshold = 10.0)  # Set the threshold for scene change detection
    # print(f"Total scene changes detected in {video_path}: {total_scene_changes}")
    # scene_change_freq = CharacsExtractor_obj.extract_scene_change_freq(video_id, video_path, threshold = 10.0)  # Calculate scene change frequency
    # if scene_change_freq is not None:
    #     print(f"Scene change frequency for video {video_id}: {scene_change_freq} changes per minute")   
    
    # print(CharacsExtractor_obj.get_video_duration_sec(video_path))

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


"""
def download_video_only(self, video_id, save_path="."):
        
        Downloads video-only (without audio) from a YouTube video using yt_dlp.

        Args:
            video_id (str): The YouTube video ID.
            save_path (str): The directory where the video will be saved.
        
        video_url = self.MetadataExtractor_obj.construct_video_url(video_id)
        if not video_url:
            print(f"Could not retrieve video URL for video ID: {video_id}")
            return
        try:
            ydl_opts = {
                'outtmpl': f"{save_path}/%(id)s.%(ext)s", # Save the video with its ID as the filename
                'noplaylist': True,  # Download only the single video, not the playlist
                'quiet': True,  # Suppress output messages
                'format': 'bv*'   # 'bv*' = best video-only stream (without audio)
            }
            ydl = yt_dlp.YoutubeDL(ydl_opts)
            ydl.download([video_url])
            return True
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False
    
    def download_audio_only(self, video_id, save_path="."):
        
        Downloads audio-only (without video) from a YouTube video using yt_dlp.

        Args:
            video_id (str): The YouTube video ID.
            save_path (str): The directory where the video will be saved.
        
        video_url = self.MetadataExtractor_obj.construct_video_url(video_id)
        if not video_url:
            print(f"Could not retrieve video URL for video ID: {video_id}")
            return
        try:
            ydl_opts = {
                'outtmpl': f"{save_path}/%(id)s.%(ext)s", # Save the video with its ID as the filename
                'noplaylist': True,  # Download only the single video, not the playlist
                'quiet': True,  # Suppress output messages
                'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',   # You can change to 'wav', 'm4a', etc.
                'preferredquality': '192',
                }],
                'format': 'bestaudio/best'
            }
            ydl = yt_dlp.YoutubeDL(ydl_opts)
            ydl.download([video_url])
            return True
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False

def detect_tone_variability_from_video(self, video_path, temp_audio_path="temp_audio.wav"):
        # Not working properly
        Estimates tone variability (in Hz) from an MP4 video file.
        Audio is extracted and analysed for pitch variability using zero-crossing rate.

        Parameters:
            video_path (str): Path to the input MP4 video file.
            temp_audio_path (str): Path to save temporary extracted audio.

        Returns:
            float: Tone variability (standard deviation of pitch) in Hz.
        
        # Step 1: Extract audio from the video using moviepy
        video_clip = mp.VideoFileClip(video_path)                           # Load video file
        video_clip.audio.write_audiofile(temp_audio_path)                   # Save extracted audio as WAV

        # Step 2: Load the extracted audio using scipy
        sr, audio = wav.read(temp_audio_path)                               # sr = sample rate, audio = audio data array

        # Step 3: Convert stereo to mono if needed
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)                                  # Average channels to get mono audio

        # Step 4: Apply pre-emphasis filter to enhance high frequencies
        pre_emphasis = 0.97
        audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])  # Simple high-pass filtering

        # Step 5: Define frame size and hop size for short-time analysis
        frame_length = int(0.03 * sr)                                       # 30 milliseconds frame length
        hop_length = int(0.015 * sr)                                        # 15 milliseconds hop size (50% overlap)

        # Step 6: Compute short-time energy for each frame (used for voice activity detection)
        frame_energy = [
            np.sum(np.square(audio[i:i+frame_length]))                      # Sum of squared samples = energy
            for i in range(0, len(audio) - frame_length, hop_length)
        ]

        # Step 7: Estimate pitch using zero-crossing rate in voiced frames
        pitches = []    # List to store estimated pitch values
        times = []  # Store time (in seconds) for each estimated pitch

        for i, start in enumerate(range(0, len(audio) - frame_length, hop_length)):
            frame = audio[start:start + frame_length]                       # Extract current frame
            energy = frame_energy[i]                                        # Get frame energy

            # Simple voice activity detection (ignore silent frames)
            if energy > 0.01 * np.max(frame_energy):
                zero_crossings = np.sum(np.abs(np.diff(np.sign(frame)))) / 2   # Count zero crossings
                estimated_pitch = (zero_crossings * sr) / (2.0 * frame_length) # Estimate pitch frequency (Hz)

                # Keep only pitch values within typical human speech range
                if 75 < estimated_pitch < 500:
                    time_in_sec = start / sr
                    pitches.append(estimated_pitch)
                    times.append(time_in_sec)

        # Step 8: Compute tone variability from pitch contour
        if len(pitches) > 0:
            pitches_smoothed = medfilt(pitches, kernel_size=5)              # Apply median filtering to smooth pitch contour
            tone_variability_hz = np.std(pitches_smoothed)                  # Compute standard deviation of pitch (Hz)
        else:
            tone_variability_hz = 0.0                                       # Handle case of no voiced frames detected

        # Step 9: Plotting pitch contour
        plt.figure(figsize=(10, 4))
        plt.plot(times, pitches_smoothed, label="Pitch Contour")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Pitch (Hz)")
        plt.title("Pitch Contour Over Time")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
        # Step 10: Delete the temporary audio file to clean up
        # os.remove(temp_audio_path)

        # Step 11: Return tone variability rounded to 3 decimal places
        print(tone_variability_hz)
        # Monotone: ~0–10 Hz.
        # Normal: ~10–40 Hz.
        # Expressive: >40 Hz.
        return round(tone_variability_hz, 3)

"""