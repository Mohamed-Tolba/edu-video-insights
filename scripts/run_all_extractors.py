"""
Module: run_all_extractors.py
Author: Mohamed Tolba
Date Created: 24-06-2025
Last Updated: 08-07-2025

Description:
    Master orchestration script that:
    - Loads video_submission.csv
    - Runs metadata, characteristics, and metrics extractors sequentially
    - Optionally triggers validation and merging logic

    Acts as a single entry point for dataset population.

To test/develop:
    - Make the file load the video_submission.csv file and forward the video IDs (or all dataframe?!) to the extractors.
"""

import subprocess

subprocess.run(["python", "extract_metadata.py"])
subprocess.run(["python", "extract_metrics.py"])
