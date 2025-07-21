import csv

def create_video_submission_csv(file_path):
    submission_file_fields = ['dataset_tag', 'video_id', 'institution_name', 
                              'speaker_name', 'course_code', 'course_name', 'unit_level',
                              'year', 'video_type', 'subject_area', 'average_percentage_viewed']
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(submission_file_fields)  # Header

def create_new_metadata_csv(file_path):
    metadata_file_fields = ['dataset_tag','video_id','institution_name','speaker_name',
                            'course_code','course_name','unit_level','year','video_type',
                            'subject_area','video_url','title','duration_sec','channel_id',
                            'channel_name','published_at']
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(metadata_file_fields)  # Header

def create_new_metrics_csv(file_path):
    metrics_file_fields = ['dataset_tag','video_id','average_percentage_viewed']
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(metrics_file_fields)  # Header

def create_new_characs_csv(file_path):
    characs_file_fields = ['dataset_tag','video_id','duration_min', 'speaking_speed_wpm', 'scene_change_per_min']
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(characs_file_fields)  # Header