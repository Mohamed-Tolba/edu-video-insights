## 📄 docs/SUBMISSION_SCHEMA.md

### `video_submission.csv` Field Descriptions

This file is the primary point of manual input. It contains data that cannot be automatically extracted and must be supplied by video owners or contributors.

| Column             | Required? | Description |
|--------------------|-----------|-------------|
| `video_id`         | ✅        | YouTube video ID (e.g., `abc123XYZ`) |
| `institution_name` | ✅        | Full name of the university or organisation |
| `speaker_name`     | ✅        | Name of the lecturer or presenter |
| `course_code`      | ✅        | University course code (e.g., `TRC2201`) |
| `unit_level`       | ✅        | e.g., `1st year`, `2nd year` |
| `week_number`      | ✅        | e.g., `Week 1`, `Week 5` |
| `video_type`       | ✅        | e.g., `lecture`, `tutorial`, `demo` |
| `subject_area`     | ✅        | Academic field or discipline |
| `retention_rate`   | Optional  | Average percentage viewed (if owned) |
| `avg_view_duration`| Optional  | Average view time in seconds (if owned) |
| `dataset_tag`      | Optional  | Suggested tag (e.g., `monash_trc2201_s2-2024`) |
| `notes`            | Optional  | Any additional remarks |
