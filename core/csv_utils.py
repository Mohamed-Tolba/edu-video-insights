"""
Module: csv_utils.py
Author: Mohamed Tolba
Date Created: 24-06-2025
Last Updated: 08-07-2025

Description:
    Provides utility functions and classes for consistent reading, writing, and
    managing CSV files throughout the edu-video-insights pipeline.

Tested:
 - Initiation and reading of a CSV file.
 - Finding rows by value in a specific column.
 - saving the CSV file after modifications.
 - removing invalid rows.
 - removing duplicate rows.
 - removing extra columns (unnammed columns).
 - adding new rows.
 - populate row by field.
 - automated sorting by the field in ascending or descending order.
 - remove row(s) by field.
 - clearing all rows in the CSV file, keeping only the header.
 - find the value in a specific field (column) from the row where another field matches a given value

Not Tested / Under Development:
 - Example usage in the main block.
"""

import pandas as pd
import numpy as np
import os

class CSVHandler:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = None

        if os.path.exists(file_path):
            self.df = pd.read_csv(file_path)
        else:
            raise FileNotFoundError(f"The file {file_path} does not exist.")

    def save(self):
        """Save the updated DataFrame to the CSV file."""
        self.df.to_csv(self.file_path, index=False)

    def find_row_by_value(self, match_field: str, match_value: str) -> list:
        """Find all row indices where the target value appears in a specific column."""
        if match_field not in self.df.columns:
            raise ValueError(f"Header '{match_field}' not found in the CSV file.")

        matches = self.df.index[self.df[match_field] == match_value].tolist()
        return matches if matches else None

    def get_cell_value_by_match(self, match_field: str, match_value: str, target_field: str):
        """
        Returns the value in target_field where match_field equals match_value.
        Only the last matching row is considered.
        """
        if match_field not in self.df.columns:
            raise ValueError(f"'{match_field}' column does not exist.")
        if target_field not in self.df.columns:
            raise ValueError(f"'{target_field}' column does not exist.")

        matches = self.df[self.df[match_field] == match_value]

        if matches.empty:
            print(f"⚠️ No match found for {match_field} == '{match_value}'")
            return None

        value = matches.iloc[-1][target_field]  # last match; change to iloc[0] for first
        return value


    def populate_row_by_field(self, match_field: str, match_value: str, row_data: dict) -> bool:
        """
        Finds the row where match_field == match_value and updates it using row_data.
        Only the last matching row is updated.
        """
        if match_field not in self.df.columns:
            raise ValueError(f"'{match_field}' column is missing from the CSV.")

        # Find matching row(s)
        matches = self.find_row_by_value(match_field, match_value)

        if not matches:
            print(f"❌ No row found where {match_field} == '{match_value}'")
            return False

        row_index = matches[-1]  # Update the last match; change to matches[0] to update the first

        for key in self.df.columns:
            if key != match_field:  # Don't overwrite the match field
                self.df.at[row_index, key] = row_data.get(key, np.nan)

        self.save()
        return True

    def add_new_row(self, new_row: dict):
        """Append a new row (dictionary) to the CSV."""
        # Ensure only known columns are used
        filtered_row = {col: new_row.get(col, np.nan) for col in self.df.columns} # Build a row dictionary aligned with DataFrame columns, using np.nan for any missing fields
        new_row_df = pd.DataFrame([filtered_row]) # Convert the dictionary to a DataFrame
        self.df = pd.concat([self.df, new_row_df], ignore_index=True) # Concatenate the new row DataFrame to the existing DataFrame
        self.save()
        
    def remove_invalid_rows(self, required_fields = ["video_id"]):
        """Remove fully empty rows from the CSV."""
        # self.df = self.df.dropna(how='all') # Drop the row only if all columns in that row are NaN
        #required_fields = ["video_id", "institution_name", "speaker_name", "course_code", "unit_level", "week_number", "year", "video_type", "subject_area", "retention_rate", "avg_view_duration", "dataset_tag"]
        
        self.df = self.df.dropna(subset = required_fields) # Drop the row if any of the required fields is NaN
        self.df = self.df.reset_index(drop = True) 
        self.save()

    def remove_repeated_rows(self):
        """Remove duplicate rows (fully repeated)."""
        required_fields = ["video_id"]
        self.df = self.df.drop_duplicates(subset = required_fields, keep='last')  # keep the last occurrence of each duplicate
        self.df = self.df.reset_index(drop = True)  # Reset the index after dropping duplicates
        self.save()

    def remove_row_by_field(self, match_field: str, match_value: str):
        """
        Removes all rows where the given match_field equals the specified match_value.
        """
        if match_field not in self.df.columns:
            raise ValueError(f"'{match_field}' column does not exist in the CSV.")

        # Identify rows to remove
        condition = self.df[match_field] == match_value # Create a boolean mask where the field matches the given value
        num_removed = condition.sum() # Count how many rows match the condition (True values)
        # print(f"Condition: {condition}")
        if num_removed == 0:
            print(f"⚠️ No rows found where {match_field} == '{match_value}'")
            return

        # Filter out matching rows
        self.df = self.df[~condition] # Keep only rows where the condition is False (i.e. remove the matched rows)
        self.save()

    def remove_extra_columns(self):
        """Remove unnamed or empty columns."""
        self.df = self.df.loc[:, ~self.df.columns.str.contains('^Unnamed')] # Remove columns that start with 'Unnamed'
        # self.df = self.df.dropna(axis=1, how='all') # Drop columns that are completely empty
        self.save()

    def clean_csv(self, required_fields = ["video_id"]):
        """Clean the CSV by removing empty rows, columns, and duplicates."""
        self.remove_invalid_rows(required_fields) # Drop the row if any of the required fields is NaN
        self.remove_repeated_rows() # Drop the repeated rows except the last occurrence
        self.remove_extra_columns() # Drop the unnamed columns
    
    def sort_by_field(self, field_name: str, ascending: bool = True):
        """
        Sort the DataFrame by the specified field.

        Parameters:
        - field_name: the column to sort by
        - ascending: sort order (default True = ascending, False = descending)
        """
        if field_name not in self.df.columns:
            raise ValueError(f"'{field_name}' is not a valid column in the CSV.")

        self.df = self.df.sort_values(by = field_name, ascending = ascending, na_position = 'last')
        self.df = self.df.reset_index(drop = True)  # Reset the index after sorting
        self.save()

    def clear_all_rows(self, msg ="🧹 All data removed."):
        """
        Delete all rows in the CSV file, keeping only the header.
        """
        # Keep only the column headers, drop all data rows
        self.df = self.df.iloc[0:0]
        self.save()
        print(msg)
 
        
if __name__ == "__main__":
    # Example usage
    file_path = '../temp/video_submission.csv'  # Replace with your CSV file path
    handler = CSVHandler(file_path)
    # rows = handler.find_row_by_value("video_id","GxmDYSewCws")  # Example usage of the method

    # print(handler.df)
    print(handler.df.columns[0])
    # try:
    #     # Example: Append a new row
    #     video_id, video_type, subject_area, retention_rate,avg_view_duration = "Test_value_1", "lecture", "Computer Science", 0.75, 120.5
    #     new_row = {
    #         "video_id": "Test_value_1",
    #         "institution_name": "Test Institution",
    #         "video_type": "lecture",
    #         "subject_area": "Computer Science",
    #         "retention_rate": 0.75,
    #         "avg_view_duration": 120.5}
    #     handler.add_new_row(new_row)
    #     print(handler.df)
    # except Exception as e:
    #     print(f"❌ An error occurred: {e}")
# 
    # handler.clean_csv()  # Clean the CSV by removing empty rows, columns, and duplicates
    # print(handler.df)
# 
    # video_id, video_type, subject_area, retention_rate,avg_view_duration = "Test_value_1", "lecture", "Computer Science", 0.75, 300
    # modified_row_data = {
    #         "video_id": "Test_value_1",
    #         "institution_name": "Test Institution2",
    #         "video_type": "lecture",
    #         "subject_area": "Computer Science",
    #         "retention_rate": 0.75,
    #         "avg_view_duration": 420}
# 
    # handler.populate_row_by_field("video_id", "Test_value_1", modified_row_data)
    # handler.clean_csv()  # Clean the CSV by removing empty rows, columns, and duplicates
    # print(handler.df)
# 
    # handler.sort_by_field("dataset_tag", ascending = False)  # Sort by the field in ascending or descending order
    # print(handler.df)
# 
    # handler.remove_row_by_field("video_id", "Test_value_1")  # Remove the row where video_id == "Test_value_1"
    # print(handler.df)

    # handler.clear_all_rows()  # Clear all rows in the CSV file, keeping only the header
    # print(handler.df)  # Should print an empty DataFrame with only the header