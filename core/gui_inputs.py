"""
Module: gui_inputs.py
Author: Mohamed Tolba
Date Created: 13-07-2025
Last Updated: 13-07-2025

Description:
    Provides reusable GUI-based input utilities using tkinter. Includes:
    - A single-field input dialog function
    - A multi-field custom input dialog class

    These functions are useful for collecting user input in desktop applications
    where command-line interaction is not preferred.
"""
"""
❌ Note, the tkinter-based GUI dialogs will not work with Streamlit.
❌ The code has not been tested or used, but it is provided as a utility for future use.
"""

import tkinter as tk  # Basic GUI framework
from tkinter import simpledialog  # Easy dialog box support

def get_user_input_gui(field_name: str) -> str:
    """
    Prompts the user for a single input value using a dialog box.

    Args:
        field_name (str): The name of the field to prompt for.

    Returns:
        str: The user-provided value.
    """
    root = tk.Tk()                # Create a root window (needed by tkinter)
    root.withdraw()              # Hide the main window, only show the dialog

    # Prompt the user for input with a title and message
    value = simpledialog.askstring(
        title=f"Input Required - {field_name}", 
        prompt=f"Enter {field_name.replace('_', ' ').title()}:"
    )

    root.destroy()               # Close the root window after the dialog
    return value                 # Return the entered value


def get_multiple_inputs_gui(field_names: list) -> dict:
    """
    Displays a dialog box that collects multiple fields in one window.

    Args:
        field_names (list): List of field names to prompt the user for.

    Returns:
        dict: A dictionary of field names to user-provided values.
    """
    inputs = {}  # Dictionary to store the results

    class MultiInputDialog(simpledialog.Dialog):
        """
        Custom dialog class for collecting multiple user inputs.
        """

        def body(self, master):
            self.entries = {}  # Store entry widgets keyed by field name
            for i, field in enumerate(field_names):
                # Create a label for each field
                tk.Label(master, text=field.replace('_', ' ').title() + ":").grid(row=i, column=0, sticky='e')
                
                # Create an entry box for each field
                entry = tk.Entry(master)
                entry.grid(row=i, column=1)
                self.entries[field] = entry  # Save the entry box

            return list(self.entries.values())[0]  # Focus on the first field

        def apply(self):
            # Save values from each entry box when the dialog is confirmed
            for field, entry in self.entries.items():
                inputs[field] = entry.get()

    root = tk.Tk()         # Create a root window for the dialog
    root.withdraw()       # Hide the main window
    MultiInputDialog(root, title="Enter Required Information")  # Show custom dialog
    root.destroy()        # Close the root window after completion

    return inputs  # Return dictionary of user inputs

if __name__ == "__main__":
    field_list = ["institution_name", "speaker_name", "course_code"]
    user_inputs = get_multiple_inputs_gui(field_list)
    print(f"The inputs are: {user_inputs}")  # Print the collected inputs for testing