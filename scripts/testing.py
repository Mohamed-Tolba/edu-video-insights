import sys
import os

# Build the absolute path to the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add to Python module search path
sys.path.append(parent_dir)

id_file_path = parent_dir + '/' + 'temp/user_id.txt'

user_id_string = ''

# Read from file
with open(id_file_path, 'r') as f:
    user_id_string = f.read()

user_id_int = int(user_id_string)
user_id_int = user_id_int + 1 
user_id_string = f"{user_id_int:06d}"  # Output: '000012'

# Write to file
with open(id_file_path, 'w') as f:
    f.write(user_id_string)