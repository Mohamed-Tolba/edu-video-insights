import streamlit as st
import sys, os, requests
from utilities import load_css
load_css()
# --------------------------------------------------------------------------------------------------------
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) # Build the absolute path to the parent directory
sys.path.append(parent_dir) # Add to Python module search path
# Load core modules
from core.keys_manager import load_api_key  # Import the function to load the API key
from core.metadata_core import MetadataExtractor  # Import the MetadataExtractor class from the core module
# Load gui functions
from gui.sidebar import add_sidebar  # Import the function to add the sidebar
add_sidebar(parent_dir)             # Add the sidebar to the Streamlit app
from gui.tabs.page1_tabs import *  # Import the function to render the first tab
from gui.initialise_stat import *
# --------------------------------------------------------------------------------------------------------
# Load the YouTube Data API key ---------------------------------------------------------------------------
API_KEY = load_api_key(parent_dir + '/' + "keys/youtube_data_API_key.txt")  # Load the YouTube Data API key from the specified file
MetadataExtractor_obj = MetadataExtractor(API_KEY)
# --------------------------------------------------------------------------------------------------------

# Set the page configuration for the Streamlit app -------------------------------------------------------
st.set_page_config(
    page_title="Data Collection",
    page_icon=":book:",
    layout="wide",  # Use 'wide' layout for more space
    initial_sidebar_state="expanded"  # Start with the sidebar expanded
)
# Page title
st.header("🗄️ Data Collection")
# st.header
# --------------------------------------------------------------------------------------------------------

# Creating a user ID -------------------------------------------------------------------------------------
id_file_path = parent_dir + '/' + 'temp/user_id.txt'
user_id_string = ''
# Read from file
with open(id_file_path, 'r') as f:
    user_id_string = f.read() 
user_id = user_id_string
# -------------------------------------------------------------------------------------------------------- 


tab1, tab2, tab3, tab4 = st.tabs(["📤 Create new data file", "🛠️ Extract/Import metadata", "🛠️ Extract/Import metrics", "🔍 Extract/Import characteristics"])
with tab1:   
    if st.button("Upload New Data"):
        st.session_state['button1_1_1'] = 1
        user_id_int = int(user_id_string)
        user_id_int = user_id_int + 1 
        user_id_string = f"{user_id_int:04d}"  # Output: '000012'
        # Write to file
        with open(id_file_path, 'w') as f:
            f.write(user_id_string)
        user_id = user_id_string 
    if st.session_state['button1_1_1'] == 1:
        import_new_data(MetadataExtractor_obj, parent_dir, user_id)  # Render the first tab for uploading CSV files and preparing submission data

with tab2:
    # # create two columns
    # col1, col2 = st.columns([1, 1])
    # with col1:
    #     if st.button("Extract Metadata from Youtube"):
    #         st.session_state['button1_2_1'] = 1 
    #         st.session_state['button1_2_2'] = 0      
    # with col2:
    #     if st.button("Extract Metrics"):
    #         st.session_state['button1_2_1'] = 0
    #         st.session_state['button1_2_2'] = 1
    
    if st.button("Extract Metadata from Youtube"):
        st.session_state['button1_2_1'] = 1

    
    if st.session_state['button1_2_1'] == 1:
        user_API_KEY = st.text_input("Enter your YouTube Data API Key ([How to get a YouTube API key](https://developers.google.com/youtube/v3/getting-started)):", placeholder="AIza…", type="password")
        if user_API_KEY:
            url = "https://www.googleapis.com/youtube/v3/videos"
            params = {"id": "dQw4w9WgXcQ", "key": user_API_KEY, "part": "id"}
            resp = requests.get(url, params=params)
            if resp.status_code == 200:
                st.success("✅ API key is valid!")
                st.info("Extracting metadata for each video...")
                extract_metadata(parent_dir, user_API_KEY, user_id)  # Render the first tab for uploading CSV files and preparing submission data
            else:
                error = resp.json().get("error", {}).get("message", resp.text)
                st.error(f"❌ Invalid key: {error}")
with tab3:
    if st.button("Extract Metrics"):
        st.session_state['button1_3_1'] = 1
    if st.session_state['button1_3_1'] == 1: 
        st.info("Extracting metrics for each video...")
        extract_metrics(parent_dir, user_id)
with tab4:
    st.info("🚧 Under Construction 🚧")