import streamlit as st

def load_css(parent_dir):
    style_file_path = parent_dir + '/' + 'gui/style.css'
    with open(style_file_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)