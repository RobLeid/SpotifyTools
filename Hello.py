import streamlit as st
from utils.ui_components import display_spotify_attribution

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to Robert's Spotify ISRC Finder!👋")

display_spotify_attribution()

st.markdown(
    """
    ---
    ## 👈 Please select a page on the sidebar 
    """
)
