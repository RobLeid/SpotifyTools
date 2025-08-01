"""
Common operations for the Spotify ISRC Finder application.
This module contains reusable logic patterns to reduce code duplication across pages.
"""

import streamlit as st
from typing import Optional
from .auth import get_access_token
from .api_improved import SpotifyAPIClient


def get_authenticated_client() -> Optional[SpotifyAPIClient]:
    """
    Get an authenticated Spotify API client.
    
    Returns:
        SpotifyAPIClient instance if authentication successful, None otherwise
    """
    access_token = get_access_token()
    if not access_token:
        return None
    
    return SpotifyAPIClient(access_token)


