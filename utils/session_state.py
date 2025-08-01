"""
Session state management utilities for the Spotify ISRC Finder application.

This module provides centralized functions to manage Streamlit session state
across different pages, ensuring UI persistence after download button clicks.
"""

import streamlit as st
from typing import Any, Dict, Optional, Tuple

def _get_session_key(page_name: str, key_type: str) -> str:
    """
    Generate a consistent session state key for a given page and data type.
    
    Args:
        page_name: Name of the page (e.g., 'tracks', 'albums', 'artist_catalog')
        key_type: Type of data (e.g., 'search_results', 'display_data', 'search_completed')
    
    Returns:
        Formatted session state key
    """
    return f"{page_name}_{key_type}"


def init_page_session_state(page_name: str) -> None:
    """
    Initialize session state variables for a specific page.
    
    This function ensures all necessary session state variables exist for a page,
    preventing KeyError exceptions and providing consistent initialization.
    
    Args:
        page_name: Name of the page to initialize session state for
    """
    # Initialize search results storage
    results_key = _get_session_key(page_name, "search_results")
    if results_key not in st.session_state:
        st.session_state[results_key] = None
    
    # Initialize display data storage
    display_key = _get_session_key(page_name, "display_data")
    if display_key not in st.session_state:
        st.session_state[display_key] = None
    
    # Initialize search completion flag
    completed_key = _get_session_key(page_name, "search_completed")
    if completed_key not in st.session_state:
        st.session_state[completed_key] = False
    
    # Initialize metadata storage (for additional page-specific data)
    metadata_key = _get_session_key(page_name, "metadata")
    if metadata_key not in st.session_state:
        st.session_state[metadata_key] = {}


def store_search_results(
    page_name: str, 
    results: Any, 
    display_data: Optional[Any] = None, 
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Store search results and related data in session state.
    
    This function updates all relevant session state variables for a page after
    a successful search operation, ensuring data persistence across reruns.
    
    Args:
        page_name: Name of the page storing results
        results: Main search results (typically a DataFrame or list of DataFrames)
        display_data: Additional data needed for UI display (e.g., formatted sections)
        metadata: Optional metadata about the search (e.g., artist name, search parameters)
    """
    # Store search results
    results_key = _get_session_key(page_name, "search_results")
    st.session_state[results_key] = results
    
    # Store display data
    display_key = _get_session_key(page_name, "display_data")
    st.session_state[display_key] = display_data
    
    # Mark search as completed
    completed_key = _get_session_key(page_name, "search_completed")
    st.session_state[completed_key] = True
    
    # Store or update metadata
    if metadata:
        metadata_key = _get_session_key(page_name, "metadata")
        if metadata_key in st.session_state:
            st.session_state[metadata_key].update(metadata)
        else:
            st.session_state[metadata_key] = metadata


def get_search_results(page_name: str) -> Tuple[Any, Any, Dict[str, Any], bool]:
    """
    Retrieve stored search results and related data from session state.
    
    This function provides a consistent way to access stored search data,
    returning None values if no search has been completed yet.
    
    Args:
        page_name: Name of the page to retrieve results for
    
    Returns:
        Tuple containing:
        - results: Stored search results (or None)
        - display_data: Stored display data (or None)
        - metadata: Stored metadata dictionary (or empty dict)
        - search_completed: Boolean indicating if a search has been completed
    """
    # Initialize session state if needed
    init_page_session_state(page_name)
    
    # Retrieve all stored data
    results = st.session_state.get(_get_session_key(page_name, "search_results"))
    display_data = st.session_state.get(_get_session_key(page_name, "display_data"))
    metadata = st.session_state.get(_get_session_key(page_name, "metadata"), {})
    search_completed = st.session_state.get(_get_session_key(page_name, "search_completed"), False)
    
    return results, display_data, metadata, search_completed


def clear_search_results(page_name: str) -> None:
    """
    Clear all stored search results for a specific page.
    
    This function resets all session state variables for a page to their initial
    values, typically called when a new search is initiated.
    
    Args:
        page_name: Name of the page to clear results for
    """
    # Clear search results
    results_key = _get_session_key(page_name, "search_results")
    if results_key in st.session_state:
        st.session_state[results_key] = None
    
    # Clear display data
    display_key = _get_session_key(page_name, "display_data")
    if display_key in st.session_state:
        st.session_state[display_key] = None
    
    # Reset search completion flag
    completed_key = _get_session_key(page_name, "search_completed")
    if completed_key in st.session_state:
        st.session_state[completed_key] = False
    
    # Clear metadata
    metadata_key = _get_session_key(page_name, "metadata")
    if metadata_key in st.session_state:
        st.session_state[metadata_key] = {}