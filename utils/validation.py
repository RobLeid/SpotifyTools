"""
Input validation and sanitization utilities for the Spotify ISRC Finder application.
"""

import re
import html
from typing import List, Optional, Tuple
import streamlit as st
from .constants import (
    MAX_INPUT_LENGTH,
    MAX_ITEMS_PER_REQUEST,
    SPOTIFY_ID_LENGTH,
    ERROR_MESSAGES,
)


class ValidationError(Exception):
    """Custom exception for validation errors."""

    pass


def sanitize_input(user_input: str) -> str:
    """
    Sanitize user input to prevent XSS and other attacks.

    Args:
        user_input: Raw user input string

    Returns:
        Sanitized input string

    Raises:
        ValidationError: If input is invalid
    """
    if not isinstance(user_input, str):
        raise ValidationError("Input must be a string")

    # Remove null bytes and control characters
    sanitized = "".join(
        char for char in user_input if ord(char) >= 32 or char in "\t\n\r"
    )

    # HTML escape to prevent XSS
    sanitized = html.escape(sanitized.strip())

    # Check length
    if len(sanitized) > MAX_INPUT_LENGTH:
        raise ValidationError(ERROR_MESSAGES["input_too_long"])

    return sanitized


def validate_spotify_id(spotify_id: str, item_type: str) -> bool:
    """
    Validate a Spotify ID format.

    Args:
        spotify_id: The Spotify ID to validate
        item_type: Type of item (track, album, artist, playlist)

    Returns:
        True if valid, False otherwise
    """
    if not spotify_id or not isinstance(spotify_id, str):
        return False

    # Check length
    if len(spotify_id) != SPOTIFY_ID_LENGTH:
        return False

    # Check characters (base62: a-z, A-Z, 0-9)
    if not re.match(r"^[a-zA-Z0-9]{22}$", spotify_id):
        return False

    return True


def parse_spotify_id_secure(user_input: str, item_type: str) -> Optional[str]:
    """
    Securely parse a Spotify ID from various input formats.

    Args:
        user_input: Raw user input
        item_type: Type of item (track, album, artist, playlist)

    Returns:
        Parsed Spotify ID or None if invalid
    """
    if not user_input:
        return None

    try:
        # Sanitize input first
        sanitized_input = sanitize_input(user_input)

        # Handle URI format: spotify:track:4iV5W9uYEdYUVa79Axb7Rh
        if sanitized_input.startswith(f"spotify:{item_type}:"):
            parts = sanitized_input.split(":")
            if len(parts) >= 3:
                spotify_id = parts[2]
                if validate_spotify_id(spotify_id, item_type):
                    return spotify_id

        # Handle URL formats with more robust parsing
        url_patterns = [
            rf"spotify\.com/{re.escape(item_type)}/([a-zA-Z0-9]{{22}})",
            rf"open\.spotify\.com/{re.escape(item_type)}/([a-zA-Z0-9]{{22}})",
        ]

        for pattern in url_patterns:
            match = re.search(pattern, sanitized_input)
            if match:
                spotify_id = match.group(1)
                if validate_spotify_id(spotify_id, item_type):
                    return spotify_id

        # Handle plain ID
        if validate_spotify_id(sanitized_input, item_type):
            return sanitized_input

        # Log invalid input for debugging
        st.warning(f"Invalid {item_type} input: '{user_input[:50]}...' (truncated)")

    except ValidationError as e:
        st.error(f"Validation error: {e}")
    except Exception as e:
        st.error(f"Unexpected error parsing {item_type} ID: {e}")

    return None


def validate_batch_size(items: List[str], max_items: int) -> Tuple[bool, str]:
    """
    Validate batch size doesn't exceed limits.

    Args:
        items: List of items to validate
        max_items: Maximum allowed items

    Returns:
        Tuple of (is_valid, error_message)
    """
    if len(items) > max_items:
        return False, ERROR_MESSAGES["too_many_items"]

    return True, ""


def parse_multi_spotify_ids_secure(user_input: str, item_type: str) -> List[str]:
    """
    Securely parse multiple Spotify IDs from text input.

    Args:
        user_input: Raw user input with multiple IDs
        item_type: Type of item (track, album, artist, playlist)

    Returns:
        List of valid Spotify IDs
    """
    if not user_input:
        return []

    try:
        # Don't sanitize here - parse_spotify_id_secure will do it
        # This prevents double HTML escaping
        # Split by lines and clean
        raw_items = [item.strip() for item in user_input.splitlines() if item.strip()]

        # Validate batch size
        is_valid, error_msg = validate_batch_size(raw_items, MAX_ITEMS_PER_REQUEST)
        if not is_valid:
            st.error(error_msg)
            return []

        valid_ids = []
        invalid_count = 0

        for item in raw_items:
            parsed_id = parse_spotify_id_secure(item, item_type)
            if parsed_id:
                valid_ids.append(parsed_id)
            else:
                invalid_count += 1

        # Provide feedback
        if valid_ids:
            st.success(f"✅ Found {len(valid_ids)} valid {item_type} IDs")

        if invalid_count > 0:
            st.warning(f"⚠️ Skipped {invalid_count} invalid entries")

        return valid_ids

    except ValidationError as e:
        st.error(f"Validation error: {e}")
        return []
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return []


def validate_market_code(market: str) -> bool:
    """
    Validate market code format (ISO 3166-1 alpha-2).

    Args:
        market: Market code to validate

    Returns:
        True if valid, False otherwise
    """
    if not market or not isinstance(market, str):
        return False

    # Market codes are 2 uppercase letters
    return bool(re.match(r"^[A-Z]{2}$", market.upper()))


def sanitize_market_code(market: str) -> str:
    """
    Sanitize and validate market code.

    Args:
        market: Raw market code

    Returns:
        Sanitized market code

    Raises:
        ValidationError: If market code is invalid
    """
    if not market:
        raise ValidationError("Market code cannot be empty")

    sanitized = sanitize_input(market).upper()

    if not validate_market_code(sanitized):
        raise ValidationError(f"Invalid market code: {market}")

    return sanitized
