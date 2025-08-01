from typing import Set, Dict, List
from .constants import COUNTRY_TO_REGION, REGIONS, MARKETS


def ms_to_min_sec(ms):
    """Converts milliseconds to a mm:ss format string."""
    if ms is None or ms < 0:
        return "0:00"

    try:
        minutes = ms // 60000
        seconds = (ms % 60000) // 1000
        return f"{minutes}:{seconds:02d}"
    except (TypeError, ValueError):
        return "0:00"


def safe_get(data, key, default="N/A"):
    """Safely get a value from a dictionary"""
    if not isinstance(data, dict):
        return default
    return data.get(key, default)


def get_artist_names(artists):
    """Extract artist names from artists array"""
    if not artists or not isinstance(artists, list):
        return "Unknown Artist"

    try:
        names = []
        for artist in artists:
            if isinstance(artist, dict) and "name" in artist:
                names.append(artist["name"])
        return ", ".join(names) if names else "Unknown Artist"
    except:
        return "Unknown Artist"


def process_track_data(tracks):
    """Processes raw track JSON data into a simplified format for a DataFrame."""
    if not tracks:
        return []

    simplified_data = []

    for track in tracks:
        if not track or not isinstance(track, dict):
            continue

        try:
            simplified_data.append(
                {
                    "Track Artist(s)": get_artist_names(track.get("artists", [])),
                    "Track Name": safe_get(track, "name", "Unknown Track"),
                    "ISRC": safe_get(track.get("external_ids", {}), "isrc", "N/A"),
                    "Duration": ms_to_min_sec(track.get("duration_ms", 0)),
                    "Explicit": "Yes" if track.get("explicit", False) else "No",
                    "Spotify URL": safe_get(
                        track.get("external_urls", {}), "spotify", "N/A"
                    ),
                }
            )
        except Exception as e:
            # Skip tracks that can't be processed but log the error
            import streamlit as st

            st.warning(f"Skipped track due to processing error: {str(e)}")
            continue

    return simplified_data


def process_album_track_data(album_data, track_items, full_tracks):
    """Processes album and track JSON data for a DataFrame."""
    if not album_data or not isinstance(album_data, dict):
        return []

    # Extract album metadata
    album_artists = get_artist_names(album_data.get("artists", []))
    album_name = safe_get(album_data, "name", "Unknown Album")
    upc = safe_get(album_data.get("external_ids", {}), "upc", "N/A")
    release_date = safe_get(album_data, "release_date", "N/A")
    release_type = safe_get(album_data, "album_type", "N/A").capitalize()
    label = safe_get(album_data, "label", "N/A")
    album_url = safe_get(album_data.get("external_urls", {}), "spotify", "N/A")

    # Extract and format available markets (countries)
    available_markets = album_data.get("available_markets", [])
    if isinstance(available_markets, list) and available_markets:
        countries = ", ".join(sorted(available_markets))
    else:
        countries = "N/A"

    # Calculate unavailable markets
    if isinstance(available_markets, list):
        available_set = set(available_markets)
        all_markets_set = set(MARKETS)
        unavailable_markets = sorted(list(all_markets_set - available_set))
        unavailable_countries = (
            ", ".join(unavailable_markets) if unavailable_markets else ""
        )
    else:
        unavailable_countries = ", ".join(
            sorted(MARKETS)
        )  # All markets are unavailable if no data

    # Extract P-line (phonogram copyright)
    p_line = "N/A"
    copyrights = album_data.get("copyrights", [])
    if isinstance(copyrights, list):
        for copyright_info in copyrights:
            if isinstance(copyright_info, dict) and copyright_info.get("type") == "P":
                p_line = copyright_info.get("text", "N/A")
                break

    simplified_data = []

    # Process tracks (ensure we have matching track_items and full_tracks)
    for i, (track_item, full_track) in enumerate(zip(track_items, full_tracks)):
        if not full_track or not isinstance(full_track, dict):
            continue

        try:
            simplified_data.append(
                {
                    "Album Artist(s)": album_artists,
                    "Album Name": album_name,
                    "UPC": upc,
                    "Release Date": release_date,
                    "Release Type": release_type,
                    "Label": label,
                    "℗ Line": p_line,
                    "Album Spotify URL": album_url,
                    "Available Markets": countries,
                    "Unavailable Markets": unavailable_countries,
                    "Disc Number": (
                        track_item.get("disc_number", 1) if track_item else 1
                    ),
                    "Track Number": (
                        track_item.get("track_number", i + 1) if track_item else i + 1
                    ),
                    "Track Artist(s)": get_artist_names(full_track.get("artists", [])),
                    "Track Name": safe_get(full_track, "name", "Unknown Track"),
                    "ISRC": safe_get(full_track.get("external_ids", {}), "isrc", "N/A"),
                    "Explicit": "Yes" if full_track.get("explicit", False) else "No",
                    "Duration": ms_to_min_sec(full_track.get("duration_ms", 0)),
                    "Track Spotify URL": safe_get(
                        full_track.get("external_urls", {}), "spotify", "N/A"
                    ),
                }
            )
        except Exception:
            # Skip tracks that can't be processed
            continue

    return simplified_data


def process_artist_album_data(album_data, track_items, full_tracks):
    """Processes album and track data specifically for artist catalogs."""
    # Use the same processing as regular albums
    return process_album_track_data(album_data, track_items, full_tracks)


def process_artist_search_data(artists):
    """Processes raw artist search JSON data into a simplified format for a DataFrame."""
    if not artists:
        return []

    simplified_data = []

    for artist in artists:
        if not artist or not isinstance(artist, dict):
            continue

        try:
            # Process follower count
            followers = artist.get("followers", {})
            follower_count = (
                followers.get("total", 0) if isinstance(followers, dict) else 0
            )

            # Format follower count with commas
            follower_count_formatted = f"{follower_count:,}" if follower_count else "0"

            simplified_data.append(
                {
                    "Artist Name": safe_get(artist, "name", "Unknown Artist"),
                    "Artist ID": safe_get(artist, "id", "N/A"),
                    "Followers": follower_count_formatted,
                    "Spotify URL": safe_get(
                        artist.get("external_urls", {}), "spotify", "N/A"
                    ),
                }
            )
        except Exception as e:
            # Skip artists that can't be processed but log the error
            import streamlit as st

            st.warning(f"Skipped artist due to processing error: {str(e)}")
            continue

    return simplified_data


def group_markets_by_region(
    markets: Set[str], include_unknown: bool = False
) -> Dict[str, List[str]]:
    """
    Groups a set of market codes by their geographic regions.

    Args:
        markets: A set of ISO country codes (e.g., {'US', 'GB', 'JP'})
        include_unknown: If True, unknown markets are included under 'UNKNOWN' key

    Returns:
        A dictionary where keys are region names and values are lists of country codes
        sorted alphabetically within each region. If include_unknown is True and there
        are unknown markets, they will be included under the 'UNKNOWN' key.

    Example:
        {'NORTH_AMERICA': ['CA', 'MX', 'US'], 'EUROPE': ['DE', 'FR', 'GB']}
    """
    grouped = {}
    unknown_markets = []

    for market in markets:
        # Get the region for this market
        region = COUNTRY_TO_REGION.get(market)

        if region is None:
            # Handle unknown markets
            if include_unknown:
                unknown_markets.append(market)
            continue

        # Initialize the region list if needed
        if region not in grouped:
            grouped[region] = []

        # Add the market to its region
        grouped[region].append(market)

    # Sort markets alphabetically within each region
    for region in grouped:
        grouped[region].sort()

    # Add unknown markets if requested and present
    if include_unknown and unknown_markets:
        grouped["UNKNOWN"] = sorted(unknown_markets)

    return grouped


def calculate_region_percentage(
    region_markets: List[str], total_region_markets: List[str]
) -> float:
    """
    Calculates the percentage of markets from a region that are present.

    Args:
        region_markets: List of market codes present in the current section (e.g., available or missing)
        total_region_markets: List of all market codes that belong to this region

    Returns:
        The percentage as a float (0-100), rounded to 1 decimal place.
        Returns 0.0 if total_region_markets is empty.

    Example:
        calculate_region_percentage(['US', 'CA'], ['US', 'CA', 'MX']) -> 66.7
    """
    if not total_region_markets:
        return 0.0

    # Calculate the percentage
    percentage = (len(region_markets) / len(total_region_markets)) * 100

    # Round to 1 decimal place
    return round(percentage, 1)


def get_all_markets_for_region(region: str) -> List[str]:
    """
    Gets all market codes that belong to a specific region.

    Args:
        region: The region key (e.g., 'NORTH_AMERICA', 'EUROPE')

    Returns:
        A sorted list of all market codes in that region.
        Returns an empty list if the region is not found.

    Example:
        get_all_markets_for_region('NORTH_AMERICA') -> ['BZ', 'CA', 'CR', 'GT', 'HN', 'MX', 'NI', 'PA', 'PR', 'SV', 'US']
    """
    markets = []

    # Find all markets that belong to this region
    for market, market_region in COUNTRY_TO_REGION.items():
        if market_region == region:
            markets.append(market)

    # Sort the markets alphabetically
    markets.sort()

    return markets


def is_valid_market(market: str) -> bool:
    """
    Checks if a market code is valid (exists in COUNTRY_TO_REGION mapping).

    Args:
        market: The market code to validate (e.g., 'US', 'GB')

    Returns:
        True if the market code is valid, False otherwise.

    Example:
        is_valid_market('US') -> True
        is_valid_market('XX') -> False
    """
    return market in COUNTRY_TO_REGION
