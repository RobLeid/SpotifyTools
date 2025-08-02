import streamlit as st
import pandas as pd
import time

from utils.validation import parse_spotify_id_secure
from utils.data_processing import process_artist_album_data, process_artist_search_data
from utils.rate_limiting import RateLimitExceeded
from utils.constants import MARKETS
from utils.ui_components import (
    create_download_button,
    display_album_row,
    display_rate_limit_error,
    normalize_text_for_search,
    display_spotify_attribution,
    create_single_artist_markets_component,
)
from utils.common_operations import get_authenticated_client
from utils.album_sorting import AlbumSorter, SortOrder, safe_sort_albums_with_recovery
from utils.session_state import (
    init_page_session_state,
    store_search_results,
    get_search_results,
    clear_search_results,
)


def perform_catalog_search(artist_id, artist_name, market, market_view="multiple", selected_markets=None):
    """Perform the catalog search for a given artist ID."""
    # Clear previous search results
    clear_search_results("single_artist_catalog")

    if market_view in ["multiple", "all"]:
        # Use Global Artist Catalog logic for multiple/all markets
        if selected_markets is None:
            selected_markets = MARKETS
        perform_global_catalog_search(artist_id, artist_name, selected_markets)
    else:
        # Use Single Artist Catalog logic (original)
        perform_single_catalog_search(artist_id, artist_name, market)


def perform_global_catalog_search(artist_id, artist_name, selected_markets=None):
    """Perform the global catalog search for a given artist ID across selected markets."""
    if selected_markets is None:
        selected_markets = MARKETS
    spotify_client = get_authenticated_client()
    if not spotify_client:
        return

    all_dataframes = []
    album_sections = {}
    unique_album_ids = set()
    albums_by_type = {"album": [], "single": [], "compilation": []}

    with st.status(
        f"⏳ Fetching albums for {artist_name} across {len(selected_markets)} markets...", expanded=False
    ) as status:
        try:
            # STEP 1: Query selected markets to collect unique album IDs
            total_markets = len(selected_markets)
            status.update(
                label=f"Querying {total_markets} markets for albums...", state="running"
            )

            for market_idx, market in enumerate(selected_markets, 1):
                status.update(
                    label=f"Querying market {market_idx}/{total_markets}: {market}...",
                    state="running",
                )
                try:
                    # Fetch albums for this market
                    market_albums = spotify_client.fetch_artist_albums_comprehensive(
                        artist_id, market
                    )

                    if market_albums:
                        # Add unique albums to our collections
                        for album in market_albums:
                            album_id = album.get("id")
                            if album_id and album_id not in unique_album_ids:
                                unique_album_ids.add(album_id)
                                album_type = album.get("album_type", "album")
                                if album_type in albums_by_type:
                                    albums_by_type[album_type].append(album)

                except Exception as e:
                    # Continue with other markets if one fails
                    st.warning(f"Failed to fetch from market {market}: {str(e)}")
                    continue

            # Flatten all albums into a single list
            all_albums = []
            for album_type in ["album", "single", "compilation"]:
                all_albums.extend(albums_by_type[album_type])

            if not all_albums:
                status.update(
                    label="No albums found for this artist in any market.",
                    state="warning",
                    expanded=False,
                )
                return

            status.update(
                label=f"Found {len(all_albums)} unique albums across {total_markets} markets...",
                state="running",
            )

            # STEP 2: Get detailed album info for all albums in batches
            status.update(label="Fetching album details...", state="running")
            album_details = {}
            all_track_ids = []
            track_to_album_map = {}

            # Process albums in batches of 20 (Spotify API limit)
            total_batches = (len(all_albums) + 19) // 20  # Calculate total batches
            for batch_num, i in enumerate(range(0, len(all_albums), 20), 1):
                status.update(
                    label=f"Processing album batch {batch_num}/{total_batches}...",
                    state="running",
                )
                batch_albums = all_albums[i : i + 20]
                batch_ids = [album["id"] for album in batch_albums]

                # Get detailed album information
                album_batch_data = spotify_client._make_request(
                    f"albums?ids={','.join(batch_ids)}"
                )

                if album_batch_data and "albums" in album_batch_data:
                    for album_data in album_batch_data["albums"]:
                        if album_data:
                            album_details[album_data["id"]] = album_data
                            # Extract track IDs for batch fetching
                            track_items = album_data.get("tracks", {}).get("items", [])

                            # Check if album has more tracks than returned (>50)
                            total_tracks = album_data.get("total_tracks", 0)
                            if total_tracks > 50 and len(track_items) == 50:
                                # Fetch all tracks for this album
                                all_album_tracks = (
                                    spotify_client.fetch_album_tracks_all(
                                        album_data["id"]
                                    )
                                )
                                track_items = all_album_tracks
                                # Update the album data with all tracks
                                album_data["tracks"]["items"] = all_album_tracks

                            for track_item in track_items:
                                if track_item.get("id"):
                                    all_track_ids.append(track_item["id"])
                                    track_to_album_map[track_item["id"]] = album_data[
                                        "id"
                                    ]

            # STEP 3: Fetch ALL track details in one batch operation
            track_batches = (
                len(all_track_ids) + 49
            ) // 50  # Calculate total track batches (50 per batch)
            status.update(
                label=f"Fetching {len(all_track_ids)} track details in {track_batches} batches...",
                state="running",
            )

            def update_track_status(message):
                status.update(label=message, state="running")

            all_full_tracks = spotify_client.fetch_tracks_by_ids(
                all_track_ids, status_callback=update_track_status
            )

            # Create a map of track ID to full track data
            track_data_map = {track["id"]: track for track in all_full_tracks if track}

            # STEP 4: Process each album group with pre-fetched data
            groups_to_process = {
                g: [a for a in all_albums if a.get("album_type") == g]
                for g in ["album", "single", "compilation"]
            }
            groups_to_process = {
                k: v for k, v in groups_to_process.items() if v
            }  # Remove empty groups

            # STEP 4.1: Apply sorting to each album type independently with enhanced unknown date handling
            status.update(
                label="Sorting albums by release date within each type...",
                state="running",
            )
            sorting_start_time = time.time()

            try:
                # Prepare album data for sorting - need to map album IDs to detailed data
                sorted_groups = {}

                for group_name, albums_list in groups_to_process.items():
                    # Create list of detailed album data for sorting
                    detailed_albums = []
                    for album in albums_list:
                        album_id = album.get("id")
                        if album_id and album_id in album_details:
                            # Use the detailed album data for sorting
                            detailed_album = album_details[album_id]
                            detailed_albums.append(detailed_album)
                        else:
                            # Fallback to basic album data
                            detailed_albums.append(album)

                    if detailed_albums:
                        # Use safe sorting with comprehensive error recovery and graceful degradation
                        status.update(
                            label=f"Sorting {len(detailed_albums)} {group_name}s with enhanced error handling...",
                            state="running",
                        )

                        sort_result = safe_sort_albums_with_recovery(
                            detailed_albums,
                            primary_strategy="release_date",
                            secondary_strategy="album_id",
                            order=SortOrder.DESCENDING,
                            separate_unknown=True,
                            preserve_unknown_order=True,
                            fallback_to_original=True,
                        )

                        # Store separated known and unknown albums for proper display
                        sorted_groups[group_name] = {
                            "known": sort_result["known"],
                            "unknown": sort_result["unknown"],
                        }

                        status.update(
                            label=f"Sorted {len(detailed_albums)} {group_name}s "
                            f"({len(sort_result['known'])} with dates, {len(sort_result['unknown'])} unknown)",
                            state="running",
                        )
                    else:
                        sorted_groups[group_name] = {"known": [], "unknown": []}

                # Update groups_to_process with sorted data structure
                groups_to_process = sorted_groups

                # Performance monitoring
                sorting_time_ms = (time.time() - sorting_start_time) * 1000
                if sorting_time_ms > 500:
                    st.warning(
                        f"⚠️ Sorting took {sorting_time_ms:.1f}ms (exceeds 500ms target)"
                    )
                else:
                    status.update(
                        label=f"Enhanced sorting completed in {sorting_time_ms:.1f}ms",
                        state="running",
                    )

            except Exception as e:
                # This should rarely happen with safe_sort_albums_with_recovery, but add extra safety
                st.error(f"⚠️ Critical sorting error: {str(e)}. Using original order.")

                # Convert to new structure with all albums marked as "known" to preserve original order
                fallback_groups = {}
                for group_name, albums_list in groups_to_process.items():
                    if isinstance(albums_list, list):
                        fallback_groups[group_name] = {
                            "known": albums_list,
                            "unknown": []
                        }
                    else:
                        fallback_groups[group_name] = albums_list
                groups_to_process = fallback_groups

            # STEP 5: Process each sorted album group
            for group_idx, (group_name, albums_data) in enumerate(
                groups_to_process.items(), 1
            ):
                # Handle both old and new data structures
                if isinstance(albums_data, dict) and "known" in albums_data:
                    known_albums = albums_data["known"]
                    unknown_albums = albums_data["unknown"]
                    total_albums = len(known_albums) + len(unknown_albums)
                else:
                    # Fallback for old structure
                    known_albums = albums_data
                    unknown_albums = []
                    total_albums = len(known_albums)

                status.update(
                    label=f"Processing {group_name}s ({group_idx}/{len(groups_to_process)} groups, {total_albums} items)...",
                    state="running",
                )

                # Process albums with known release dates first
                known_section_dataframes = []
                for i, album in enumerate(known_albums):
                    status.update(
                        label=f"Processing {group_name} {i+1}/{total_albums}: {album['name']}",
                        state="running",
                    )
                    try:
                        album_id = album["id"]
                        if album_id in album_details:
                            album_data = album_details[album_id]
                            track_items = album_data.get("tracks", {}).get("items", [])

                            # Get full track data for this album
                            full_tracks = []
                            for track_item in track_items:
                                if (
                                    track_item.get("id")
                                    and track_item["id"] in track_data_map
                                ):
                                    full_tracks.append(track_data_map[track_item["id"]])

                            if album_data and full_tracks:
                                tracks = process_artist_album_data(
                                    album_data, track_items, full_tracks
                                )
                                df = pd.DataFrame(tracks)
                                known_section_dataframes.append(
                                    (
                                        df,
                                        album_data.get("name"),
                                        (
                                            album_data["images"][0]["url"]
                                            if album_data.get("images")
                                            else None
                                        ),
                                        album_data["id"],
                                        album_data.get("available_markets", []),
                                        "known",
                                    )
                                )
                    except Exception as e:
                        st.warning(f"Failed to process album {album['name']}: {str(e)}")
                        continue

                # Process albums with unknown release dates
                unknown_section_dataframes = []
                for i, album in enumerate(unknown_albums):
                    album_number = len(known_albums) + i + 1
                    status.update(
                        label=f"Processing unknown date {group_name} {album_number}/{total_albums}: {album['name']}",
                        state="running",
                    )
                    try:
                        album_id = album["id"]
                        if album_id in album_details:
                            album_data = album_details[album_id]
                            track_items = album_data.get("tracks", {}).get("items", [])

                            # Get full track data for this album
                            full_tracks = []
                            for track_item in track_items:
                                if (
                                    track_item.get("id")
                                    and track_item["id"] in track_data_map
                                ):
                                    full_tracks.append(track_data_map[track_item["id"]])

                            if album_data and full_tracks:
                                tracks = process_artist_album_data(
                                    album_data, track_items, full_tracks
                                )
                                df = pd.DataFrame(tracks)
                                unknown_section_dataframes.append(
                                    (
                                        df,
                                        album_data.get("name"),
                                        (
                                            album_data["images"][0]["url"]
                                            if album_data.get("images")
                                            else None
                                        ),
                                        album_data["id"],
                                        album_data.get("available_markets", []),
                                        "unknown",
                                    )
                                )
                    except Exception as e:
                        st.warning(f"Failed to process album {album['name']}: {str(e)}")
                        continue

                # Store both known and unknown dataframes
                album_sections[group_name] = {
                    "known": known_section_dataframes,
                    "unknown": unknown_section_dataframes,
                }
                all_dataframes.extend(
                    [df for df, _, _, _, _, _ in known_section_dataframes]
                )
                all_dataframes.extend(
                    [df for df, _, _, _, _, _ in unknown_section_dataframes]
                )

            st.success(
                f"✅ Processed {len(all_albums)} unique albums across {total_markets} markets!"
            )

        except RateLimitExceeded:
            display_rate_limit_error()
        except Exception as e:
            st.error(f"Error fetching albums: {str(e)}")

    if all_dataframes:
        status.update(
            label=f"✅ Done! Found {len(all_albums)} unique albums across {total_markets} markets!",
            state="complete",
            expanded=False,
        )

        combined_df = pd.concat(all_dataframes, ignore_index=True)

        # Store search results in session state
        store_search_results(
            "single_artist_catalog",
            results=combined_df,
            display_data=album_sections,
            metadata={
                "artist_name": artist_name,
                "artist_id": artist_id,
                "search_type": "global",
                "markets": selected_markets,
                "download_filename": f"{artist_name}_Releases.xlsx"
            }
        )

    # Note: Display logic moved to main function to show from session state





def perform_single_catalog_search(artist_id, artist_name, market):
    """Original Single Artist Catalog logic."""
    spotify_client = get_authenticated_client()
    if not spotify_client:
        return

    all_dataframes = []
    album_sections = {}

    with st.status(f"⏳ Fetching albums for {artist_name} in {market} market...", expanded=False) as status:
        try:
            status.update(label="Fetching artist discography...", state="running")
            albums = spotify_client.fetch_artist_albums_comprehensive(artist_id, market)
            if not albums:
                status.update(
                    label="No albums found for this artist.",
                    state="warning",
                    expanded=False,
                )
                return

            status.update(label=f"Found {len(albums)} albums...", state="running")

            # STEP 1: Get detailed album info for all albums in batches
            status.update(label="Fetching album details...", state="running")
            album_details = {}
            all_track_ids = []
            track_to_album_map = {}

            # Process albums in batches of 20 (Spotify API limit)
            total_batches = (len(albums) + 19) // 20  # Calculate total batches
            for batch_num, i in enumerate(range(0, len(albums), 20), 1):
                status.update(
                    label=f"Processing album batch {batch_num}/{total_batches}...",
                    state="running",
                )
                batch_albums = albums[i : i + 20]
                batch_ids = [album["id"] for album in batch_albums]

                # Get detailed album information
                album_batch_data = spotify_client._make_request(
                    f"albums?ids={','.join(batch_ids)}"
                )

                if album_batch_data and "albums" in album_batch_data:
                    for album_data in album_batch_data["albums"]:
                        if album_data:
                            album_details[album_data["id"]] = album_data
                            # Extract track IDs for batch fetching
                            track_items = album_data.get("tracks", {}).get("items", [])

                            # Check if album has more tracks than returned (>50)
                            total_tracks = album_data.get("total_tracks", 0)
                            if total_tracks > 50 and len(track_items) == 50:
                                # Fetch all tracks for this album
                                all_album_tracks = (
                                    spotify_client.fetch_album_tracks_all(
                                        album_data["id"]
                                    )
                                )
                                track_items = all_album_tracks
                                # Update the album data with all tracks
                                album_data["tracks"]["items"] = all_album_tracks

                            for track_item in track_items:
                                if track_item.get("id"):
                                    all_track_ids.append(track_item["id"])
                                    track_to_album_map[track_item["id"]] = album_data[
                                        "id"
                                    ]

            # STEP 2: Fetch ALL track details in one batch operation
            track_batches = (
                len(all_track_ids) + 49
            ) // 50  # Calculate total track batches (50 per batch)
            status.update(
                label=f"Fetching {len(all_track_ids)} track details in {track_batches} batches...",
                state="running",
            )

            def update_track_status(message):
                status.update(label=message, state="running")

            all_full_tracks = spotify_client.fetch_tracks_by_ids(
                all_track_ids, status_callback=update_track_status
            )

            # Create a map of track ID to full track data
            track_data_map = {track["id"]: track for track in all_full_tracks if track}

            # STEP 3: Process each album group with pre-fetched data
            groups_to_process = {
                g: [a for a in albums if a.get("album_type") == g]
                for g in ["album", "single", "compilation"]
            }
            groups_to_process = {
                k: v for k, v in groups_to_process.items() if v
            }  # Remove empty groups

            # STEP 3.1: Apply sorting to each album type independently with enhanced unknown date handling
            status.update(
                label="Sorting albums by release date within each type...",
                state="running",
            )
            sorting_start_time = time.time()

            try:
                # Prepare album data for sorting - need to map album IDs to detailed data
                sorted_groups = {}

                for group_name, albums_list in groups_to_process.items():
                    # Create list of detailed album data for sorting
                    detailed_albums = []
                    for album in albums_list:
                        album_id = album.get("id")
                        if album_id and album_id in album_details:
                            # Use the detailed album data for sorting
                            detailed_album = album_details[album_id]
                            detailed_albums.append(detailed_album)
                        else:
                            # Fallback to basic album data
                            detailed_albums.append(album)

                    if detailed_albums:
                        # Use safe sorting with comprehensive error recovery and graceful degradation
                        status.update(
                            label=f"Sorting {len(detailed_albums)} {group_name}s with enhanced error handling...",
                            state="running",
                        )

                        sort_result = safe_sort_albums_with_recovery(
                            detailed_albums,
                            primary_strategy="release_date",
                            secondary_strategy="album_id",
                            order=SortOrder.DESCENDING,
                            separate_unknown=True,
                            preserve_unknown_order=True,
                            fallback_to_original=True,
                        )

                        # Store separated known and unknown albums for proper display
                        sorted_groups[group_name] = {
                            "known": sort_result["known"],
                            "unknown": sort_result["unknown"],
                        }

                        status.update(
                            label=f"Sorted {len(detailed_albums)} {group_name}s "
                            f"({len(sort_result['known'])} with dates, {len(sort_result['unknown'])} unknown)",
                            state="running",
                        )
                    else:
                        sorted_groups[group_name] = {"known": [], "unknown": []}

                # Update groups_to_process with sorted data structure
                groups_to_process = sorted_groups

                # Performance monitoring
                sorting_time_ms = (time.time() - sorting_start_time) * 1000
                if sorting_time_ms > 500:
                    st.warning(
                        f"⚠️ Sorting took {sorting_time_ms:.1f}ms (exceeds 500ms target)"
                    )
                else:
                    status.update(
                        label=f"Enhanced sorting completed in {sorting_time_ms:.1f}ms",
                        state="running",
                    )

            except Exception as e:
                # This should rarely happen with safe_sort_albums_with_recovery, but add extra safety
                st.error(f"⚠️ Critical sorting error: {str(e)}. Using original order.")

                # Convert to new structure with all albums marked as "known" to preserve original order
                fallback_groups = {}
                for group_name, albums_list in groups_to_process.items():
                    if isinstance(albums_list, list):
                        fallback_groups[group_name] = {
                            "known": albums_list,
                            "unknown": []
                        }
                    else:
                        fallback_groups[group_name] = albums_list
                groups_to_process = fallback_groups

            # STEP 4: Process each sorted album group
            for group_idx, (group_name, albums_data) in enumerate(
                groups_to_process.items(), 1
            ):
                # Handle both old and new data structures
                if isinstance(albums_data, dict) and "known" in albums_data:
                    known_albums = albums_data["known"]
                    unknown_albums = albums_data["unknown"]
                    total_albums = len(known_albums) + len(unknown_albums)
                else:
                    # Fallback for old structure
                    known_albums = albums_data
                    unknown_albums = []
                    total_albums = len(known_albums)

                status.update(
                    label=f"Processing {group_name}s ({group_idx}/{len(groups_to_process)} groups, {total_albums} items)...",
                    state="running",
                )

                # Process albums with known release dates first
                known_section_dataframes = []
                for i, album in enumerate(known_albums):
                    status.update(
                        label=f"Processing {group_name} {i+1}/{total_albums}: {album['name']}",
                        state="running",
                    )
                    try:
                        album_id = album["id"]
                        if album_id in album_details:
                            album_data = album_details[album_id]
                            track_items = album_data.get("tracks", {}).get("items", [])

                            # Get full track data for this album
                            full_tracks = []
                            for track_item in track_items:
                                if (
                                    track_item.get("id")
                                    and track_item["id"] in track_data_map
                                ):
                                    full_tracks.append(track_data_map[track_item["id"]])

                            if album_data and full_tracks:
                                tracks = process_artist_album_data(
                                    album_data, track_items, full_tracks
                                )
                                df = pd.DataFrame(tracks)
                                known_section_dataframes.append(
                                    (
                                        df,
                                        album_data.get("name"),
                                        (
                                            album_data["images"][0]["url"]
                                            if album_data.get("images")
                                            else None
                                        ),
                                        album_data["id"],
                                        album_data.get("available_markets", []),
                                        "known",
                                    )
                                )
                    except Exception as e:
                        st.warning(f"Failed to process album {album['name']}: {str(e)}")
                        continue

                # Process albums with unknown release dates
                unknown_section_dataframes = []
                for i, album in enumerate(unknown_albums):
                    album_number = len(known_albums) + i + 1
                    status.update(
                        label=f"Processing unknown date {group_name} {album_number}/{total_albums}: {album['name']}",
                        state="running",
                    )
                    try:
                        album_id = album["id"]
                        if album_id in album_details:
                            album_data = album_details[album_id]
                            track_items = album_data.get("tracks", {}).get("items", [])

                            # Get full track data for this album
                            full_tracks = []
                            for track_item in track_items:
                                if (
                                    track_item.get("id")
                                    and track_item["id"] in track_data_map
                                ):
                                    full_tracks.append(track_data_map[track_item["id"]])

                            if album_data and full_tracks:
                                tracks = process_artist_album_data(
                                    album_data, track_items, full_tracks
                                )
                                df = pd.DataFrame(tracks)
                                unknown_section_dataframes.append(
                                    (
                                        df,
                                        album_data.get("name"),
                                        (
                                            album_data["images"][0]["url"]
                                            if album_data.get("images")
                                            else None
                                        ),
                                        album_data["id"],
                                        album_data.get("available_markets", []),
                                        "unknown",
                                    )
                                )
                    except Exception as e:
                        st.warning(f"Failed to process album {album['name']}: {str(e)}")
                        continue

                # Store both known and unknown dataframes
                album_sections[group_name] = {
                    "known": known_section_dataframes,
                    "unknown": unknown_section_dataframes,
                }
                all_dataframes.extend(
                    [df for df, _, _, _, _, _ in known_section_dataframes]
                )
                all_dataframes.extend(
                    [df for df, _, _, _, _, _ in unknown_section_dataframes]
                )

            st.success(f"✅ Processed {len(albums)} albums!")

        except RateLimitExceeded:
            display_rate_limit_error()
        except Exception as e:
            st.error(f"Error fetching albums: {str(e)}")

    # Display results
    if all_dataframes:
        status.update(
            label=f"✅ Done! Found {len(albums)} albums!",
            state="complete",
            expanded=False,
        )

        combined_df = pd.concat(all_dataframes, ignore_index=True)

        # Store search results in session state
        store_search_results(
            "single_artist_catalog",
            results=combined_df,
            display_data=album_sections,
            metadata={
                "artist_name": artist_name,
                "artist_id": artist_id,
                "search_type": "single",
                "market": market,
                "download_filename": f"{artist_name}_Releases.xlsx"
            }
        )

    # Note: Display logic moved to main function to show from session state



def main():
    st.title("🎤 Single Artist Catalog")
    st.caption("Get all releases by a single artist.")

    # Initialize session state for this page
    init_page_session_state("single_artist_catalog")

    # Display Spotify attribution (required by Spotify Developer Terms)
    display_spotify_attribution()

    # Add market selection (Multiple or All Markets only)
    market_view, selected_markets = create_single_artist_markets_component(key="single_artist_catalog")

    # Search method selection
    search_method = st.radio(
        "How would you like to find the artist?",
        ["Search by Artist Name", "Enter Artist ID/URL directly"],
        key="search_method",
    )

    if search_method == "Search by Artist Name":
        # Artist name search interface - no additional market selector needed
        search_query = st.text_input("Enter artist name or partial name to search")
        market = "US"  # Default market for artist search only (album data uses selected_markets)

        # Fixed search limit of 20
        search_limit = 20

        # Initialize session state for search results
        if "search_results" not in st.session_state:
            st.session_state.search_results = None
        if "search_query_done" not in st.session_state:
            st.session_state.search_query_done = ""

        if st.button("🔍 Search for Artists") and search_query:

            # Perform artist search
            spotify_client = get_authenticated_client()
            if not spotify_client:
                return

            with st.status("⏳ Searching for artists...", expanded=False) as status:
                try:
                    status.update(
                        label=f"Searching for '{search_query}'...", state="running"
                    )

                    artists = spotify_client.search_artists(
                        search_query, limit=search_limit, market=market
                    )

                    if artists:
                        # Process and filter the artist data
                        processed_artists = process_artist_search_data(artists)

                        if processed_artists:
                            # Convert to DataFrame for consistency with display logic
                            df = pd.DataFrame(processed_artists)

                            # Store DataFrame in session state
                            st.session_state.search_results = df
                            st.session_state.search_query_done = search_query

                            status.update(
                                label=f"Found {len(processed_artists)} artists!",
                                state="complete",
                                expanded=False,
                            )
                        else:
                            st.session_state.search_results = None
                            status.update(
                                label="No valid artist data found.",
                                state="warning",
                                expanded=False,
                            )

                    else:
                        st.session_state.search_results = None
                        status.update(
                            label="No artists found matching your search",
                            state="warning",
                            expanded=False,
                        )
                        st.warning("No artists found. Try a different search term.")

                except RateLimitExceeded:
                    st.session_state.search_results = None
                    display_rate_limit_error()
                except Exception as e:
                    st.session_state.search_results = None
                    status.update(
                        label=f"Search failed: {str(e)}", state="error", expanded=False
                    )
                    st.error(f"Search failed: {str(e)}")

        # Check if there's a pending catalog search to execute
        if "pending_catalog_search" in st.session_state:
            pending = st.session_state.pending_catalog_search
            del st.session_state.pending_catalog_search  # Clear it immediately
            perform_catalog_search(
                pending["artist_id"], pending["artist_name"], pending["market"],
                pending.get("market_view", "multiple"),
                pending.get("selected_markets", None)
            )

        # Display search results outside the status block
        elif st.session_state.search_results is not None:
            filtered_df = st.session_state.search_results

            if not filtered_df.empty:
                st.subheader(f"🎤 Search Results for '{st.session_state.search_query_done}'")

                if len(filtered_df) == 1:
                    # Auto-select if only one result
                    selected_artist = filtered_df.iloc[0]
                    st.info(
                        f"Found exactly one match: **{selected_artist['Artist Name']}** (ID: `{selected_artist['Artist ID']}`)"
                    )

                    if st.button(
                        f"📦 Get Artist Catalog for {selected_artist['Artist Name']}",
                        key="single_result",
                    ):
                        # Store the artist info for catalog search and clear search results
                        st.session_state.pending_catalog_search = {
                            "artist_id": selected_artist["Artist ID"],
                            "artist_name": selected_artist["Artist Name"],
                            "market": market,
                            "market_view": market_view,
                            "selected_markets": selected_markets,
                        }
                        st.session_state.search_results = None
                        st.rerun()

                else:
                    # Multiple results - let user choose
                    st.write(f"Found {len(filtered_df)} artists matching your search:")

                    # Create selectbox for artist selection with Artist ID included
                    artist_options = [
                        f"{row['Artist Name']} ({row['Followers']} followers) - ID: {row['Artist ID']}"
                        for _, row in filtered_df.iterrows()
                    ]
                    selected_index = st.selectbox(
                        "Select an artist to get their catalog:",
                        range(len(artist_options)),
                        format_func=lambda x: artist_options[x],
                        key="artist_selection"
                    )

                    if st.button("📦 Get Artist Catalog", key="multi_result"):
                        selected_artist = filtered_df.iloc[selected_index]
                        # Store the artist info for catalog search and clear search results
                        st.session_state.pending_catalog_search = {
                            "artist_id": selected_artist["Artist ID"],
                            "artist_name": selected_artist["Artist Name"],
                            "market": market,
                            "market_view": market_view,
                            "selected_markets": selected_markets,
                        }
                        st.session_state.search_results = None
                        st.rerun()

    else:
        # Direct ID/URL input - no additional market selector needed
        artist_input = st.text_input("Enter a Spotify artist URI, URL, or ID")
        market = "US"  # Default market for artist info only (album data uses selected_markets)

        if st.button("🔍 Get Artist Catalog") and artist_input:
            artist_id = parse_spotify_id_secure(artist_input, "artist")
            if not artist_id:
                return

            # Get artist name for display
            spotify_client = get_authenticated_client()
            if spotify_client:
                try:
                    artist_data = spotify_client._make_request(f"artists/{artist_id}")
                    artist_name = (
                        artist_data.get("name", "Unknown Artist")
                        if artist_data
                        else "Unknown Artist"
                    )
                except:
                    artist_name = "Unknown Artist"
            else:
                return

            perform_catalog_search(artist_id, artist_name, market, market_view, selected_markets)

    # Display results from session state (if any)
    results, display_data, metadata, search_completed = get_search_results("single_artist_catalog")

    if search_completed and results is not None:
        # Display download button with dynamic key to prevent re-rendering issues
        if not results.empty:
            download_filename = metadata.get("download_filename", "Artist_Releases.xlsx")
            
            # Generate a stable, unique key based on search context
            artist_id = metadata.get("artist_id", "unknown")
            search_type = metadata.get("search_type", "single")
            markets = metadata.get("markets", metadata.get("market", []))
            
            # Create hash for complex market data
            if isinstance(markets, list):
                markets_str = "_".join(sorted(markets))
            else:
                markets_str = str(markets)
            
            download_key = f"download_all_{artist_id}_{search_type}_{hash(markets_str)}"
            
            create_download_button(
                df=results,
                label="📦 Download All Albums to Excel",
                file_name=download_filename,
                key=download_key,
            )

        # Display album sections
        if display_data:
            for group_name, section_data in display_data.items():
                if section_data and (section_data.get("known") or section_data.get("unknown")):
                    st.header(group_name.capitalize() + "s")
                    st.divider()

                    # Display albums with known release dates first
                    known_dataframes = section_data.get("known", [])
                    unknown_dataframes = section_data.get("unknown", [])

                    # Display albums with known release dates first (sorted by date, newest first)
                    if known_dataframes:
                        for (
                            df,
                            album_name,
                            album_image_url,
                            album_id,
                            available_markets,
                            section_type,
                        ) in known_dataframes:
                            album_data = {
                                "name": album_name,
                                "images": [{"url": album_image_url}] if album_image_url else [],
                            }
                            display_album_row(
                                album_data,
                                df,
                                album_id,
                                available_markets=available_markets,
                                show_availability_tooltip=True,
                            )

                    # Display albums with unknown release dates in separate section with clear visual separation
                    if unknown_dataframes:
                        st.divider()  # Visual separator between known and unknown sections
                        st.subheader(f"📅 Unknown Release Date {group_name.capitalize()}s")
                        st.caption(
                            f"These {group_name}s do not have release date information available. They are displayed in their original discovery order."
                        )

                        # Add a subtle visual indicator for the unknown section
                        with st.container():
                            st.markdown("---")
                            for (
                                df,
                                album_name,
                                album_image_url,
                                album_id,
                                available_markets,
                                section_type,
                            ) in unknown_dataframes:
                                album_data = {
                                    "name": album_name,
                                    "images": (
                                        [{"url": album_image_url}] if album_image_url else []
                                    ),
                                }
                                display_album_row(
                                    album_data,
                                    df,
                                    album_id,
                                    available_markets=available_markets,
                                    show_availability_tooltip=True,
                                )


if __name__ == "__main__":
    main()
