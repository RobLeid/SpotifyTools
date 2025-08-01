import streamlit as st
import pandas as pd
import time
from utils.rate_limiting import RateLimitExceeded
from utils.validation import parse_multi_spotify_ids_secure
from utils.data_processing import process_artist_album_data, process_artist_search_data
from utils.constants import MARKETS
from utils.ui_components import (
    create_download_button,
    display_processing_info,
    display_rate_limit_error,
    normalize_text_for_search,
    display_spotify_attribution,
    create_multiple_markets_only_component,
)
from utils.common_operations import get_authenticated_client
from utils.album_sorting import AlbumSorter, SortOrder, safe_sort_albums_with_recovery
from utils.session_state import (
    init_page_session_state,
    store_search_results,
    get_search_results,
    clear_search_results,
)


def perform_multiple_catalog_search(artist_worklist, market, market_view="multiple", selected_markets=None):
    """Perform catalog search for multiple artists across selected markets."""
    # Clear previous search results
    clear_search_results("multiple_artist_catalog")
    
    # Always use multiple markets mode for Multiple Artist Catalog
    if selected_markets is None:
        selected_markets = MARKETS  # Fallback to all markets if none provided
    perform_multiple_global_catalog_search(artist_worklist, market, market_view, selected_markets)


def perform_multiple_global_catalog_search(artist_worklist, market, market_view="multiple", selected_markets=None):
    """Perform catalog search for multiple artists across selected markets."""
    if not artist_worklist:
        st.error("No artists in worklist.")
        return

    # Handle selected markets based on mode
    if selected_markets is None or len(selected_markets) == 0:
        # Fallback to US market if no markets provided
        selected_markets = ["US"]

    # Extract just the artist IDs for processing
    artist_ids = [artist["artist_id"] for artist in artist_worklist]

    spotify_client = get_authenticated_client()
    if not spotify_client:
        return

    # Full-width processing info
    processing_info = st.empty()
    if market_view == "single":
        processing_info.info(f"🎯 Processing {len(artist_ids)} artists in {market} market...")
    elif market_view == "multiple":
        processing_info.info(f"🎯 Processing {len(artist_ids)} artists across {len(selected_markets)} selected markets...")
    else:  # "all"
        processing_info.info(f"🎯 Processing {len(artist_ids)} artists across all {len(selected_markets)} markets...")

    all_data = []
    start_time = time.time()

    # Full-width status bar
    status_message = "⏳ Processing multiple artists"
    if market_view == "single":
        status_message += f" in {market} market..."
    elif market_view == "multiple":
        status_message += f" across {len(selected_markets)} markets..."
    else:  # "all"
        status_message += " across all markets..."
    
    with st.status(status_message, expanded=False) as status:
        try:
            # Process each artist sequentially (like Single Artist Catalog does)
            for artist_idx, artist_id in enumerate(artist_ids, 1):
                artist_name = next(
                    (
                        a["artist_name"]
                        for a in artist_worklist
                        if a["artist_id"] == artist_id
                    ),
                    artist_id,
                )
                status.update(
                    label=f"Processing artist {artist_idx}/{len(artist_ids)}: {artist_name}",
                    state="running",
                )

                if market_view in ["multiple", "all"]:
                    # Multiple/All markets logic - collect unique albums across selected markets
                    unique_album_ids = set()
                    albums_by_type = {"album": [], "single": [], "compilation": []}
                    
                    for mkt_idx, mkt in enumerate(selected_markets, 1):
                        status.update(
                            label=f"Artist {artist_idx}: Querying market {mkt_idx}/{len(selected_markets)}: {mkt}...",
                            state="running",
                        )
                        try:
                            # Fetch albums for this market
                            market_albums = spotify_client.fetch_artist_albums_comprehensive(
                                artist_id, mkt
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
                            continue
                    
                    # Flatten all albums into a single list
                    albums = []
                    for album_type in ["album", "single", "compilation"]:
                        albums.extend(albums_by_type[album_type])
                else:
                    # Single market logic
                    albums = spotify_client.fetch_artist_albums_comprehensive(
                        artist_id, market
                    )
                    
                if not albums:
                    st.warning(f"No albums found for artist {artist_name}")
                    continue

                status.update(
                    label=f"Artist {artist_idx}: Found {len(albums)} albums...",
                    state="running",
                )

                # Get detailed album info for all albums in batches (same as Single Artist)
                album_details = {}
                all_track_ids = []
                track_to_album_map = {}

                # Process albums in batches of 20
                for i in range(0, len(albums), 20):
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
                                track_items = album_data.get("tracks", {}).get(
                                    "items", []
                                )

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
                                        track_to_album_map[track_item["id"]] = (
                                            album_data["id"]
                                        )

                if not all_track_ids:
                    st.warning(f"No tracks found for artist {artist_name}")
                    continue

                # Fetch ALL track details in batch operation
                status.update(
                    label=f"Artist {artist_idx}: Fetching {len(all_track_ids)} track details...",
                    state="running",
                )

                def update_track_status(message):
                    status.update(
                        label=f"Artist {artist_idx}: {message}", state="running"
                    )

                all_full_tracks = spotify_client.fetch_tracks_by_ids(
                    all_track_ids, status_callback=update_track_status
                )

                # Create a map of track ID to full track data
                track_data_map = {
                    track["id"]: track for track in all_full_tracks if track
                }

                # Sort albums by release date (same as Global Artist Catalog)
                status.update(
                    label=f"Artist {artist_idx}: Sorting {len(albums)} albums by release date...",
                    state="running",
                )
                
                try:
                    # Create detailed album data for sorting
                    detailed_albums = []
                    for album in albums:
                        album_id = album.get("id")
                        if album_id and album_id in album_details:
                            # Use the detailed album data for sorting
                            detailed_album = album_details[album_id]
                            detailed_albums.append(detailed_album)
                        else:
                            # Fallback to basic album data
                            detailed_albums.append(album)
                    
                    # Use safe sorting with error recovery
                    sort_result = safe_sort_albums_with_recovery(
                        detailed_albums,
                        primary_strategy="release_date",
                        secondary_strategy="album_id",
                        order=SortOrder.DESCENDING,
                        separate_unknown=True,
                        preserve_unknown_order=True,
                        fallback_to_original=True,
                    )
                    
                    # Combine sorted albums (known dates first, then unknown)
                    sorted_albums = sort_result["known"] + sort_result["unknown"]
                    albums = sorted_albums  # Update albums list with sorted order
                    
                    status.update(
                        label=f"Artist {artist_idx}: Sorted {len(albums)} albums "
                        f"({len(sort_result['known'])} with dates, {len(sort_result['unknown'])} unknown)",
                        state="running",
                    )
                    
                except Exception as e:
                    # If sorting fails, continue with original order
                    status.update(
                        label=f"Artist {artist_idx}: Sorting failed, using original order",
                        state="running",
                    )

                # Process each album with pre-fetched data (now in sorted order)
                for album in albums:
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
                            all_data.extend(tracks)

            elapsed = time.time() - start_time
            status_message = f"✅ Completed! Processed {len(artist_ids)} artist(s), {len(all_data)} tracks in {elapsed:.2f}s"
            status.update(label=status_message, state="complete", expanded=False)

        except RateLimitExceeded:
            elapsed = time.time() - start_time
            display_rate_limit_error()
            status.update(
                label=f"❌ Rate limit exceeded - returning partial data ({elapsed:.2f}s)",
                state="error",
                expanded=False,
            )
        except Exception as e:
            elapsed = time.time() - start_time
            st.error(f"❌ Unexpected error: {e}")
            status.update(
                label=f"❌ Error occurred - returning partial data ({elapsed:.2f}s)",
                state="error",
                expanded=False,
            )

    if all_data:
        processing_info.empty()  # Remove the processing message
        df = pd.DataFrame(all_data)

        if not df.empty:
            # Store results in session state
            store_search_results(
                "multiple_artist_catalog",
                results=df,
                metadata={
                    "artist_count": len(artist_ids),
                    "track_count": len(df),
                    "download_filename": f"Multiple_Artists_Releases_{len(artist_ids)}_artists.xlsx"
                }
            )
    else:
        st.error("❌ No data was successfully retrieved. This could be due to:")
        st.markdown(
            """
        - Invalid artist IDs
        - API rate limiting
        - Network connectivity issues
        - Artists with no available releases in the selected market
        - Check the warnings above for specific issues with each artist
        """
        )


def main():
    st.title("🎤 Multiple Artist Catalog")
    st.caption("Build a worklist of artists and get all their releases in one report.")

    # Initialize session state for this page
    init_page_session_state("multiple_artist_catalog")

    # Display Spotify attribution (required by Spotify Developer Terms)
    display_spotify_attribution()
    
    # Add multiple markets selector (only option for Multiple Artist Catalog)
    selected_markets = create_multiple_markets_only_component(key="multiple_artist_catalog")
    market_view = "multiple"  # Always multiple for this page

    # Initialize session state for artist worklist
    if "artist_worklist" not in st.session_state:
        st.session_state.artist_worklist = []
    if "search_results" not in st.session_state:
        st.session_state.search_results = None
    if "search_query_done" not in st.session_state:
        st.session_state.search_query_done = ""
    if "pending_multiple_catalog_search" not in st.session_state:
        st.session_state.pending_multiple_catalog_search = None

    # Search method selection
    search_method = st.radio(
        "How would you like to add artists?",
        ["Search by Artist Name", "Enter Artist IDs/URLs directly"],
        key="search_method",
    )

    # Initialize market variable to ensure it's always available
    market = "US"  # Default value

    if search_method == "Search by Artist Name":
        # Artist name search interface - no additional market selector needed
        search_query = st.text_input("Enter artist name or partial name to search")
        market = "US"  # Default market for artist search only (album data uses selected_markets)

        # Fixed search limit of 20
        search_limit = 20

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
                        artist_data = process_artist_search_data(artists)

                        if artist_data:
                            # Create DataFrame
                            df = pd.DataFrame(artist_data)

                            # Filter results to only include artists whose names contain the search words
                            normalized_search_query = normalize_text_for_search(
                                search_query
                            )
                            search_words = normalized_search_query.split()

                            def contains_search_words(artist_name):
                                normalized_artist_name = normalize_text_for_search(
                                    artist_name
                                )
                                return all(
                                    word in normalized_artist_name
                                    for word in search_words
                                )

                            filtered_df = df[
                                df["Artist Name"].apply(contains_search_words)
                            ]

                            if not filtered_df.empty:
                                status.update(
                                    label=f"✅ Found {len(filtered_df)} relevant artists!",
                                    state="complete",
                                )
                                # Store results in session state
                                st.session_state.search_results = filtered_df
                                st.session_state.search_query_done = search_query
                            else:
                                status.update(
                                    label=f"No artists found with names containing all words: {', '.join(search_words)}",
                                    state="warning",
                                )
                                st.session_state.search_results = None
                        else:
                            status.update(
                                label="No valid artist data found.", state="warning"
                            )
                            st.session_state.search_results = None
                    else:
                        status.update(
                            label=f"No artists found for '{search_query}'",
                            state="warning",
                        )
                        st.session_state.search_results = None

                except RateLimitExceeded:
                    status.update(
                        label="⏱️ Rate limit exceeded - please try again later",
                        state="error",
                    )
                    display_rate_limit_error()
                    st.session_state.search_results = None
                except Exception as e:
                    status.update(label=f"Error: {str(e)}", state="error")
                    st.session_state.search_results = None

        # Display search results with "Add to Worklist" buttons
        if st.session_state.search_results is not None:
            filtered_df = st.session_state.search_results
            search_query_done = st.session_state.search_query_done

            st.subheader(f"🎤 Search Results for '{search_query_done}'")

            for idx, (_, artist) in enumerate(filtered_df.iterrows()):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(
                        f"**{artist['Artist Name']}** ({artist['Followers']} followers) - ID: `{artist['Artist ID']}`"
                    )
                with col2:
                    # Check if artist is already in worklist
                    already_added = any(
                        a["artist_id"] == artist["Artist ID"]
                        for a in st.session_state.artist_worklist
                    )

                    if already_added:
                        st.write("✅ Added")
                    else:
                        if st.button(f"➕ Add", key=f"add_{artist['Artist ID']}_{idx}"):
                            st.session_state.artist_worklist.append(
                                {
                                    "artist_name": artist["Artist Name"],
                                    "artist_id": artist["Artist ID"],
                                    "followers": artist["Followers"],
                                }
                            )
                            st.rerun()

    else:
        # Direct ID/URL input (modified from original functionality)
        artist_input = st.text_area(
            "Enter Spotify artist URIs, URLs, or IDs (one per line)"
        )

        # No additional market selector needed for multiple markets mode
        market = "US"  # Default for artist search (album data uses selected_markets)
        add_clicked = st.button("➕ Add to Worklist")

        if add_clicked and artist_input:
            artist_ids = parse_multi_spotify_ids_secure(artist_input, "artist")
            if not artist_ids:
                st.error("Please enter at least one valid artist ID.")
            else:
                # Get artist names for the worklist
                spotify_client = get_authenticated_client()
                if spotify_client:
                    for artist_id in artist_ids:
                        # Check if already in worklist
                        if not any(
                            a["artist_id"] == artist_id
                            for a in st.session_state.artist_worklist
                        ):
                            try:
                                artist_data = spotify_client._make_request(
                                    f"artists/{artist_id}"
                                )
                                artist_name = (
                                    artist_data.get("name", "Unknown Artist")
                                    if artist_data
                                    else "Unknown Artist"
                                )
                                followers = (
                                    artist_data.get("followers", {}).get("total", 0)
                                    if artist_data
                                    else 0
                                )
                                followers_formatted = (
                                    f"{followers:,}" if followers else "0"
                                )

                                st.session_state.artist_worklist.append(
                                    {
                                        "artist_name": artist_name,
                                        "artist_id": artist_id,
                                        "followers": followers_formatted,
                                    }
                                )
                            except:
                                st.session_state.artist_worklist.append(
                                    {
                                        "artist_name": "Unknown Artist",
                                        "artist_id": artist_id,
                                        "followers": "0",
                                    }
                                )
                    st.rerun()

    # Display Artist Worklist
    if st.session_state.artist_worklist:
        st.divider()
        st.subheader(
            f"📋 Artist Worklist ({len(st.session_state.artist_worklist)} artists)"
        )

        # Display worklist
        for idx, artist in enumerate(st.session_state.artist_worklist):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(
                    f"**{artist['artist_name']}** ({artist['followers']} followers) - ID: `{artist['artist_id']}`"
                )
            with col2:
                if st.button(f"❌ Remove", key=f"remove_{artist['artist_id']}_{idx}"):
                    st.session_state.artist_worklist.pop(idx)
                    st.rerun()

        # Action buttons
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("🗑️ Clear All Artists"):
                st.session_state.artist_worklist = []
                st.rerun()
        with col3:
            # Right-aligned Generate Catalog Report button
            if st.button("📦 Generate Catalog Report"):
                # Use default market for artist search (album data uses selected_markets)
                current_market = "US"

                # Store the catalog search info and clear search results
                st.session_state.pending_multiple_catalog_search = {
                    "artist_worklist": st.session_state.artist_worklist.copy(),
                    "market": current_market,
                    "market_view": market_view,
                    "selected_markets": selected_markets,
                }
                st.session_state.search_results = None
                st.rerun()

    # Execute pending catalog search at the very end (after all UI elements)
    if st.session_state.pending_multiple_catalog_search is not None:
        pending = st.session_state.pending_multiple_catalog_search
        st.session_state.pending_multiple_catalog_search = None  # Clear it immediately
        perform_multiple_catalog_search(
            pending["artist_worklist"], 
            pending["market"], 
            pending.get("market_view", "single"),
            pending.get("selected_markets", [pending["market"]])
        )

    # Display results from session state (if any)
    results, display_data, metadata, search_completed = get_search_results("multiple_artist_catalog")
    
    if search_completed and results is not None:
        if not results.empty:
            # Full-width download button for all data
            st.markdown("---")  # Add a separator for visual clarity
            download_filename = metadata.get("download_filename", "Multiple_Artists_Releases.xlsx")
            track_count = metadata.get("track_count", len(results))
            create_download_button(
                df=results,
                label=f"📦 Download All Data to Excel ({track_count} tracks)",
                file_name=download_filename,
                key="download_all_albums",
            )


if __name__ == "__main__":
    main()
