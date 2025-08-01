import pandas as pd
import streamlit as st

from utils.rate_limiting import RateLimitExceeded
from utils.validation import parse_multi_spotify_ids_secure
from utils.data_processing import process_album_track_data
from utils.ui_components import (
    create_download_button,
    display_processing_info,
    display_rate_limit_error,
    display_album_row,
    display_spotify_attribution,
)
from utils.common_operations import get_authenticated_client
from utils.session_state import (
    init_page_session_state,
    store_search_results,
    get_search_results,
    clear_search_results,
)


def main():
    st.title("💿 Spotify Album Info")

    # Initialize session state for this page
    init_page_session_state("albums")

    # Display Spotify attribution (required by Spotify Developer Terms)
    display_spotify_attribution()

    user_input = st.text_area(
        "Enter multiple Spotify album URIs, URLs, or IDs (one per line)"
    )

    if st.button("🔍 Get Album Info"):
        if not user_input:
            st.warning("Please enter at least one album ID, URI, or URL.")
            return

        # Clear previous search results
        clear_search_results("albums")

        album_ids = parse_multi_spotify_ids_secure(user_input, "album")
        if not album_ids:
            st.warning("No valid album IDs found.")
            return

        # Show processing info
        processing_info = st.empty()
        processing_info.info(
            f"🎯 Processing {len(album_ids)} album{'s' if len(album_ids) != 1 else ''}"
        )

        spotify_client = get_authenticated_client()
        if not spotify_client:
            return

        all_dataframes = []
        album_details_list = []

        with st.status("⏳ Processing albums...", expanded=False) as status:
            try:
                # STEP 1: Batch fetch all album details
                status.update(label="Fetching album details...", state="running")
                album_details_map = {}
                all_track_ids = []
                track_to_album_map = {}

                # Process albums in batches of 20 (Spotify API limit)
                for i in range(0, len(album_ids), 20):
                    batch_ids = album_ids[i : i + 20]

                    # Get detailed album information in one API call
                    album_batch_data = spotify_client._make_request(
                        f"albums?ids={','.join(batch_ids)}"
                    )

                    if album_batch_data and "albums" in album_batch_data:
                        for album_data in album_batch_data["albums"]:
                            if album_data:
                                album_details_map[album_data["id"]] = album_data
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

                # STEP 2: Batch fetch ALL track details at once
                status.update(
                    label=f"Fetching {len(all_track_ids)} track details...",
                    state="running",
                )
                all_full_tracks = spotify_client.fetch_tracks_by_ids(all_track_ids)

                # Create a map of track ID to full track data
                track_data_map = {
                    track["id"]: track for track in all_full_tracks if track
                }

                # STEP 3: Process albums with pre-fetched data
                status.update(label="Processing albums...", state="running")
                for i, album_id in enumerate(album_ids):
                    status.update(
                        label=f"Processing album {i+1}/{len(album_ids)}...",
                        state="running",
                    )

                    if album_id in album_details_map:
                        album_data = album_details_map[album_id]
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
                            simplified_data = process_album_track_data(
                                album_data, track_items, full_tracks
                            )
                            df = pd.DataFrame(simplified_data)
                            all_dataframes.append(df)
                            album_details_list.append(
                                {
                                    "df": df,
                                    "name": album_data.get("name", "Unknown Album"),
                                    "image_url": (
                                        album_data["images"][0]["url"]
                                        if album_data.get("images")
                                        else None
                                    ),
                                    "id": album_data["id"],
                                    "available_markets": album_data.get(
                                        "available_markets", []
                                    ),
                                }
                            )
                        else:
                            st.warning(
                                f"⚠️ Failed to process album {i+1} - no track data found"
                            )
                    else:
                        st.warning(f"⚠️ Failed to process album {i+1} - album not found")

                status.update(
                    label=f"✅ Optimized processing complete! Used only {len(all_track_ids)//50 + len(album_ids)//20 + 2} API calls instead of {len(album_ids)*2}+",
                    state="complete",
                    expanded=False,
                )

            except RateLimitExceeded:
                status.update(
                    label="⏱️ Rate limit exceeded - returning partial results",
                    state="error",
                    expanded=False,
                )
                display_rate_limit_error()
            except Exception as e:
                st.error(f"Error: {str(e)}")

        if all_dataframes:
            processing_info.empty()  # Remove the processing message
            status.update(
                label="✅ Done processing albums!", state="complete", expanded=False
            )

            combined_df = pd.concat(all_dataframes, ignore_index=True)
            
            # Store results in session state
            store_search_results(
                "albums",
                results=combined_df,
                display_data=album_details_list,
                metadata={
                    "album_count": len(album_details_list),
                    "download_filename": f"Albums_Data_{len(album_details_list)}_albums.xlsx"
                }
            )

        # Note: Display logic moved outside conditional to use session state

    # Display results from session state (if any)
    results, display_data, metadata, search_completed = get_search_results("albums")
    
    if search_completed and results is not None:
        # Display download button for combined results
        if not results.empty:
            download_filename = metadata.get("download_filename", "All_Albums_Tracks.xlsx")
            create_download_button(
                df=results,
                label="📦 Download All Albums to Excel",
                file_name=download_filename,
                key="download_all_albums",
            )

        # Display individual album rows
        if display_data:
            for album_details in display_data:
                album_data = {
                    "name": album_details["name"],
                    "images": (
                        [{"url": album_details["image_url"]}]
                        if album_details["image_url"]
                        else []
                    ),
                }
                display_album_row(
                    album_data,
                    album_details["df"],
                    album_details["id"],
                    available_markets=album_details.get("available_markets"),
                    show_availability_tooltip=True,
                )


if __name__ == "__main__":
    main()
