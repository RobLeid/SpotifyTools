import pandas as pd
import streamlit as st

from utils.rate_limiting import RateLimitExceeded
from utils.validation import parse_spotify_id_secure
from utils.data_processing import process_track_data
from utils.ui_components import (
    create_download_button,
    display_processing_info,
    display_rate_limit_error,
    display_spotify_attribution,
    style_dataframe_columns,
)
from utils.common_operations import get_authenticated_client
from utils.session_state import (
    init_page_session_state,
    store_search_results,
    get_search_results,
    clear_search_results,
)


def main():
    st.title("📃 Spotify Playlist Info")
    st.caption("Note: this does not work for Spotify generated playlists...")

    # Initialize session state for this page
    init_page_session_state("user_playlists")

    # Display Spotify attribution (required by Spotify Developer Terms)
    display_spotify_attribution()

    user_input = st.text_input("Enter a Spotify playlist URI, URL, or ID")

    if st.button("🔍 Get Playlist Info") and user_input:
        # Clear previous search results
        clear_search_results("user_playlists")
        
        playlist_id = parse_spotify_id_secure(user_input, "playlist")
        if not playlist_id:
            return

        # Initialize variables for results
        tracks_df = None
        playlist_name = None
        playlist_image_url = None
        track_count = 0

        with st.status("⏳ Fetching playlist...", expanded=False) as status:
            spotify_client = get_authenticated_client()
            if not spotify_client:
                return

            try:
                status.update(
                    label="Fetching playlist...", state="running", expanded=False
                )
                meta_data, playlist_tracks = spotify_client.fetch_playlist_tracks(
                    playlist_id
                )

                if meta_data and playlist_tracks:
                    playlist_name = meta_data.get("name", "Unknown Playlist")
                    playlist_image_url = (
                        meta_data["images"][0]["url"]
                        if meta_data.get("images")
                        else None
                    )
                    track_count = len(playlist_tracks)

                    tracks_only = [
                        item.get("track")
                        for item in playlist_tracks
                        if item.get("track")
                    ]
                    simplified_data = process_track_data(tracks_only)
                    df = pd.DataFrame(simplified_data)

                    if not df.empty:
                        tracks_df = df
                        # Store results in session state
                        store_search_results(
                            "user_playlists",
                            results=df,
                            metadata={
                                "playlist_name": playlist_name,
                                "playlist_image_url": playlist_image_url,
                                "track_count": track_count,
                                "download_filename": f"{playlist_name}_Tracks.xlsx"
                            }
                        )
                        status.update(
                            label="✅ Done!", state="complete", expanded=False
                        )
                    else:
                        status.update(
                            label="No track data found.",
                            state="warning",
                            expanded=False,
                        )
                else:
                    status.update(
                        label="No tracks found or invalid playlist.",
                        state="warning",
                        expanded=False,
                    )

            except RateLimitExceeded:
                status.update(
                    label="⏱️ Rate limit exceeded - please try again later",
                    state="error",
                    expanded=False,
                )
                display_rate_limit_error()
            except Exception as e:
                status.update(label=f"Error: {str(e)}", state="error", expanded=False)

        # Note: Display logic moved outside conditional to use session state

    # Display results from session state (if any)
    results, display_data, metadata, search_completed = get_search_results("user_playlists")
    
    if search_completed and results is not None:
        playlist_name = metadata.get("playlist_name", "Unknown Playlist")
        playlist_image_url = metadata.get("playlist_image_url")
        track_count = metadata.get("track_count", len(results))
        
        st.info(f"📊 Playlist contains {track_count} tracks")
        styled_df = style_dataframe_columns(results)
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        download_filename = metadata.get("download_filename", "playlist_tracks.xlsx")
        create_download_button(
            df=results,
            label="📥 Download as Excel",
            file_name=download_filename,
            key="download_playlist",
        )

        # Display playlist image
        if playlist_image_url and playlist_name:
            col1, col2, col3 = st.columns(3)
            with col2:
                try:
                    st.image(playlist_image_url, caption=playlist_name, width=300)
                except:
                    st.write(f"🎵 {playlist_name}")


if __name__ == "__main__":
    main()
