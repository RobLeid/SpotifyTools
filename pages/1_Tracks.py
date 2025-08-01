import pandas as pd
import streamlit as st
from utils.rate_limiting import RateLimitExceeded
from utils.validation import parse_multi_spotify_ids_secure
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
    st.title("🎵 Spotify Track Info")

    # Initialize session state for this page
    init_page_session_state("tracks")

    # Display Spotify attribution (required by Spotify Developer Terms)
    display_spotify_attribution()

    user_input = st.text_area("Enter Spotify track IDs, URIs, or URLs (one per line)")

    if st.button("🔍 Get Track Info"):
        if not user_input.strip():
            st.warning("Please enter at least one track ID, URI, or URL.")
            return

        # Clear previous search results
        clear_search_results("tracks")

        track_ids = parse_multi_spotify_ids_secure(user_input, "track")
        if not track_ids:
            st.warning("No valid track IDs found.")
            return

        # Show processing info
        num_batches = (len(track_ids) + 49) // 50  # 50 tracks per batch
        processing_info = st.empty()
        processing_info.info(
            f"🎯 Processing {len(track_ids)} tracks in {num_batches} batch(es)"
        )

        # Initialize variables for results
        tracks_df = None

        with st.status("⏳ Processing...", expanded=True) as status:
            spotify_client = get_authenticated_client()
            if not spotify_client:
                return

            try:
                status.update(
                    label="Fetching track data...", state="running", expanded=True
                )

                # Use improved API client with better error handling
                tracks = spotify_client.fetch_tracks_by_ids(track_ids)

                if tracks:
                    df = pd.DataFrame(process_track_data(tracks))
                    if not df.empty:
                        tracks_df = df
                        # Store results in session state
                        store_search_results(
                            "tracks",
                            results=df,
                            metadata={
                                "track_count": len(df),
                                "download_filename": f"Spotify_Tracks_{len(df)}_tracks.xlsx"
                            }
                        )
                        status.update(
                            label="✅ Done!", state="complete", expanded=False
                        )
                    else:
                        status.update(
                            label="No valid track data found.",
                            state="warning",
                            expanded=False,
                        )
                else:
                    status.update(
                        label="No valid tracks found.", state="warning", expanded=False
                    )

            except RateLimitExceeded:
                status.update(
                    label="⏱️ Rate limit exceeded - please try again in a few minutes",
                    state="error",
                    expanded=False,
                )
                display_rate_limit_error()
            except Exception as e:
                status.update(label=f"Error: {str(e)}", state="error", expanded=False)

        # Note: Display logic moved outside conditional to use session state

    # Display results from session state (if any)
    results, display_data, metadata, search_completed = get_search_results("tracks")
    
    if search_completed and results is not None:
        styled_df = style_dataframe_columns(results)
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        download_filename = metadata.get("download_filename", "spotify_tracks.xlsx")
        create_download_button(
            df=results,
            label="📥 Download as Excel",
            file_name=download_filename,
            key="download_tracks",
        )


if __name__ == "__main__":
    main()
