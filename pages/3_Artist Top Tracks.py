import pandas as pd
import streamlit as st
from PIL import Image
from urllib.request import urlopen

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
    st.title("🎤 Spotify Artist Top Tracks")

    # Initialize session state for this page
    init_page_session_state("artist_top_tracks")

    # Display Spotify attribution (required by Spotify Developer Terms)
    display_spotify_attribution()

    user_input = st.text_input("Enter a Spotify artist URI, URL, or ID")

    if st.button("🔍 Get Top Tracks") and user_input:
        # Clear previous search results
        clear_search_results("artist_top_tracks")
        
        artist_id = parse_spotify_id_secure(user_input, "artist")
        if not artist_id:
            return

        # Initialize variables for results
        tracks_df = None
        artist_name = None
        artist_image_url = None

        with st.status("⏳ Fetching artist info...", expanded=False) as status:
            spotify_client = get_authenticated_client()
            if not spotify_client:
                return

            try:
                status.update(
                    label="Fetching artist data...", state="running", expanded=True
                )
                artist_data, top_tracks = (
                    spotify_client.fetch_artist_metadata_and_top_tracks(
                        artist_id, market="US"
                    )
                )

                if artist_data and top_tracks:
                    artist_name = artist_data.get("name", "Unknown Artist")
                    artist_image_url = (
                        artist_data["images"][0]["url"]
                        if artist_data.get("images")
                        else None
                    )

                    simplified_data = process_track_data(top_tracks)
                    df = pd.DataFrame(simplified_data)

                    if not df.empty:
                        tracks_df = df
                        # Store results in session state
                        store_search_results(
                            "artist_top_tracks",
                            results=df,
                            metadata={
                                "artist_name": artist_name,
                                "artist_image_url": artist_image_url,
                                "track_count": len(df),
                                "download_filename": f"{artist_name}_Top_Tracks.xlsx"
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
                        label="No top tracks found or invalid artist.",
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
    results, display_data, metadata, search_completed = get_search_results("artist_top_tracks")
    
    if search_completed and results is not None:
        artist_name = metadata.get("artist_name", "Unknown Artist")
        artist_image_url = metadata.get("artist_image_url")
        
        st.info(f"📊 Found {len(results)} top tracks for {artist_name}")
        styled_df = style_dataframe_columns(results)
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        download_filename = metadata.get("download_filename", "artist_top_tracks.xlsx")
        create_download_button(
            df=results,
            label="📥 Download as Excel",
            file_name=download_filename,
            key="download_top_tracks",
        )

        # Display artist image
        if artist_image_url and artist_name:
            col1, col2, col3 = st.columns(3)
            with col2:
                try:
                    image = Image.open(urlopen(artist_image_url))
                    st.image(image, caption=artist_name, width=300)
                except:
                    st.write(f"🎤 {artist_name}")


if __name__ == "__main__":
    main()
