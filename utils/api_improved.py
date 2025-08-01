"""
Improved Spotify API client with better error handling, rate limiting, and separate album type queries.
"""

import requests
import time
from typing import List, Dict, Optional, Tuple, Any
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed

from .constants import (
    SPOTIFY_BASE_URL,
    DEFAULT_TIMEOUT,
    MAX_RETRIES,
    TRACKS_BATCH_SIZE,
    ALBUMS_BATCH_SIZE,
    PLAYLIST_ITEMS_LIMIT,
    ARTIST_ALBUMS_LIMIT,
    INTER_BATCH_DELAY,
    INTER_PAGE_DELAY,
    MAX_WORKERS,
    ALBUM_TYPES,
    DEFAULT_MARKET,
    ERROR_MESSAGES
)
from .rate_limiting import RetryHandler, RateLimitExceeded, handle_spotify_response_errors
from .validation import validate_spotify_id, sanitize_input


class SpotifyAPIClient:
    """
    Improved Spotify API client with comprehensive error handling and rate limiting.
    """
    
    def __init__(self, access_token: str):
        """
        Initialize the Spotify API client.
        
        Args:
            access_token: Spotify access token
        """
        self.access_token = access_token
        self.headers = {"Authorization": f"Bearer {access_token}"}
        self.retry_handler = RetryHandler(max_retries=MAX_RETRIES)
    
    def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        timeout: int = DEFAULT_TIMEOUT
    ) -> Optional[Dict]:
        """
        Make a request to the Spotify API with retry logic.
        
        Args:
            endpoint: API endpoint (without base URL)
            params: Query parameters
            timeout: Request timeout in seconds
            
        Returns:
            API response data or None if failed
        """
        url = f"{SPOTIFY_BASE_URL}/{endpoint.lstrip('/')}"
        
        def _request():
            response = requests.get(url, headers=self.headers, params=params, timeout=timeout)
            handle_spotify_response_errors(response)
            return response.json()
        
        try:
            return self.retry_handler.execute_with_retry(_request)
        except RateLimitExceeded as e:
            st.error(f"⏱️ {e}")
            return None
        except Exception as e:
            st.error(f"❌ Request failed: {e}")
            return None
    
    def fetch_tracks_by_ids(self, track_ids: List[str], status_callback=None) -> List[Dict]:
        """
        Fetch multiple tracks by their IDs using optimal batching.
        
        Args:
            track_ids: List of Spotify track IDs
            status_callback: Optional callback function to update status (instead of st.write)
            
        Returns:
            List of track data dictionaries
        """
        if not track_ids:
            return []
        
        # Validate all track IDs
        valid_ids = [tid for tid in track_ids if validate_spotify_id(tid, 'track')]
        if len(valid_ids) != len(track_ids):
            st.warning(f"⚠️ Filtered out {len(track_ids) - len(valid_ids)} invalid track IDs")
        
        all_tracks = []
        total_batches = (len(valid_ids) + TRACKS_BATCH_SIZE - 1) // TRACKS_BATCH_SIZE
        
        for i in range(0, len(valid_ids), TRACKS_BATCH_SIZE):
            batch = valid_ids[i:i + TRACKS_BATCH_SIZE]
            batch_ids = ",".join(batch)
            
            batch_num = i//TRACKS_BATCH_SIZE + 1
            if status_callback:
                status_callback(f"Fetching track batch {batch_num}/{total_batches}: {len(batch)} tracks")
            else:
                st.write(f"📥 Fetching track batch {batch_num}/{total_batches}: {len(batch)} tracks")
            
            response = self._make_request(f"tracks?ids={batch_ids}")
            
            if response and "tracks" in response:
                valid_tracks = [track for track in response["tracks"] if track is not None]
                all_tracks.extend(valid_tracks)
                if not status_callback:
                    st.success(f"✅ Batch {batch_num} complete: {len(valid_tracks)} tracks retrieved")
            else:
                if not status_callback:
                    st.warning(f"⚠️ Batch {batch_num} returned no data")
            
            # Delay between batches
            if i + TRACKS_BATCH_SIZE < len(valid_ids):
                time.sleep(INTER_BATCH_DELAY)
        
        if not status_callback:
            st.info(f"📊 Retrieved {len(all_tracks)} tracks total")
        return all_tracks
    
    def fetch_album_details(self, album_id: str) -> Tuple[Optional[Dict], List[Dict]]:
        """
        Fetch album details including all tracks.
        
        Args:
            album_id: Spotify album ID
            
        Returns:
            Tuple of (album_data, full_tracks)
        """
        if not validate_spotify_id(album_id, 'album'):
            st.error(f"Invalid album ID: {album_id}")
            return None, []
        
        # Get album metadata
        album_data = self._make_request(f"albums/{album_id}")
        
        if not album_data:
            return None, []
        
        # Extract track IDs from album
        track_items = album_data.get("tracks", {}).get("items", [])
        track_ids = [track["id"] for track in track_items if track.get("id")]
        
        # Fetch full track data
        full_tracks = self.fetch_tracks_by_ids(track_ids)
        
        return album_data, full_tracks
    
    def fetch_album_tracks_all(self, album_id: str) -> List[Dict]:
        """
        Fetch ALL tracks from an album, handling pagination for albums with >50 tracks.
        
        Args:
            album_id: Spotify album ID
            
        Returns:
            List of all track items from the album
        """
        all_tracks = []
        offset = 0
        limit = 50
        
        while True:
            response = self._make_request(f"albums/{album_id}/tracks?limit={limit}&offset={offset}")
            
            if not response:
                break
                
            items = response.get("items", [])
            all_tracks.extend(items)
            
            # Check if there are more tracks
            if response.get("next") is None:
                break
                
            offset += limit
            time.sleep(INTER_PAGE_DELAY)  # Small delay between pagination requests
            
        return all_tracks
    
    def fetch_playlist_tracks(self, playlist_id: str) -> Tuple[Optional[Dict], List[Dict]]:
        """
        Fetch playlist metadata and all tracks.
        
        Args:
            playlist_id: Spotify playlist ID
            
        Returns:
            Tuple of (playlist_data, track_items)
        """
        if not validate_spotify_id(playlist_id, 'playlist'):
            st.error(f"Invalid playlist ID: {playlist_id}")
            return None, []
        
        # Get playlist metadata
        playlist_data = self._make_request(f"playlists/{playlist_id}")
        
        if not playlist_data:
            return None, []
        
        # Get all playlist tracks with pagination
        all_tracks = []
        url = f"playlists/{playlist_id}/tracks"
        page = 1
        
        while url:
            st.write(f"📥 Fetching playlist page {page}...")
            
            # Extract just the endpoint from the URL if it's a full URL
            if url.startswith('http'):
                endpoint = url.replace(SPOTIFY_BASE_URL + '/', '')
            else:
                endpoint = url
            
            response = self._make_request(endpoint, {"limit": PLAYLIST_ITEMS_LIMIT})
            
            if not response:
                break
            
            items = response.get("items", [])
            all_tracks.extend(items)
            
            # Check for next page
            next_url = response.get("next")
            if next_url:
                url = next_url.replace(SPOTIFY_BASE_URL + '/', '')
            else:
                url = None
                
            page += 1
            
            # Delay between pages
            if url:
                time.sleep(INTER_PAGE_DELAY)
        
        # Fetch full track data
        track_ids = []
        for item in all_tracks:
            if item.get("track") and item["track"].get("id"):
                track_ids.append(item["track"]["id"])
        
        if track_ids:
            full_tracks = self.fetch_tracks_by_ids(track_ids)
            track_map = {track["id"]: track for track in full_tracks if track}
            
            # Update playlist items with full track data
            for item in all_tracks:
                if item.get("track") and item["track"].get("id"):
                    track_id = item["track"]["id"]
                    if track_id in track_map:
                        item["track"] = track_map[track_id]
        
        return playlist_data, all_tracks
    
    def fetch_artist_metadata_and_top_tracks(self, artist_id: str, market: str = DEFAULT_MARKET) -> Tuple[Optional[Dict], List[Dict]]:
        """
        Fetch artist metadata and top tracks.
        
        Args:
            artist_id: Spotify artist ID
            market: Market code (ISO 3166-1 alpha-2)
            
        Returns:
            Tuple of (artist_data, top_tracks)
        """
        if not validate_spotify_id(artist_id, 'artist'):
            st.error(f"Invalid artist ID: {artist_id}")
            return None, []
        
        # Get artist metadata
        artist_data = self._make_request(f"artists/{artist_id}")
        
        # Get top tracks
        top_tracks_data = self._make_request(
            f"artists/{artist_id}/top-tracks",
            params={"market": market}
        )
        
        if not artist_data or not top_tracks_data:
            return None, []
        
        return artist_data, top_tracks_data.get("tracks", [])
    
    def fetch_artist_albums_by_type(self, artist_id: str, album_type: str, market: str = DEFAULT_MARKET, quiet: bool = False) -> List[Dict]:
        """
        Fetch artist albums for a specific type (album, single, compilation).
        
        Args:
            artist_id: Spotify artist ID
            album_type: Type of album (album, single, compilation)
            market: Market code
            quiet: If True, suppress status messages
            
        Returns:
            List of album data dictionaries
        """
        if not validate_spotify_id(artist_id, 'artist'):
            if not quiet:
                st.error(f"Invalid artist ID: {artist_id}")
            return []
        
        if album_type not in ALBUM_TYPES:
            if not quiet:
                st.error(f"Invalid album type: {album_type}")
            return []
        
        all_albums = []
        url = f"artists/{artist_id}/albums"
        params = {
            "market": market,
            "limit": ARTIST_ALBUMS_LIMIT,
            "include_groups": album_type  # Single type for better results
        }
        page = 1
        
        while url:
            if not quiet:
                st.write(f"📥 Fetching {album_type} albums page {page}...")
            
            # Extract just the endpoint from the URL if it's a full URL
            if url.startswith('http'):
                endpoint = url.replace(SPOTIFY_BASE_URL + '/', '')
                current_params = None  # Params are in the URL
            else:
                endpoint = url
                current_params = params
            
            response = self._make_request(endpoint, current_params)
            
            if not response:
                break
            
            albums = response.get("items", [])
            all_albums.extend(albums)
            
            # Check for next page
            next_url = response.get("next")
            if next_url:
                url = next_url.replace(SPOTIFY_BASE_URL + '/', '')
            else:
                url = None
                
            page += 1
            
            # Delay between pages
            if url:
                time.sleep(INTER_PAGE_DELAY)
        
        if not quiet:
            st.info(f"📊 Found {len(all_albums)} {album_type} albums")
        return all_albums
    
    def fetch_artist_albums_comprehensive(self, artist_id: str, market: str = DEFAULT_MARKET, quiet: bool = False) -> List[Dict]:
        """
        Fetch ALL artist albums by querying each type separately.
        This ensures compilations are included.
        
        Args:
            artist_id: Spotify artist ID
            market: Market code
            quiet: If True, suppress status messages
            
        Returns:
            List of all album data dictionaries
        """
        if not validate_spotify_id(artist_id, 'artist'):
            error_msg = f"Invalid artist ID: {artist_id}"
            if not quiet:
                st.error(error_msg)
            else:
                st.warning(error_msg)  # Always show validation errors even in quiet mode
            return []
        
        all_albums = []
        seen_ids = set()
        
        # Query each album type separately
        for album_type in ALBUM_TYPES:
            if not quiet:
                st.write(f"🎵 Fetching {album_type} albums...")
            albums = self.fetch_artist_albums_by_type(artist_id, album_type, market, quiet=quiet)
            
            # Deduplicate albums
            for album in albums:
                if album.get("id") not in seen_ids:
                    all_albums.append(album)
                    seen_ids.add(album.get("id"))
        
        if not quiet:
            st.success(f"✅ Found {len(all_albums)} total albums across all types")
        return all_albums
    
    def fetch_multiple_artists_catalogs(
        self,
        artist_ids: List[str],
        market: str = DEFAULT_MARKET,
        max_workers: int = MAX_WORKERS,
        status_callback=None
    ) -> Dict[str, Any]:
        """
        Fetch catalogs for multiple artists with improved threading.
        
        Args:
            artist_ids: List of Spotify artist IDs
            market: Market code
            max_workers: Maximum concurrent threads
            status_callback: Optional callback function to update status
            
        Returns:
            Dictionary mapping artist IDs to their catalog data
        """
        # Validate all artist IDs
        valid_ids = [aid for aid in artist_ids if validate_spotify_id(aid, 'artist')]
        if len(valid_ids) != len(artist_ids):
            st.warning(f"⚠️ Filtered out {len(artist_ids) - len(valid_ids)} invalid artist IDs")
        
        results = {}
        
        def process_artist(artist_id: str) -> Tuple[str, Dict]:
            """Process a single artist's catalog"""
            try:
                # Process artist
                
                # Get albums using comprehensive method (with suppressed output for multiple artist mode)
                original_status_callback = status_callback
                def quiet_status(message):
                    if original_status_callback:
                        original_status_callback(f"Artist {artist_id} - {message}")
                
                albums = self.fetch_artist_albums_comprehensive(artist_id, market, quiet=True)
                
                if not albums:
                    st.warning(f"⚠️ No albums found for artist {artist_id}")
                    return artist_id, {"albums": [], "album_data": {}, "tracks": [], "failed_albums": []}
                
                # Get album details in batches
                album_ids = [album["id"] for album in albums]
                album_data = {}
                all_track_ids = []
                failed_albums = []
                
                # Process albums in batches
                total_album_batches = (len(album_ids) + ALBUMS_BATCH_SIZE - 1) // ALBUMS_BATCH_SIZE
                for i in range(0, len(album_ids), ALBUMS_BATCH_SIZE):
                    batch_ids = album_ids[i:i + ALBUMS_BATCH_SIZE]
                    batch_ids_str = ",".join(batch_ids)
                    
                    batch_num = i//ALBUMS_BATCH_SIZE + 1
                    if status_callback:
                        status_callback(f"Processing artist {artist_id} - album batch {batch_num}/{total_album_batches}")
                    
                    response = self._make_request(f"albums?ids={batch_ids_str}")
                    
                    if response and "albums" in response:
                        for j, album in enumerate(response["albums"]):
                            if album:
                                album_data[album["id"]] = album
                                # Collect all track IDs
                                track_items = album.get("tracks", {}).get("items", [])
                                
                                # Check if album has more tracks than returned (>50)
                                total_tracks = album.get("total_tracks", 0)
                                if total_tracks > 50 and len(track_items) == 50:
                                    # Fetch all tracks for this album
                                    all_album_tracks = self.fetch_album_tracks_all(album["id"])
                                    track_items = all_album_tracks
                                    # Update the album data with all tracks
                                    album["tracks"]["items"] = all_album_tracks
                                
                                track_ids = [track["id"] for track in track_items if track.get("id")]
                                all_track_ids.extend(track_ids)
                            else:
                                # Track failed album
                                failed_album_id = batch_ids[j] if j < len(batch_ids) else "unknown"
                                failed_albums.append(failed_album_id)
                                st.warning(f"⚠️ Failed to get data for album {failed_album_id}")
                    else:
                        # All albums in this batch failed
                        failed_albums.extend(batch_ids)
                        st.error(f"❌ Failed to fetch album batch {batch_num} for artist {artist_id}")
                    
                    # Delay between album batches
                    time.sleep(INTER_BATCH_DELAY)
                
                if not all_track_ids:
                    st.warning(f"⚠️ No track IDs found for artist {artist_id}")
                    return artist_id, {"albums": albums, "album_data": album_data, "tracks": [], "failed_albums": failed_albums}
                
                # Fetch all tracks for this artist
                if status_callback:
                    def track_status_callback(message):
                        status_callback(f"Artist {artist_id} - {message}")
                    all_tracks = self.fetch_tracks_by_ids(all_track_ids, status_callback=track_status_callback)
                else:
                    all_tracks = self.fetch_tracks_by_ids(all_track_ids)
                
                return artist_id, {
                    "albums": albums,
                    "album_data": album_data,
                    "tracks": all_tracks,
                    "failed_albums": failed_albums
                }
                
            except Exception as e:
                st.error(f"❌ Error processing artist {artist_id}: {e}")
                return artist_id, {"albums": [], "album_data": {}, "tracks": [], "failed_albums": []}
        
        # Process artists with limited concurrency
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_artist = {executor.submit(process_artist, artist_id): artist_id for artist_id in valid_ids}
            completed_count = 0
            
            for future in as_completed(future_to_artist):
                artist_id, result = future.result()
                results[artist_id] = result
                completed_count += 1
                
                if status_callback:
                    status_callback(f"Completed artist {completed_count}/{len(valid_ids)}: {artist_id}")
                else:
                    st.success(f"✅ Completed artist {completed_count}/{len(valid_ids)}: {artist_id}")
        
        return results
    
    def search_artists(self, query: str, limit: int = 20, market: str = DEFAULT_MARKET) -> List[Dict]:
        """
        Search for artists by name.
        
        Args:
            query: Search query (artist name or partial name)
            limit: Maximum number of results (1-50)
            market: Market code
            
        Returns:
            List of artist data dictionaries
        """
        if not query or not query.strip():
            st.error("Search query cannot be empty")
            return []
        
        # Limit range validation
        limit = max(1, min(limit, 50))
        
        try:
            params = {
                "q": query.strip(),
                "type": "artist",
                "limit": limit,
                "market": market
            }
            
            response = self._make_request("search", params)
            
            if response and "artists" in response and "items" in response["artists"]:
                return response["artists"]["items"]
            else:
                st.warning(f"No artists found for query: '{query}'")
                return []
                
        except Exception as e:
            st.error(f"Error searching for artists: {e}")
            return []

