"""
Global market aggregation utility for collecting unique albums across all Spotify markets.
This module provides functionality to query all available markets and aggregate unique album IDs.
"""

from typing import List, Dict, Set, Optional, Tuple, Any, Callable
import streamlit as st
from .constants import MARKETS, ALBUM_TYPES
from .api_improved import SpotifyAPIClient


class GlobalMarketAggregator:
    """
    Aggregates album data across all Spotify markets for a given artist.

    This class handles the logic for querying multiple markets sequentially,
    deduplicating albums, and organizing them by type (album, single, compilation).
    """

    def __init__(self, spotify_client: SpotifyAPIClient):
        """
        Initialize the aggregator with a Spotify API client.

        Args:
            spotify_client: Authenticated SpotifyAPIClient instance
        """
        self.spotify_client = spotify_client
        self.unique_album_ids: Set[str] = set()
        self.albums_by_type: Dict[str, List[Dict]] = {
            "album": [],
            "single": [],
            "compilation": [],
        }

    def get_all_markets(self) -> List[str]:
        """
        Get the list of all available Spotify markets.

        Returns:
            List of market codes (e.g., ['US', 'GB', 'CA', ...])
        """
        return MARKETS.copy()

    def aggregate_artist_albums(
        self,
        artist_id: str,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> Tuple[Dict[str, List[Dict]], int]:
        """
        Aggregate all unique albums for an artist across all markets.

        Args:
            artist_id: Spotify artist ID
            progress_callback: Optional callback function for progress updates
                             Should accept (message: str, current: int, total: int)

        Returns:
            Tuple of (albums_by_type dict, total_markets_queried)
        """
        markets = self.get_all_markets()
        total_markets = len(markets)

        # Reset state for new aggregation
        self.unique_album_ids.clear()
        for album_type in self.albums_by_type:
            self.albums_by_type[album_type].clear()

        for market_idx, market in enumerate(markets, 1):
            if progress_callback:
                progress_callback(
                    f"Querying market {market}...", market_idx, total_markets
                )

            try:
                # Fetch albums for this market
                market_albums = self.spotify_client.fetch_artist_albums_comprehensive(
                    artist_id, market
                )

                if market_albums:
                    self._process_market_albums(market_albums)

            except Exception as e:
                # Log the error but continue with other markets
                if progress_callback:
                    progress_callback(
                        f"Failed to fetch from market {market}: {str(e)}",
                        market_idx,
                        total_markets,
                    )
                continue

        return self.albums_by_type.copy(), total_markets

    def _process_market_albums(self, market_albums: List[Dict]) -> None:
        """
        Process albums from a single market, adding unique ones to the aggregation.
        Memory-efficient processing for large datasets (500+ albums).

        Args:
            market_albums: List of album dictionaries from the API
        """
        for album in market_albums:
            album_id = album.get("id")
            if album_id and album_id not in self.unique_album_ids:
                self.unique_album_ids.add(album_id)

                # Categorize by album type
                album_type = album.get("album_type", "album")
                if album_type in self.albums_by_type:
                    # For memory efficiency with large datasets, only store essential album data
                    # during aggregation phase. Full details will be fetched later in batch.
                    essential_album = {
                        "id": album_id,
                        "name": album.get("name", ""),
                        "album_type": album_type,
                        "release_date": album.get("release_date", ""),
                        "total_tracks": album.get("total_tracks", 0),
                        "images": album.get("images", []),
                        "available_markets": album.get("available_markets", []),
                    }
                    self.albums_by_type[album_type].append(essential_album)

    def get_total_unique_albums(self) -> int:
        """
        Get the total number of unique albums found across all markets.

        Returns:
            Total count of unique albums
        """
        return len(self.unique_album_ids)

    def get_albums_by_type_count(self) -> Dict[str, int]:
        """
        Get the count of albums by type.

        Returns:
            Dictionary with counts for each album type
        """
        return {
            album_type: len(albums)
            for album_type, albums in self.albums_by_type.items()
        }

    def get_flattened_albums(self) -> List[Dict]:
        """
        Get all albums as a flattened list, maintaining type order.

        Returns:
            List of all albums in order: albums, singles, compilations
        """
        flattened = []
        for album_type in ["album", "single", "compilation"]:
            flattened.extend(self.albums_by_type[album_type])
        return flattened

    def clear_memory(self) -> None:
        """
        Clear all stored data to free memory after processing.
        Useful for artists with very large discographies (500+ albums).
        """
        self.unique_album_ids.clear()
        for album_type in self.albums_by_type:
            self.albums_by_type[album_type].clear()


def aggregate_global_albums(
    spotify_client: SpotifyAPIClient,
    artist_id: str,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> Tuple[Dict[str, List[Dict]], int, int]:
    """
    Convenience function to aggregate albums globally for an artist.

    Args:
        spotify_client: Authenticated SpotifyAPIClient instance
        artist_id: Spotify artist ID
        progress_callback: Optional callback for progress updates

    Returns:
        Tuple of (albums_by_type, total_unique_albums, markets_queried)
    """
    aggregator = GlobalMarketAggregator(spotify_client)
    albums_by_type, markets_queried = aggregator.aggregate_artist_albums(
        artist_id, progress_callback
    )
    total_unique = aggregator.get_total_unique_albums()

    return albums_by_type, total_unique, markets_queried


def create_streamlit_progress_callback(
    status_widget,
) -> Callable[[str, int, int], None]:
    """
    Create a progress callback function that updates a Streamlit status widget.

    Args:
        status_widget: Streamlit status widget to update

    Returns:
        Callback function for progress updates
    """

    def callback(message: str, current: int, total: int) -> None:
        status_widget.update(label=f"{message} ({current}/{total})", state="running")

    return callback
