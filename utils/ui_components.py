"""
Common UI components for the Spotify ISRC Finder application.
This module contains reusable UI patterns to reduce code duplication across pages.
"""

import streamlit as st
import pandas as pd
import unicodedata
import re
from PIL import Image
from urllib.request import urlopen
from typing import Optional, Dict, Any, List, Union
from functools import lru_cache
from io import BytesIO
from .tools import to_excel
from .constants import REGIONS, REGION_COLORS, REGION_ICONS, MARKETS
from .data_processing import get_all_markets_for_region, calculate_region_percentage

def normalize_text_for_search(text):
    """
    Normalize text for search comparison by removing diacritics and special characters.

    Examples:
    - "Beyoncé" → "beyonce"
    - "Björk" → "bjork"
    - "Céline Dion" → "celine dion"
    - "Sigur Rós" → "sigur ros"
    - "Mötley Crüe" → "motley crue"
    """
    if not text:
        return ""

    # Convert to lowercase
    text = text.lower()

    # Normalize Unicode (NFD = Normalization Form Decomposed)
    # This separates characters from their diacritical marks
    text = unicodedata.normalize('NFD', text)

    # Remove all combining characters (diacritical marks)
    text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')

    # Handle some specific character replacements that aren't covered by NFD
    replacements = {
        'æ': 'ae',
        'œ': 'oe',
        'ø': 'o',
        'å': 'a',
        'ß': 'ss',
        'ł': 'l',
        'đ': 'd',
        'ħ': 'h',
        'ĸ': 'k',
        'ŋ': 'ng',
        'ð': 'd',
        'þ': 'th',
        '&': 'and',  # Handle ampersands
    }

    for original, replacement in replacements.items():
        text = text.replace(original, replacement)

    # Remove any remaining non-alphanumeric characters except spaces
    text = re.sub(r'[^a-z0-9\s]', '', text)

    # Normalize whitespace
    text = ' '.join(text.split())

    return text


def style_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply column width styling to dataframes for better display.

    Args:
        df: DataFrame to style

    Returns:
        Styled DataFrame with specific column widths
    """
    if df.empty:
        return df

    # Define column widths in pixels
    column_widths = {
        'Available Markets': 75,       # Very narrow width for markets
        'Unavailable Markets': 75,     # Very narrow width for markets
        '℗ Line': 75,                 # Very narrow width for P Line
        'Album Spotify URL': 100,      # Narrow width for URLs
        'Track Spotify URL': 100       # Narrow width for URLs
    }

    # Create style configuration
    style_dict = []
    for col in df.columns:
        if col in column_widths:
            style_dict.append({
                'selector': f'th:contains("{col}")',
                'props': [('max-width', f'{column_widths[col]}px')]
            })
            style_dict.append({
                'selector': f'td:nth-child({list(df.columns).index(col) + 1})',
                'props': [
                    ('max-width', f'{column_widths[col]}px'),
                    ('overflow', 'hidden'),
                    ('text-overflow', 'ellipsis'),
                    ('white-space', 'nowrap')
                ]
            })

    # Apply styles if any column needs styling
    if style_dict:
        return df.style.set_table_styles(style_dict)

    return df


@st.cache_data
def _generate_excel_data(df_json: str, df_shape: tuple) -> Optional[BytesIO]:
    """
    Cache-friendly Excel generation function.

    Args:
        df_json: JSON representation of the DataFrame
        df_shape: Shape of the DataFrame (for cache key)

    Returns:
        Excel file bytes or None
    """
    df = pd.read_json(df_json)
    return to_excel(df)


@st.fragment
def create_download_button(
    df: pd.DataFrame,
    label: str,
    file_name: str,
    key: Optional[str] = None
) -> None:
    """
    Create a standardized Excel download button in an isolated fragment.

    Args:
        df: DataFrame to download
        label: Button label text
        file_name: Name for the downloaded file
        key: Optional unique key for the button
    """
    # Use cached Excel generation to avoid regenerating the same data
    excel_data = _generate_excel_data(df.to_json(), df.shape)

    if excel_data is not None:
        st.download_button(
            label=label,
            data=excel_data,
            file_name=file_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=key
        )


def display_image_safe(
    image_url: Optional[str],
    caption: str,
    width: Optional[int] = None
) -> None:
    """
    Display an image with safe fallback for loading errors.

    Args:
        image_url: URL of the image to display
        caption: Caption for the image
        width: Optional width for the image
    """
    if image_url:
        try:
            image = Image.open(urlopen(image_url))
            st.image(image, caption=caption, width=width)
        except:
            st.write(f"🖼️ {caption}")
    else:
        st.write(f"🖼️ {caption}")


def display_album_row(
    album_data: Dict[str, Any],
    df: pd.DataFrame,
    album_id: str,
    available_markets: Optional[List[str]] = None,
    show_availability_tooltip: bool = False
) -> None:
    """
    Display an album row with image, data, availability badges, and download button.

    Args:
        album_data: Album metadata dictionary
        df: DataFrame with track data
        album_id: Unique album ID for button keys
        available_markets: List of market codes where the album is available
        show_availability_tooltip: Whether to show detailed tooltip on badge click
    """
    import time

    # Create a container for the album row to minimize re-renders
    album_container = st.container()

    with album_container:
        col1, col2 = st.columns([1, 3])

        with col1:
            album_name = album_data.get("name", "Unknown Album")
            album_image_url = album_data["images"][0]["url"] if album_data.get("images") else None
            display_image_safe(album_image_url, album_name)

            # Display availability summary if market data is provided
            if available_markets is not None or album_data.get("available_markets") is not None:
                # Use provided available_markets or fall back to album_data
                markets = available_markets if available_markets is not None else album_data.get("available_markets")

                if markets is not None:
                    # Calculate market counts
                    available_count = len(markets)
                    total_markets = len(MARKETS)
                    missing_count = total_markets - available_count

                    # Display market counts summary with larger numbers and centered layout
                    st.markdown(f"""
                    <div style="display: flex; gap: 30px; margin: 12px 0; justify-content: center;">
                        <div style="text-align: center;">
                            <div style="font-size: 28px; font-weight: bold; color: #4CAF50; line-height: 1;">{available_count}</div>
                            <div style="font-size: 12px; color: #4CAF50; font-weight: 500; margin-top: 2px;">Available Markets</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 28px; font-weight: bold; color: #F44336; line-height: 1;">{missing_count}</div>
                            <div style="font-size: 12px; color: #F44336; font-weight: 500; margin-top: 2px;">Unavailable Markets</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Use popover for market details with overlay effect
                    if show_availability_tooltip:
                        try:
                            with st.popover("🌍 View Market Details", use_container_width=False):
                                # Create cache key for this album's market data
                                cache_key = f"market_tooltip_{album_id}"
                                cache_timestamp_key = f"market_tooltip_timestamp_{album_id}"

                                # Check if we have cached data and if it's still valid (5 minutes)
                                current_time = time.time()
                                cached_timestamp = st.session_state.get(cache_timestamp_key, 0)
                                cache_valid = (current_time - cached_timestamp) < 300  # 5 minutes

                                if cache_key in st.session_state and cache_valid:
                                    # Display cached content using Streamlit components
                                    display_market_details_streamlit(markets, album_id)
                                else:
                                    # Generate and cache the market details
                                    display_market_details_streamlit(markets, album_id)
                                    st.session_state[cache_timestamp_key] = current_time
                        except Exception:
                            # Fallback to expander if popover is not available
                            with st.expander("🌍 View Market Details", expanded=False):
                                display_market_details_streamlit(markets, album_id)

            create_download_button(
                df=df,
                label="📥 Download Excel",
                file_name=f"{album_name}_tracks.xlsx",
                key=f"download_{album_id}"
            )

        with col2:
            styled_df = style_dataframe_columns(df)
            st.dataframe(styled_df, use_container_width=True, hide_index=True)

        st.divider()


def display_processing_info(message: str, icon: str = "🎯") -> None:
    """
    Display standardized processing information.

    Args:
        message: Message to display
        icon: Icon to use (default: 🎯)
    """
    st.info(f"{icon} {message}")


def display_rate_limit_error() -> None:
    """Display standardized rate limit error message."""
    st.error("⏱️ Rate limit exceeded. Returning partial data.")
    st.info(
        "💡 **Tips:**\n"
        "- Wait a few minutes before trying again\n"
        "- Try processing fewer items\n"
        "- The improved API automatically handles rate limiting and retries"
    )


def display_artist_search_widget(
    default_query: str = "",
    default_limit: int = 20,
    default_market: str = "US",
    key_prefix: str = "search"
) -> tuple:
    """
    Display a reusable artist search widget.

    Args:
        default_query: Default search query
        default_limit: Default result limit
        default_market: Default market code
        key_prefix: Prefix for widget keys to avoid conflicts

    Returns:
        Tuple of (search_query, search_limit, market, search_clicked)
    """
    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input(
            "Enter artist name or partial name to search",
            value=default_query,
            key=f"{key_prefix}_query"
        )
    with col2:
        search_limit = st.selectbox(
            "Results limit",
            [10, 20, 30, 50],
            index=[10, 20, 30, 50].index(default_limit),
            key=f"{key_prefix}_limit"
        )

    from .constants import MARKETS
    market = st.selectbox(
        "Select Market (Country Code)",
        MARKETS,
        index=MARKETS.index(default_market),
        key=f"{key_prefix}_market"
    )

    search_clicked = st.button(
        "🔍 Search Artists",
        key=f"{key_prefix}_button"
    )

    return search_query, search_limit, market, search_clicked


def display_artist_search_results(
    artists_df: pd.DataFrame,
    search_query: str,
    download_key: str = "download_artist_search",
    allow_selection: bool = False,
    selection_key: str = "selected_artists"
) -> Optional[pd.DataFrame]:
    """
    Display artist search results with optional selection.

    Args:
        artists_df: DataFrame with artist search results
        search_query: Original search query for display
        download_key: Unique key for download button
        allow_selection: Whether to show selection checkboxes
        selection_key: Key for selection state

    Returns:
        Selected artists DataFrame if allow_selection=True, otherwise None
    """
    if artists_df is None or artists_df.empty:
        return None

    st.subheader(f"🎤 Search Results for '{search_query}'")

    if allow_selection:
        # Add selection column
        artists_df_display = artists_df.copy()
        artists_df_display.insert(0, "Select", False)

        # Use data editor for selection
        edited_df = st.data_editor(
            artists_df_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Select": st.column_config.CheckboxColumn(
                    "Select",
                    help="Select artists to use in catalog search"
                )
            },
            key=selection_key
        )

        # Get selected artists
        selected_artists = edited_df[edited_df["Select"] == True].drop(columns=["Select"])

        if not selected_artists.empty:
            st.success(f"✅ {len(selected_artists)} artist(s) selected")

        # Download button for all results
        create_download_button(
            df=artists_df,
            label="📥 Download All Results as Excel",
            file_name=f"artist_search_{search_query.replace(' ', '_')}.xlsx",
            key=download_key
        )

        return selected_artists
    else:
        # Simple display without selection
        styled_df = style_dataframe_columns(artists_df)
        st.dataframe(styled_df, use_container_width=True, hide_index=True)

        create_download_button(
            df=artists_df,
            label="📥 Download Search Results as Excel",
            file_name=f"artist_search_{search_query.replace(' ', '_')}.xlsx",
            key=download_key
        )

        return None


def display_spotify_attribution() -> None:
    """
    Display Spotify attribution as required by Spotify Developer Terms.
    Shows a small, unobtrusive 'Powered by Spotify' notice in the sidebar.
    """
    with st.sidebar:
        st.markdown("---")
        st.markdown(
            """
            <div style="text-align: center; padding: 10px; font-size: 12px; color: #666;">
                <div style="margin-bottom: 5px;">
                    <span style="color: #1DB954; font-weight: bold; font-size: 14px;">♪ Spotify</span>
                </div>
                <div>Powered by Spotify Web API</div>
            </div>
            """,
            unsafe_allow_html=True
        )


def display_region_header(region: str, percentage: float, color: str, icon: str) -> None:
    """
    Display a styled region header with icon, name, and percentage.

    Args:
        region: The region key (e.g., 'NORTH_AMERICA')
        percentage: The percentage value to display (0-100)
        color: Hex color code for the region (e.g., '#64B5F6')
        icon: Emoji icon for the region (e.g., '🌎')
    """
    # Get the display name for the region
    region_name = REGIONS.get(region, region)

    # Create the HTML for the styled header
    header_html = f"""
    <div style="
        background-color: {color}20;
        border-left: 4px solid {color};
        padding: 12px 16px;
        margin: 10px 0;
        border-radius: 0 8px 8px 0;
        display: flex;
        align-items: center;
        gap: 12px;
    ">
        <span style="font-size: 24px;">{icon}</span>
        <div style="flex-grow: 1;">
            <h4 style="
                margin: 0;
                color: {color};
                font-weight: 600;
                font-size: 18px;
            ">{region_name}</h4>
        </div>
        <div style="
            background-color: {color};
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-weight: 500;
            font-size: 14px;
        ">{percentage}%</div>
    </div>
    """

    st.markdown(header_html, unsafe_allow_html=True)


def display_grouped_markets(grouped_markets: Dict[str, List[str]], market_type: str) -> None:
    """
    Display markets grouped by region with headers and responsive layout.

    Args:
        grouped_markets: Dictionary mapping region keys to lists of market codes
        market_type: Type of markets being displayed ('available' or 'missing')
    """
    # Define the order of regions for consistent display
    region_order = [
        "NORTH_AMERICA", "LATIN_AMERICA", "SOUTH_AMERICA", "CARIBBEAN", "EUROPE",
        "AFRICA", "MIDDLE_EAST", "ASIA", "OCEANIA"
    ]

    # Display each region in the defined order
    for region in region_order:
        if region not in grouped_markets:
            continue

        markets_in_region = grouped_markets[region]
        if not markets_in_region:
            continue

        # Get all markets for this region to calculate percentage
        all_region_markets = get_all_markets_for_region(region)
        percentage = calculate_region_percentage(markets_in_region, all_region_markets)

        # Get region styling
        color = REGION_COLORS.get(region, "#808080")
        icon = REGION_ICONS.get(region, "🌐")

        # Display region header
        display_region_header(region, percentage, color, icon)

        # Display markets in responsive columns
        # Calculate number of columns based on number of markets
        num_markets = len(markets_in_region)
        if num_markets <= 10:
            num_cols = 2
        elif num_markets <= 20:
            num_cols = 3
        else:
            num_cols = 4

        # Create columns
        cols = st.columns(num_cols)

        # Distribute markets across columns
        for idx, market in enumerate(markets_in_region):
            col_idx = idx % num_cols
            with cols[col_idx]:
                # Display market code with subtle styling
                st.markdown(
                    f"""<div style="
                        padding: 4px 8px;
                        margin: 2px 0;
                        background-color: rgba(255, 255, 255, 0.05);
                        border-radius: 4px;
                        font-family: monospace;
                        font-size: 14px;
                    ">{market}</div>""",
                    unsafe_allow_html=True
                )

        # Add spacing between regions
        st.markdown("<br>", unsafe_allow_html=True)


def create_availability_badges(available_markets: Optional[List[str]], album_id: str = None, show_tooltip: bool = False) -> str:
    """
    Create HTML for availability badges showing available and missing market counts.

    Args:
        available_markets: List of market codes where the album is available.
                         Can be None to indicate no market data.
        album_id: Unique identifier for the album (used for tooltip interaction)
        show_tooltip: Whether to include tooltip functionality

    Returns:
        HTML string containing the styled badges
    """
    # Handle no market data case
    if available_markets is None:
        return """
        <div style="display: inline-flex; gap: 8px; align-items: center;">
            <div style="
                padding: 4px 12px;
                background-color: #666666;
                color: white;
                border-radius: 16px;
                font-size: 13px;
                font-weight: 500;
            ">N/A</div>
        </div>
        """

    # Calculate counts
    available_count = len(available_markets)
    total_markets = len(MARKETS)
    missing_count = total_markets - available_count

    # Define colors based on availability
    available_color = "#4CAF50"  # Green
    missing_color = "#F44336"     # Red

    # Create HTML for badges
    cursor_style = 'pointer' if show_tooltip else 'default'
    onclick_attr = f'onclick="toggleTooltip(\'{album_id}\')"' if show_tooltip else ''

    badges_html = f"""
    <div style="display: inline-flex; gap: 8px; align-items: center; margin: 8px 0;">
        <div style="
            padding: 4px 12px;
            background-color: {available_color};
            color: white;
            border-radius: 16px;
            font-size: 13px;
            font-weight: 500;
            cursor: {cursor_style};
        " {onclick_attr}>Available: {available_count}</div>
        <div style="
            padding: 4px 12px;
            background-color: {missing_color};
            color: white;
            border-radius: 16px;
            font-size: 13px;
            font-weight: 500;
            cursor: {cursor_style};
        " {onclick_attr}>Unavailable: {missing_count}</div>
    </div>
    """

    return badges_html


def display_market_details_streamlit(available_markets: List[str], album_id: str) -> None:
    """
    Display market details using native Streamlit components for better rendering.

    Args:
        available_markets: List of market codes where the album is available
        album_id: Unique identifier for the album
    """
    from .data_processing import group_markets_by_region

    # Calculate missing markets
    available_set = set(available_markets) if available_markets else set()
    all_markets_set = set(MARKETS)
    missing_markets = list(all_markets_set - available_set)

    # Group markets by region
    available_grouped = group_markets_by_region(available_markets) if available_markets else {}
    missing_grouped = group_markets_by_region(missing_markets) if missing_markets else {}

    # Define region order for consistent display
    region_order = [
        "NORTH_AMERICA", "LATIN_AMERICA", "SOUTH_AMERICA", "CARIBBEAN", "EUROPE",
        "AFRICA", "MIDDLE_EAST", "ASIA", "OCEANIA"
    ]

    # Start with the market details directly

    # Available Markets Section
    if available_grouped:
        st.markdown("### ✅ Available Markets")

        # Create columns for regions
        available_regions = [r for r in region_order if r in available_grouped and available_grouped[r]]
        if available_regions:
            num_cols = min(3, len(available_regions))
            cols = st.columns(num_cols)

            for idx, region in enumerate(available_regions):
                with cols[idx % num_cols]:
                    markets = available_grouped[region]
                    region_name = REGIONS.get(region, region)
                    region_icon = REGION_ICONS.get(region, "🌐")

                    st.markdown(f"**{region_icon} {region_name}** ({len(markets)})")

                    # Display markets as a formatted list
                    market_text = ", ".join(sorted(markets))
                    st.markdown(f"<small style='font-family: monospace; background-color: rgba(76, 175, 80, 0.1); padding: 4px; border-radius: 4px;'>{market_text}</small>", unsafe_allow_html=True)
                    st.write("")  # Add spacing

    # Unavailable Markets Section
    if missing_grouped:
        st.markdown("### ❌ Unavailable Markets")

        # Create columns for regions
        missing_regions = [r for r in region_order if r in missing_grouped and missing_grouped[r]]
        if missing_regions:
            num_cols = min(3, len(missing_regions))
            cols = st.columns(num_cols)

            for idx, region in enumerate(missing_regions):
                with cols[idx % num_cols]:
                    markets = missing_grouped[region]
                    region_name = REGIONS.get(region, region)
                    region_icon = REGION_ICONS.get(region, "🌐")

                    st.markdown(f"**{region_icon} {region_name}** ({len(markets)})")

                    # Display markets as a formatted list
                    market_text = ", ".join(sorted(markets))
                    st.markdown(f"<small style='font-family: monospace; background-color: rgba(244, 67, 54, 0.1); padding: 4px; border-radius: 4px; color: #ff6666;'>{market_text}</small>", unsafe_allow_html=True)
                    st.write("")  # Add spacing


def create_market_details_display(available_markets: List[str], album_id: str) -> str:
    """
    Create an optimized market details display for Streamlit expander.

    Args:
        available_markets: List of market codes where the album is available
        album_id: Unique identifier for the album

    Returns:
        HTML string containing the market details display
    """
    from .data_processing import group_markets_by_region

    # Calculate missing markets
    available_set = set(available_markets) if available_markets else set()
    all_markets_set = set(MARKETS)
    missing_markets = list(all_markets_set - available_set)

    # Group markets by region
    available_grouped = group_markets_by_region(available_markets) if available_markets else {}
    missing_grouped = group_markets_by_region(missing_markets) if missing_markets else {}

    # Define region order for consistent display
    region_order = [
        "NORTH_AMERICA", "LATIN_AMERICA", "SOUTH_AMERICA", "CARIBBEAN", "EUROPE",
        "AFRICA", "MIDDLE_EAST", "ASIA", "OCEANIA"
    ]

    # Build display HTML
    display_html = "<div style='padding: 10px;'>"

    # Summary stats
    display_html += f"""
    <div style="display: flex; gap: 20px; margin-bottom: 20px;">
        <div style="text-align: center;">
            <div style="font-size: 24px; font-weight: bold; color: #4CAF50;">{len(available_markets)}</div>
            <div style="font-size: 12px; color: #888;">Available Markets</div>
        </div>
        <div style="text-align: center;">
            <div style="font-size: 24px; font-weight: bold; color: #F44336;">{len(missing_markets)}</div>
            <div style="font-size: 12px; color: #888;">Unavailable Markets</div>
        </div>
    </div>
    """

    # Available Markets Section
    if available_grouped:
        display_html += """
        <div style="margin-bottom: 20px;">
            <h4 style="color: #4CAF50; margin: 0 0 12px 0; font-size: 16px;">✅ Available Markets</h4>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px;">
        """

        for region in region_order:
            if region in available_grouped and available_grouped[region]:
                markets = available_grouped[region]
                region_name = REGIONS.get(region, region)
                region_color = REGION_COLORS.get(region, "#808080")
                region_icon = REGION_ICONS.get(region, "🌐")

                display_html += f"""
                <div style="background-color: rgba(255, 255, 255, 0.05); padding: 10px; border-radius: 6px; border-left: 3px solid {region_color};">
                    <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 8px;">
                        <span>{region_icon}</span>
                        <span style="color: {region_color}; font-weight: 600; font-size: 13px;">{region_name}</span>
                        <span style="margin-left: auto; font-size: 12px; color: #888;">({len(markets)})</span>
                    </div>
                    <div style="display: flex; flex-wrap: wrap; gap: 4px;">
                """

                for market in sorted(markets):
                    display_html += f"""
                        <span style="
                            padding: 2px 6px;
                            background-color: rgba(76, 175, 80, 0.15);
                            border: 1px solid rgba(76, 175, 80, 0.3);
                            border-radius: 3px;
                            font-size: 11px;
                            font-family: monospace;
                        ">{market}</span>
                    """

                display_html += """
                    </div>
                </div>
                """

        display_html += """
            </div>
        </div>
        """

    # Unavailable Markets Section
    if missing_grouped:
        display_html += """
        <div>
            <h4 style="color: #F44336; margin: 0 0 12px 0; font-size: 16px;">❌ Unavailable Markets</h4>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px;">
        """

        for region in region_order:
            if region in missing_grouped and missing_grouped[region]:
                markets = missing_grouped[region]
                region_name = REGIONS.get(region, region)
                region_color = REGION_COLORS.get(region, "#808080")
                region_icon = REGION_ICONS.get(region, "🌐")

                display_html += f"""
                <div style="background-color: rgba(255, 255, 255, 0.05); padding: 10px; border-radius: 6px; border-left: 3px solid {region_color};">
                    <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 8px;">
                        <span>{region_icon}</span>
                        <span style="color: {region_color}; font-weight: 600; font-size: 13px;">{region_name}</span>
                        <span style="margin-left: auto; font-size: 12px; color: #888;">({len(markets)})</span>
                    </div>
                    <div style="display: flex; flex-wrap: wrap; gap: 4px;">
                """

                for market in sorted(markets):
                    display_html += f"""
                        <span style="
                            padding: 2px 6px;
                            background-color: rgba(244, 67, 54, 0.15);
                            border: 1px solid rgba(244, 67, 54, 0.3);
                            border-radius: 3px;
                            font-size: 11px;
                            font-family: monospace;
                            color: #ff9999;
                        ">{market}</span>
                    """

                display_html += """
                    </div>
                </div>
                """

        display_html += """
            </div>
        </div>
        """

    display_html += "</div>"

    return display_html


def create_availability_tooltip(available_markets: List[str], album_id: str) -> str:
    """
    Create an interactive tooltip showing regional market groupings.

    Args:
        available_markets: List of market codes where the album is available
        album_id: Unique identifier for the album

    Returns:
        HTML string containing the tooltip with regional market display
    """
    from .data_processing import group_markets_by_region

    # Calculate missing markets
    available_set = set(available_markets) if available_markets else set()
    all_markets_set = set(MARKETS)
    missing_markets = list(all_markets_set - available_set)

    # Group markets by region
    available_grouped = group_markets_by_region(available_markets) if available_markets else {}
    missing_grouped = group_markets_by_region(missing_markets) if missing_markets else {}

    # Define region order for consistent display
    region_order = [
        "NORTH_AMERICA", "LATIN_AMERICA", "SOUTH_AMERICA", "CARIBBEAN", "EUROPE",
        "AFRICA", "MIDDLE_EAST", "ASIA", "OCEANIA"
    ]

    # Build tooltip HTML
    tooltip_html = f"""
    <div id="tooltip-{album_id}" style="
        display: none;
        position: relative;
        margin-top: 10px;
        padding: 16px;
        background-color: #1e1e1e;
        border: 1px solid #333;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        max-height: 400px;
        overflow-y: auto;
    ">
    """

    # Available Markets Section
    if available_grouped:
        tooltip_html += """
        <div style="margin-bottom: 20px;">
            <h4 style="color: #4CAF50; margin: 0 0 12px 0; font-size: 16px;">Available Markets</h4>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 16px;">
        """

        for region in region_order:
            if region in available_grouped and available_grouped[region]:
                markets = available_grouped[region]
                region_name = REGIONS.get(region, region)
                region_color = REGION_COLORS.get(region, "#808080")
                region_icon = REGION_ICONS.get(region, "🌐")

                tooltip_html += f"""
                <div style="background-color: rgba(255, 255, 255, 0.05); padding: 12px; border-radius: 6px;">
                    <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 8px;">
                        <span>{region_icon}</span>
                        <span style="color: {region_color}; font-weight: 600; font-size: 14px;">{region_name}</span>
                    </div>
                    <div style="display: flex; flex-wrap: wrap; gap: 4px;">
                """

                for market in sorted(markets):
                    tooltip_html += f"""
                        <span style="
                            padding: 2px 6px;
                            background-color: rgba(76, 175, 80, 0.2);
                            border: 1px solid rgba(76, 175, 80, 0.3);
                            border-radius: 4px;
                            font-size: 12px;
                            font-family: monospace;
                        ">{market}</span>
                    """

                tooltip_html += """
                    </div>
                </div>
                """

        tooltip_html += """
            </div>
        </div>
        """

    # Unavailable Markets Section
    if missing_grouped:
        tooltip_html += """
        <div>
            <h4 style="color: #F44336; margin: 0 0 12px 0; font-size: 16px;">Unavailable Markets</h4>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 16px;">
        """

        for region in region_order:
            if region in missing_grouped and missing_grouped[region]:
                markets = missing_grouped[region]
                region_name = REGIONS.get(region, region)
                region_color = REGION_COLORS.get(region, "#808080")
                region_icon = REGION_ICONS.get(region, "🌐")

                tooltip_html += f"""
                <div style="background-color: rgba(255, 255, 255, 0.05); padding: 12px; border-radius: 6px;">
                    <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 8px;">
                        <span>{region_icon}</span>
                        <span style="color: {region_color}; font-weight: 600; font-size: 14px;">{region_name}</span>
                    </div>
                    <div style="display: flex; flex-wrap: wrap; gap: 4px;">
                """

                for market in sorted(markets):
                    tooltip_html += f"""
                        <span style="
                            padding: 2px 6px;
                            background-color: rgba(244, 67, 54, 0.2);
                            border: 1px solid rgba(244, 67, 54, 0.3);
                            border-radius: 4px;
                            font-size: 12px;
                            font-family: monospace;
                            color: #ff9999;
                        ">{market}</span>
                    """

                tooltip_html += """
                    </div>
                </div>
                """

        tooltip_html += """
            </div>
        </div>
        """

    tooltip_html += """
    </div>
    """

    # Add JavaScript for tooltip toggle
    tooltip_html += f"""
    <script>
    function toggleTooltip(albumId) {{
        var tooltip = document.getElementById('tooltip-' + albumId);
        if (tooltip) {{
            if (tooltip.style.display === 'none' || tooltip.style.display === '') {{
                tooltip.style.display = 'block';
            }} else {{
                tooltip.style.display = 'none';
            }}
        }}
    }}
    </script>
    """

    return tooltip_html


# Removed create_market_toggle_component - unused function that was causing session state errors


def create_multiple_markets_only_component(key: str = "multiple_markets") -> list:
    """
    Create a market selection component for Multiple Artist Catalog - only supports multiple markets (up to 10).
    Uses simple key-only approach to avoid double-click issues.

    Args:
        key: Unique key for the Streamlit widget

    Returns:
        list: Selected market codes (1-10 markets)
    """
    from .constants import MARKETS

    # Initialize widget session state once - ONLY touch this on first load
    widget_key = f"{key}_multiselect"
    if widget_key not in st.session_state:
        st.session_state[widget_key] = ["US"]  # Default to US only

    # Create multiselect - let it manage its own state completely
    selected_markets = st.multiselect(
        "Choose Markets (1-10 required):",
        options=MARKETS,
        key=widget_key,
        help="Select 1-10 markets to query. More markets = longer processing time."
    )

    # Validate selection count
    if len(selected_markets) > 10:
        st.error("⚠️ **Too many markets selected!** Please select no more than 10 markets.")
        st.info(f"📝 Showing first 10 markets: {', '.join(selected_markets[:10])}")
        # Return trimmed version but don't modify widget state
        selected_markets = selected_markets[:10]

    if len(selected_markets) == 0:
        st.warning("⚠️ Please select at least one market.")
        # Return fallback but don't modify widget state
        selected_markets = ["US"]

    # Buttons removed temporarily to eliminate session state conflicts
    # TODO: Implement alternative approach for Key Markets and Clear functionality

    # Show processing estimate
    if selected_markets:
        estimated_time = f"{len(selected_markets) * 10}-{len(selected_markets) * 15} seconds per artist"
        st.caption(f"⏱️ **Estimated processing time**: {estimated_time} for {len(selected_markets)} market(s)")

    return selected_markets


def create_single_artist_markets_component(key: str = "single_artist_markets") -> tuple[str, list]:
    """
    Create a market selection component for Single Artist Catalog - supports Multiple and All Markets.

    Args:
        key: Unique key for the Streamlit widget

    Returns:
        tuple: (market_mode, selected_markets_list)
            - market_mode: "multiple" or "all"
            - selected_markets_list: List of selected market codes
    """
    from .constants import MARKETS, MARKET_TIERS

    # Initialize view mode session state
    if f"{key}_view" not in st.session_state:
        st.session_state[f"{key}_view"] = "multiple"  # Default to Multiple Markets

    # Radio button implementation (stacked vertically)
    view_option = st.radio(
        "Market Selection Mode:",
        options=["multiple", "all"],
        format_func=lambda x: {
            "multiple": "🌐 Multiple Markets",
            "all": "🌍 All Markets"
        }[x],
        horizontal=False,
        key=f"{key}_radio",
        help="Multiple: Select specific markets | All: Query all 185 markets (slowest)"
    )

    # Update session state
    st.session_state[f"{key}_view"] = view_option

    selected_markets = []

    if view_option == "multiple":
        # Multiple market selector (no limit for single artist)
        # Uses simple key-only approach to avoid double-click issues

        # Initialize widget session state once - ONLY touch this on first load
        widget_key = f"{key}_multiselect"
        if widget_key not in st.session_state:
            st.session_state[widget_key] = ["US"]  # Default to US

        # Create multiselect - let it manage its own state completely
        selected_markets = st.multiselect(
            "Choose Markets (1-10 required):",
            options=MARKETS,
            key=widget_key,
            help="Select markets to query. More markets = longer processing time."
        )

        # Validate and handle empty selection
        if len(selected_markets) == 0:
            st.warning("⚠️ Please select at least one market.")
            # Return fallback but don't modify widget state
            selected_markets = ["US"]

        # Buttons removed temporarily to eliminate session state conflicts
        # TODO: Implement alternative approach for Key Markets and Clear functionality

        # Show processing estimate
        if selected_markets:
            estimated_time = f"{len(selected_markets) * 5}-{len(selected_markets) * 10} seconds"
            st.caption(f"⏱️ **Estimated processing time**: {estimated_time} for {len(selected_markets)} market(s)")

    else:  # "all"
        # All markets mode with warning
        st.warning(
            "⚠️ **All Markets Mode**: This will query all 185 available markets. "
            "Processing will take significantly longer (5 minutes or more)."
        )
        st.info(
            "💡 **Tip**: Consider using 'Multiple Markets' mode with 5-10 key markets "
            "for faster results while still getting good global coverage."
        )
        selected_markets = MARKETS

        # Show market count
        st.caption(f"📊 **Will query**: {len(MARKETS)} markets")

    return view_option, selected_markets
