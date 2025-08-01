"""
Constants and configuration values for the Spotify ISRC Finder application.
"""

# API Configuration
SPOTIFY_BASE_URL = "https://api.spotify.com/v1"
SPOTIFY_AUTH_URL = "https://accounts.spotify.com/api/token"

# Rate Limiting and Timeouts
DEFAULT_TIMEOUT = 30  # seconds
MAX_RETRIES = 5
INITIAL_BACKOFF_DELAY = 1.0  # seconds
MAX_BACKOFF_DELAY = 60.0  # seconds
BACKOFF_MULTIPLIER = 2.0
JITTER_RANGE = 1.0  # seconds

# Batch Sizes (Spotify API limits)
TRACKS_BATCH_SIZE = 50  # Maximum tracks per request
ALBUMS_BATCH_SIZE = 20  # Maximum albums per request
PLAYLIST_ITEMS_LIMIT = 100  # Maximum playlist items per page
ARTIST_ALBUMS_LIMIT = 50  # Maximum artist albums per page

# Request Delays
INTER_BATCH_DELAY = 0.1  # seconds between batches
INTER_PAGE_DELAY = 0.1  # seconds between pages
RATE_LIMIT_DELAY = 0.5  # seconds between requests to avoid rate limits

# Thread Pool Configuration
MAX_WORKERS = 2  # Maximum concurrent threads for API requests

# Input Validation
MAX_INPUT_LENGTH = 10000  # Maximum characters in text input
MAX_ITEMS_PER_REQUEST = 1000  # Maximum items to process in one request
SPOTIFY_ID_LENGTH = 22  # Standard Spotify ID length

# Album Types for separate queries
ALBUM_TYPES = ["album", "single", "compilation"]

# Album Sorting Configuration
SORTING_STRATEGIES = [
    "release_date", "album_name", "album_id", "popularity", 
    "track_count", "artist_name"
]
DEFAULT_SORT_STRATEGY = "release_date"
DEFAULT_SECONDARY_SORT = "album_id"
SORT_ORDERS = ["asc", "desc"]
DEFAULT_SORT_ORDER = "desc"

# Sorting Strategy Metadata
SORTING_STRATEGY_INFO = {
    "release_date": {
        "name": "Release Date",
        "description": "Sort by album release date (newest first by default)",
        "data_type": "datetime",
        "supports_reverse": True,
        "default_order": "desc"
    },
    "album_name": {
        "name": "Album Name",
        "description": "Sort alphabetically by album name",
        "data_type": "string",
        "supports_reverse": True,
        "default_order": "asc"
    },
    "album_id": {
        "name": "Album ID",
        "description": "Sort by Spotify album ID (for stable sorting)",
        "data_type": "string",
        "supports_reverse": True,
        "default_order": "asc"
    },
    "popularity": {
        "name": "Popularity",
        "description": "Sort by popularity score (0-100, highest first by default)",
        "data_type": "integer",
        "supports_reverse": True,
        "default_order": "desc"
    },
    "track_count": {
        "name": "Track Count",
        "description": "Sort by number of tracks in album",
        "data_type": "integer",
        "supports_reverse": True,
        "default_order": "desc"
    },
    "artist_name": {
        "name": "Artist Name",
        "description": "Sort alphabetically by primary artist name",
        "data_type": "string",
        "supports_reverse": True,
        "default_order": "asc"
    }
}

# Sort Order Configuration
SORT_ORDER_INFO = {
    "asc": {
        "name": "Ascending",
        "description": "Sort from lowest to highest (A-Z, oldest to newest, 0-100)",
        "symbol": "↑"
    },
    "desc": {
        "name": "Descending", 
        "description": "Sort from highest to lowest (Z-A, newest to oldest, 100-0)",
        "symbol": "↓"
    }
}

# Multi-Strategy Sorting Presets
SORTING_PRESETS = {
    "newest_first": [
        ("release_date", "desc"),
        ("album_name", "asc"),
        ("album_id", "asc")
    ],
    "oldest_first": [
        ("release_date", "asc"),
        ("album_name", "asc"),
        ("album_id", "asc")
    ],
    "alphabetical": [
        ("album_name", "asc"),
        ("release_date", "desc"),
        ("album_id", "asc")
    ],
    "most_popular": [
        ("popularity", "desc"),
        ("release_date", "desc"),
        ("album_id", "asc")
    ],
    "artist_alphabetical": [
        ("artist_name", "asc"),
        ("release_date", "desc"),
        ("album_name", "asc")
    ],
    "track_count": [
        ("track_count", "desc"),
        ("release_date", "desc"),
        ("album_id", "asc")
    ]
}

# Sorting Configuration Limits
SORTING_CONFIG = {
    "max_strategies_per_sort": 5,
    "max_albums_for_performance_warning": 1000,
    "performance_timeout_ms": 500,
    "unknown_data_handling": "separate",  # "separate" or "integrate"
    "stable_sort_fallback": "album_id"
}

# Date Format Constants for Album Sorting
SPOTIFY_DATE_FORMATS = {
    "full": "%Y-%m-%d",      # YYYY-MM-DD
    "month": "%Y-%m",        # YYYY-MM  
    "year": "%Y"             # YYYY
}

# Date Handling Configuration
DATE_HANDLING_CONFIG = {
    "unknown_date_fallback": "9999-12-31",  # Far future date for unknown dates
    "min_valid_year": 1900,
    "max_valid_year": 2050,
    "normalize_partial_dates": True,  # Convert YYYY and YYYY-MM to full dates
    "partial_date_defaults": {
        "month": 1,  # Default month for year-only dates
        "day": 1     # Default day for year/month-only dates
    }
}

# Sorting Performance Thresholds
SORTING_PERFORMANCE = {
    "small_dataset": 100,      # Albums
    "medium_dataset": 500,     # Albums  
    "large_dataset": 1000,     # Albums
    "max_dataset": 5000,       # Albums
    "performance_warning_threshold": 1000,
    "performance_timeout_ms": 500
}

# Sorting Error Handling
SORTING_ERROR_CONFIG = {
    "continue_on_error": True,
    "fallback_to_original_order": True,
    "log_individual_errors": True,
    "max_logged_errors": 10,
    "error_recovery_strategies": [
        "use_fallback_value",
        "skip_invalid_item", 
        "use_original_order"
    ]
}

# Supported Markets
DEFAULT_MARKET = "US"
MARKETS = [
    "AD","AE","AG","AL","AM","AO","AR","AT","AU","AZ","BA","BB","BD","BE","BF","BG","BH","BI","BJ","BN",
    "BO","BR","BS","BT","BW","BY","BZ","CA","CD","CG","CH","CI","CL","CM","CO","CR","CV","CW","CY","CZ",
    "DE","DJ","DK","DM","DO","DZ","EC","EE","EG","ES","ET","FI","FJ","FM","FR","GA","GB","GD","GE","GH",
    "GM","GN","GQ","GR","GT","GW","GY","HK","HN","HR","HT","HU","ID","IE","IL","IN","IQ","IS","IT","JM",
    "JO","JP","KE","KG","KH","KI","KM","KN","KR","KW","KZ","LA","LB","LC","LI","LK","LR","LS","LT","LU",
    "LV","LY","MA","MC","MD","ME","MG","MH","MK","ML","MN","MO","MR","MT","MU","MV","MW","MX","MY","MZ",
    "NA","NE","NG","NI","NL","NO","NP","NR","NZ","OM","PA","PE","PG","PH","PK","PL","PR","PS","PT","PW",
    "PY","QA","RO","RS","RW","SA","SB","SC","SE","SG","SI","SK","SL","SM","SN","SR","ST","SV","SZ","TD",
    "TG","TH","TJ","TL","TN","TO","TR","TT","TV","TW","TZ","UA","UG","US","UY","UZ","VC","VE","VN","VU",
    "WS","XK","ZA","ZM","ZW"
]

# Market Prioritization for Optimized All Markets Search
# Based on music industry importance, streaming volumes, and API response patterns
MARKET_TIERS = {
    # Tier 1: Major music markets - highest priority, fastest processing
    "tier1": ["US", "GB", "CA", "DE", "FR", "JP"],
    
    # Tier 2: Important regional markets - medium priority
    "tier2": ["AU", "NL", "SE", "NO", "DK", "FI", "CH", "AT", "BE", "IT", 
              "ES", "BR", "MX", "AR", "KR", "SG", "HK", "NZ", "IE", "PL"],
    
    # Tier 3: All remaining markets - lowest priority, loaded on demand
    "tier3": [market for market in MARKETS if market not in 
              ["US", "GB", "CA", "DE", "FR", "JP", "AU", "NL", "SE", "NO", 
               "DK", "FI", "CH", "AT", "BE", "IT", "ES", "BR", "MX", "AR", 
               "KR", "SG", "HK", "NZ", "IE", "PL"]]
}

# Market processing modes for user selection
MARKET_PROCESSING_MODES = {
    "fast": {
        "name": "⚡ Fast Mode",
        "description": "Major markets only (6 markets)",
        "markets": MARKET_TIERS["tier1"],
        "estimated_time": "5-10 seconds"
    },
    "balanced": {
        "name": "⚖️ Balanced Mode", 
        "description": "Major + regional markets (26 markets)",
        "markets": MARKET_TIERS["tier1"] + MARKET_TIERS["tier2"],
        "estimated_time": "15-30 seconds"
    },
    "complete": {
        "name": "🌍 Complete Mode",
        "description": "All available markets (185 markets)",
        "markets": MARKETS,
        "estimated_time": "45-90 seconds"
    }
}

# Error Messages
ERROR_MESSAGES = {
    "invalid_id": "Invalid Spotify ID format",
    "rate_limit": "Rate limit exceeded. Please try again later.",
    "auth_failed": "Authentication failed. Please check your credentials.",
    "network_error": "Network error. Please check your connection.",
    "invalid_input": "Invalid input detected",
    "input_too_long": f"Input exceeds maximum length of {MAX_INPUT_LENGTH} characters",
    "too_many_items": f"Too many items. Maximum allowed: {MAX_ITEMS_PER_REQUEST}",
}

# Success Messages
SUCCESS_MESSAGES = {
    "batch_complete": "Batch processing completed successfully",
    "data_fetched": "Data fetched successfully",
    "validation_passed": "Input validation passed",
}

# File Export
EXCEL_MIME_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

# Region Constants
REGIONS = {
    "NORTH_AMERICA": "North America",
    "LATIN_AMERICA": "Latin America",
    "SOUTH_AMERICA": "South America",
    "CARIBBEAN": "Caribbean",
    "EUROPE": "Europe",
    "AFRICA": "Africa",
    "MIDDLE_EAST": "Middle East",
    "ASIA": "Asia",
    "OCEANIA": "Oceania"
}

# Country to Region Mapping
COUNTRY_TO_REGION = {
    # North America
    "CA": "NORTH_AMERICA",  # Canada
    "US": "NORTH_AMERICA",  # United States
    "MX": "NORTH_AMERICA",  # Mexico
    "PR": "NORTH_AMERICA",  # Puerto Rico
    
    # Latin America
    "GT": "LATIN_AMERICA",  # Guatemala
    "BZ": "LATIN_AMERICA",  # Belize
    "SV": "LATIN_AMERICA",  # El Salvador
    "HN": "LATIN_AMERICA",  # Honduras
    "NI": "LATIN_AMERICA",  # Nicaragua
    "CR": "LATIN_AMERICA",  # Costa Rica
    "PA": "LATIN_AMERICA",  # Panama
    
    # South America
    "AR": "SOUTH_AMERICA",  # Argentina
    "BO": "SOUTH_AMERICA",  # Bolivia
    "BR": "SOUTH_AMERICA",  # Brazil
    "CL": "SOUTH_AMERICA",  # Chile
    "CO": "SOUTH_AMERICA",  # Colombia
    "EC": "SOUTH_AMERICA",  # Ecuador
    "GY": "SOUTH_AMERICA",  # Guyana
    "PY": "SOUTH_AMERICA",  # Paraguay
    "PE": "SOUTH_AMERICA",  # Peru
    "SR": "SOUTH_AMERICA",  # Suriname
    "UY": "SOUTH_AMERICA",  # Uruguay
    "VE": "SOUTH_AMERICA",  # Venezuela
    
    # Caribbean
    "AG": "CARIBBEAN",  # Antigua and Barbuda
    "BS": "CARIBBEAN",  # Bahamas
    "BB": "CARIBBEAN",  # Barbados
    "CW": "CARIBBEAN",  # Curaçao
    "DM": "CARIBBEAN",  # Dominica
    "DO": "CARIBBEAN",  # Dominican Republic
    "GD": "CARIBBEAN",  # Grenada
    "HT": "CARIBBEAN",  # Haiti
    "JM": "CARIBBEAN",  # Jamaica
    "KN": "CARIBBEAN",  # Saint Kitts and Nevis
    "LC": "CARIBBEAN",  # Saint Lucia
    "VC": "CARIBBEAN",  # Saint Vincent and the Grenadines
    "TT": "CARIBBEAN",  # Trinidad and Tobago
    
    # Europe
    "AD": "EUROPE",  # Andorra
    "AL": "EUROPE",  # Albania
    "AT": "EUROPE",  # Austria
    "BA": "EUROPE",  # Bosnia and Herzegovina
    "BE": "EUROPE",  # Belgium
    "BG": "EUROPE",  # Bulgaria
    "BY": "EUROPE",  # Belarus
    "CH": "EUROPE",  # Switzerland
    "CY": "EUROPE",  # Cyprus
    "CZ": "EUROPE",  # Czech Republic
    "DE": "EUROPE",  # Germany
    "DK": "EUROPE",  # Denmark
    "EE": "EUROPE",  # Estonia
    "ES": "EUROPE",  # Spain
    "FI": "EUROPE",  # Finland
    "FR": "EUROPE",  # France
    "GB": "EUROPE",  # United Kingdom
    "GR": "EUROPE",  # Greece
    "HR": "EUROPE",  # Croatia
    "HU": "EUROPE",  # Hungary
    "IE": "EUROPE",  # Ireland
    "IS": "EUROPE",  # Iceland
    "IT": "EUROPE",  # Italy
    "LI": "EUROPE",  # Liechtenstein
    "LT": "EUROPE",  # Lithuania
    "LU": "EUROPE",  # Luxembourg
    "LV": "EUROPE",  # Latvia
    "MC": "EUROPE",  # Monaco
    "MD": "EUROPE",  # Moldova
    "ME": "EUROPE",  # Montenegro
    "MK": "EUROPE",  # North Macedonia
    "MT": "EUROPE",  # Malta
    "NL": "EUROPE",  # Netherlands
    "NO": "EUROPE",  # Norway
    "PL": "EUROPE",  # Poland
    "PT": "EUROPE",  # Portugal
    "RO": "EUROPE",  # Romania
    "RS": "EUROPE",  # Serbia
    "SE": "EUROPE",  # Sweden
    "SI": "EUROPE",  # Slovenia
    "SK": "EUROPE",  # Slovakia
    "SM": "EUROPE",  # San Marino
    "UA": "EUROPE",  # Ukraine
    "XK": "EUROPE",  # Kosovo
    
    # Africa
    "AO": "AFRICA",  # Angola
    "BF": "AFRICA",  # Burkina Faso
    "BI": "AFRICA",  # Burundi
    "BJ": "AFRICA",  # Benin
    "BW": "AFRICA",  # Botswana
    "CD": "AFRICA",  # Democratic Republic of the Congo
    "CG": "AFRICA",  # Republic of the Congo
    "CI": "AFRICA",  # Côte d'Ivoire
    "CM": "AFRICA",  # Cameroon
    "CV": "AFRICA",  # Cape Verde
    "DJ": "AFRICA",  # Djibouti
    "DZ": "AFRICA",  # Algeria
    "EG": "AFRICA",  # Egypt
    "ET": "AFRICA",  # Ethiopia
    "GA": "AFRICA",  # Gabon
    "GH": "AFRICA",  # Ghana
    "GM": "AFRICA",  # Gambia
    "GN": "AFRICA",  # Guinea
    "GQ": "AFRICA",  # Equatorial Guinea
    "GW": "AFRICA",  # Guinea-Bissau
    "KE": "AFRICA",  # Kenya
    "KM": "AFRICA",  # Comoros
    "LR": "AFRICA",  # Liberia
    "LS": "AFRICA",  # Lesotho
    "LY": "AFRICA",  # Libya
    "MA": "AFRICA",  # Morocco
    "MG": "AFRICA",  # Madagascar
    "ML": "AFRICA",  # Mali
    "MR": "AFRICA",  # Mauritania
    "MU": "AFRICA",  # Mauritius
    "MW": "AFRICA",  # Malawi
    "MZ": "AFRICA",  # Mozambique
    "NA": "AFRICA",  # Namibia
    "NE": "AFRICA",  # Niger
    "NG": "AFRICA",  # Nigeria
    "RW": "AFRICA",  # Rwanda
    "SC": "AFRICA",  # Seychelles
    "SL": "AFRICA",  # Sierra Leone
    "SN": "AFRICA",  # Senegal
    "ST": "AFRICA",  # São Tomé and Príncipe
    "SZ": "AFRICA",  # Eswatini
    "TD": "AFRICA",  # Chad
    "TG": "AFRICA",  # Togo
    "TN": "AFRICA",  # Tunisia
    "TZ": "AFRICA",  # Tanzania
    "UG": "AFRICA",  # Uganda
    "ZA": "AFRICA",  # South Africa
    "ZM": "AFRICA",  # Zambia
    "ZW": "AFRICA",  # Zimbabwe
    
    # Middle East
    "AE": "MIDDLE_EAST",  # United Arab Emirates
    "AM": "MIDDLE_EAST",  # Armenia
    "AZ": "MIDDLE_EAST",  # Azerbaijan
    "BH": "MIDDLE_EAST",  # Bahrain
    "GE": "MIDDLE_EAST",  # Georgia
    "IL": "MIDDLE_EAST",  # Israel
    "IQ": "MIDDLE_EAST",  # Iraq
    "JO": "MIDDLE_EAST",  # Jordan
    "KW": "MIDDLE_EAST",  # Kuwait
    "LB": "MIDDLE_EAST",  # Lebanon
    "OM": "MIDDLE_EAST",  # Oman
    "PS": "MIDDLE_EAST",  # Palestine
    "QA": "MIDDLE_EAST",  # Qatar
    "SA": "MIDDLE_EAST",  # Saudi Arabia
    "TR": "MIDDLE_EAST",  # Turkey
    
    # Asia
    "BD": "ASIA",  # Bangladesh
    "BN": "ASIA",  # Brunei
    "BT": "ASIA",  # Bhutan
    "HK": "ASIA",  # Hong Kong
    "ID": "ASIA",  # Indonesia
    "IN": "ASIA",  # India
    "JP": "ASIA",  # Japan
    "KG": "ASIA",  # Kyrgyzstan
    "KH": "ASIA",  # Cambodia
    "KR": "ASIA",  # South Korea
    "KZ": "ASIA",  # Kazakhstan
    "LA": "ASIA",  # Laos
    "LK": "ASIA",  # Sri Lanka
    "MN": "ASIA",  # Mongolia
    "MO": "ASIA",  # Macao
    "MV": "ASIA",  # Maldives
    "MY": "ASIA",  # Malaysia
    "NP": "ASIA",  # Nepal
    "PH": "ASIA",  # Philippines
    "PK": "ASIA",  # Pakistan
    "SG": "ASIA",  # Singapore
    "TH": "ASIA",  # Thailand
    "TJ": "ASIA",  # Tajikistan
    "TL": "ASIA",  # Timor-Leste
    "TW": "ASIA",  # Taiwan
    "UZ": "ASIA",  # Uzbekistan
    "VN": "ASIA",  # Vietnam
    
    # Oceania
    "AU": "OCEANIA",  # Australia
    "FJ": "OCEANIA",  # Fiji
    "FM": "OCEANIA",  # Micronesia
    "KI": "OCEANIA",  # Kiribati
    "MH": "OCEANIA",  # Marshall Islands
    "NR": "OCEANIA",  # Nauru
    "NZ": "OCEANIA",  # New Zealand
    "PG": "OCEANIA",  # Papua New Guinea
    "PW": "OCEANIA",  # Palau
    "SB": "OCEANIA",  # Solomon Islands
    "TO": "OCEANIA",  # Tonga
    "TV": "OCEANIA",  # Tuvalu
    "VU": "OCEANIA",  # Vanuatu
    "WS": "OCEANIA",  # Samoa
}

# Region Colors (Dark Theme Optimized)
REGION_COLORS = {
    "NORTH_AMERICA": "#64B5F6",  # Light Blue
    "LATIN_AMERICA": "#FFAB91",  # Light Orange/Coral
    "SOUTH_AMERICA": "#81C784",  # Light Green
    "CARIBBEAN": "#4DD0E1",      # Turquoise
    "EUROPE": "#BA68C8",         # Light Purple
    "AFRICA": "#FFB74D",         # Light Orange
    "MIDDLE_EAST": "#A1887F",    # Light Brown/Tan
    "ASIA": "#F06292",           # Light Red/Pink
    "OCEANIA": "#4DB6AC"         # Light Teal
}

# Region Icons
REGION_ICONS = {
    "NORTH_AMERICA": "🌎",
    "LATIN_AMERICA": "🏛️",
    "SOUTH_AMERICA": "🌎",
    "CARIBBEAN": "🏝️",
    "EUROPE": "🌍",
    "AFRICA": "🌍",
    "MIDDLE_EAST": "🏜️",
    "ASIA": "🌏",
    "OCEANIA": "🌏"
}