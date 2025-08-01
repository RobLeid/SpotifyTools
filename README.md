## Overview

This is a Streamlit-based Spotify ISRC Finder application that provides tools for searching and analyzing Spotify catalog data, including tracks, albums, and artist information.

## Key Commands

### Running the Application
```bash
# Run the main Streamlit app
streamlit run Hello.py

# Run specific pages directly
streamlit run pages/1_Tracks.py
streamlit run pages/2_Albums.py
streamlit run pages/5_Single\ Artist\ Catalog.py
streamlit run pages/6_Multiple\ Artist\ Catalog.py
```

### Development Setup
```bash
# Install dependencies
pip install -r requirements.txt
```

## Architecture

### Application Structure
- **Hello.py**: Main entry point, welcome page with Spotify attribution
- **pages/**: Individual Streamlit pages for different features
  - Track search, Album search, User Playlists, Artist Top Tracks
  - Single/Multiple Artist Catalog analysis
- **utils/**: Shared utility modules
  - **auth.py**: Spotify API authentication using client credentials
  - **api_improved.py**: Enhanced API client with rate limiting
  - **album_sorting.py**: Album sorting and grouping logic
  - **global_market_aggregator.py**: Market availability aggregation
  - **ui_components.py**: Reusable UI components
  - **validation.py**: Input validation and sanitization

### Key Features
- Spotify API integration with rate limiting and exponential backoff
- Album availability analysis across different markets/regions
- ISRC (International Standard Recording Code) lookup
- Excel export functionality with formatting
- Dark theme optimized UI with region-based color coding

### Authentication
The app uses Streamlit secrets for Spotify API credentials:
- CLIENT_ID and CLIENT_SECRET must be configured in Streamlit secrets
- Uses Spotify's client credentials flow (no user authentication required)

### Data Processing
- Comprehensive album discovery (albums, singles, compilations)
- Market availability aggregation with region grouping
- Batch processing optimization for multiple artists
- Export functionality to formatted Excel files


## Copilot Assistance

This project was built and improved with the assistance of Copilot for code generation, refactoring, and documentation suggestions. Copilot was used to accelerate development, improve code quality, and automate repetitive tasks.

---

## Important Notes

- The app requires Spotify API credentials (CLIENT_ID and CLIENT_SECRET)
- All API calls implement rate limiting to respect Spotify's limits
- The tests directory appears to be empty but test files are referenced
- Documentation includes visual validation for dark theme compatibility

## TEST Directory Notes

- This directory is intended for development, testing, and documentation related to the Spotify ISRC Finder project.
- Place any experimental scripts, notebooks, or documentation drafts here.
- If you need to test new features or workflows, use this directory to avoid affecting production code.
- Local configuration files (such as `.env` or `.streamlit/secrets.toml`) should not be committed to version control.
- For sensitive credentials, always use the Streamlit secrets mechanism or environment variables.

## Local Setup Tips

- If you encounter import errors, ensure your working directory is set to the TEST directory root when running scripts.
- Use a virtual environment to manage dependencies:
  ```bash
  python -m venv .venv
  .venv\Scripts\activate  # On Windows
  pip install -r requirements.txt
  ```
- For Excel or data file testing, place sample files in the TEST directory or a subfolder and update your script paths accordingly.