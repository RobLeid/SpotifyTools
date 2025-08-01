"""
Album sorting utilities with extensible strategy pattern framework.

This module provides a base sorting framework for album data with support for
multiple sorting strategies. It follows the strategy pattern to allow easy
extension of sorting options (release date, album name, popularity, etc.).
"""

from typing import List, Dict, Any, Optional, Callable, Union, Tuple, NamedTuple
from datetime import datetime, date
from abc import ABC, abstractmethod
import logging
import time
import traceback
import re
import calendar
from enum import Enum
from functools import wraps

# Import sorting constants
try:
    from .constants import (
        SORTING_ERROR_CONFIG,
        SORTING_PERFORMANCE,
        DATE_HANDLING_CONFIG,
        SORTING_CONFIG
    )
except ImportError:
    # Fallback constants if import fails
    SORTING_ERROR_CONFIG = {
        "continue_on_error": True,
        "fallback_to_original_order": True,
        "log_individual_errors": True,
        "max_logged_errors": 10
    }
    SORTING_PERFORMANCE = {
        "performance_warning_threshold": 1000,
        "performance_timeout_ms": 500
    }
    DATE_HANDLING_CONFIG = {
        "min_valid_year": 1900,
        "max_valid_year": 2050,
        "warn_future_dates": True,
        "warn_old_dates": True,
        "old_date_threshold": 1950,
        "allow_partial_dates": True,
        "normalize_to_start_of_period": True
    }
    SORTING_CONFIG = {
        "max_strategies_per_sort": 5
    }

# Configure logging
logger = logging.getLogger(__name__)


class ParsedDate(NamedTuple):
    """Structured representation of a parsed date with metadata."""
    original_string: str
    normalized_datetime: datetime
    precision: str  # 'year', 'month', 'day'
    is_valid: bool
    validation_warnings: List[str]
    
    def get_sort_key(self) -> datetime:
        """Get the normalized datetime for sorting comparison."""
        return self.normalized_datetime
    
    def preserve_original_format(self) -> str:
        """Get the original date string to preserve in data exports."""
        return self.original_string


class SpotifyDateParser:
    """
    Comprehensive date parser for Spotify date formats with robust validation and normalization.
    
    This class handles all Spotify date formats (YYYY-MM-DD, YYYY-MM, YYYY) and provides
    comprehensive validation, normalization, and edge case handling while preserving
    the original date format for data exports.
    """
    
    # Date format patterns supported by Spotify
    DATE_PATTERNS = {
        'full_date': re.compile(r'^(\d{4})-(\d{2})-(\d{2})$'),      # YYYY-MM-DD
        'year_month': re.compile(r'^(\d{4})-(\d{2})$'),             # YYYY-MM
        'year_only': re.compile(r'^(\d{4})$'),                      # YYYY
        'iso_extended': re.compile(r'^(\d{4})-(\d{2})-(\d{2})T.*'), # YYYY-MM-DDTHH:MM:SS (extended ISO)
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the date parser with configuration.
        
        Args:
            config: Optional configuration dictionary for date handling
        """
        self.config = config or DATE_HANDLING_CONFIG.copy()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def parse_date(self, date_str: str, strict_validation: bool = True) -> ParsedDate:
        """
        Parse a Spotify date string into a structured ParsedDate object.
        
        Args:
            date_str: Date string from Spotify API
            strict_validation: Whether to apply strict validation rules
            
        Returns:
            ParsedDate object with parsing results and metadata
            
        Raises:
            DateParsingError: If date string cannot be parsed at all
        """
        if not isinstance(date_str, str):
            raise DateParsingError(f"Expected string, got {type(date_str)}")
        
        original_str = date_str
        date_str = date_str.strip()
        warnings = []
        
        if not date_str:
            raise DateParsingError("Empty date string provided")
        
        self.logger.debug(f"Parsing date string: '{original_str}'")
        
        # Try each pattern in order of specificity
        for precision, pattern in self.DATE_PATTERNS.items():
            match = pattern.match(date_str)
            if match:
                try:
                    parsed_date = self._parse_matched_date(
                        match, precision, original_str, strict_validation
                    )
                    warnings.extend(parsed_date.validation_warnings)
                    
                    self.logger.debug(
                        f"Successfully parsed '{original_str}' as {precision} "
                        f"-> {parsed_date.normalized_datetime}"
                    )
                    return parsed_date
                    
                except Exception as e:
                    self.logger.warning(f"Failed to parse matched {precision} pattern: {e}")
                    continue
        
        # If no patterns matched, raise an error
        raise DateParsingError(f"Unrecognized date format: '{original_str}'")
    
    def _parse_matched_date(
        self, 
        match: re.Match, 
        precision: str, 
        original_str: str, 
        strict_validation: bool
    ) -> ParsedDate:
        """Parse a matched date pattern into a ParsedDate object."""
        warnings = []
        
        try:
            if precision == 'full_date':
                year, month, day = map(int, match.groups())
                warnings.extend(self._validate_date_components(year, month, day, strict_validation))
                normalized_dt = datetime(year, month, day)
                
            elif precision == 'year_month':
                year, month = map(int, match.groups())
                warnings.extend(self._validate_date_components(year, month, 1, strict_validation))
                # Normalize to first day of month for comparison
                normalized_dt = datetime(year, month, 1)
                
            elif precision == 'year_only':
                year = int(match.groups()[0])
                warnings.extend(self._validate_year(year, strict_validation))
                # Normalize to January 1st for comparison
                normalized_dt = datetime(year, 1, 1)
                
            elif precision == 'iso_extended':
                # Handle ISO format with time component - extract just the date part
                year, month, day = map(int, match.groups()[:3])
                warnings.extend(self._validate_date_components(year, month, day, strict_validation))
                normalized_dt = datetime(year, month, day)
                precision = 'full_date'  # Treat as full date for precision
                
            else:
                raise DateParsingError(f"Unknown precision type: {precision}")
            
            # Additional edge case validations
            warnings.extend(self._validate_edge_cases(normalized_dt, strict_validation))
            
            is_valid = len([w for w in warnings if 'ERROR' in w.upper()]) == 0
            
            return ParsedDate(
                original_string=original_str,
                normalized_datetime=normalized_dt,
                precision=precision,
                is_valid=is_valid,
                validation_warnings=warnings
            )
            
        except ValueError as e:
            raise DateParsingError(f"Invalid date components in '{original_str}': {str(e)}")
        except Exception as e:
            raise DateParsingError(f"Failed to parse '{original_str}': {str(e)}")
    
    def _validate_date_components(
        self, 
        year: int, 
        month: int, 
        day: int, 
        strict_validation: bool
    ) -> List[str]:
        """Validate individual date components and return warnings."""
        warnings = []
        
        # Validate year
        warnings.extend(self._validate_year(year, strict_validation))
        
        # Validate month
        if not (1 <= month <= 12):
            if strict_validation:
                raise DateValidationError(f"Invalid month: {month}. Must be between 1 and 12.")
            else:
                warnings.append(f"ERROR: Invalid month {month}")
        
        # Validate day
        if not (1 <= day <= 31):
            if strict_validation:
                raise DateValidationError(f"Invalid day: {day}. Must be between 1 and 31.")
            else:
                warnings.append(f"ERROR: Invalid day {day}")
        
        # Validate day for specific month (including leap year handling)
        try:
            days_in_month = calendar.monthrange(year, month)[1]
            if day > days_in_month:
                if strict_validation:
                    raise DateValidationError(
                        f"Invalid day {day} for {calendar.month_name[month]} {year}. "
                        f"Month has only {days_in_month} days."
                    )
                else:
                    warnings.append(
                        f"ERROR: Invalid day {day} for {calendar.month_name[month]} {year}"
                    )
        except (ValueError, IndexError) as e:
            if strict_validation:
                raise DateValidationError(f"Date validation failed: {str(e)}")
            else:
                warnings.append(f"ERROR: Date validation failed: {str(e)}")
        
        return warnings
    
    def _validate_year(self, year: int, strict_validation: bool) -> List[str]:
        """Validate year and return warnings."""
        warnings = []
        
        min_year = self.config.get("min_valid_year", 1900)
        max_year = self.config.get("max_valid_year", 2050)
        current_year = datetime.now().year
        old_threshold = self.config.get("old_date_threshold", 1950)
        
        # Check reasonable bounds
        if year < min_year or year > max_year:
            warning_msg = f"Year {year} outside reasonable range [{min_year}, {max_year}]"
            if strict_validation and (year < min_year or year > max_year + 50):
                # Allow some leeway for future dates in non-strict mode
                raise DateValidationError(warning_msg)
            else:
                warnings.append(f"WARNING: {warning_msg}")
        
        # Check for future dates
        if year > current_year and self.config.get("warn_future_dates", True):
            warnings.append(f"WARNING: Future release date detected: {year}")
        
        # Check for very old dates
        if year < old_threshold and self.config.get("warn_old_dates", True):
            warnings.append(f"WARNING: Very old release date detected: {year}")
        
        return warnings
    
    def _validate_edge_cases(self, dt: datetime, strict_validation: bool) -> List[str]:
        """Validate edge cases like leap years and extreme dates."""
        warnings = []
        
        # Check for leap year correctness
        if dt.month == 2 and dt.day == 29:
            if not calendar.isleap(dt.year):
                if strict_validation:
                    raise DateValidationError(f"February 29 is invalid in non-leap year {dt.year}")
                else:
                    warnings.append(f"ERROR: February 29 invalid in non-leap year {dt.year}")
            else:
                warnings.append(f"INFO: Valid leap year date: {dt.strftime('%Y-%m-%d')}")
        
        # Check for very distant future dates
        current_date = datetime.now()
        if dt > current_date.replace(year=current_date.year + 10):
            warnings.append(f"WARNING: Very distant future date: {dt.strftime('%Y-%m-%d')}")
        
        # Check for historically impossible dates for music releases
        if dt.year < 1877:  # Before Edison's phonograph
            warnings.append(f"WARNING: Date predates recorded music history: {dt.strftime('%Y-%m-%d')}")
        
        return warnings
    
    def normalize_for_sorting(self, date_str: str) -> datetime:
        """
        Normalize a date string to a datetime object for sorting comparison.
        
        This method extracts just the normalized datetime for sorting,
        without the full ParsedDate metadata.
        
        Args:
            date_str: Date string to normalize
            
        Returns:
            datetime object for sorting comparison
            
        Raises:
            DateNormalizationError: If normalization fails
        """
        try:
            parsed = self.parse_date(date_str, strict_validation=False)
            if not parsed.is_valid:
                error_warnings = [w for w in parsed.validation_warnings if 'ERROR' in w.upper()]
                if error_warnings:
                    raise DateNormalizationError(f"Date normalization failed: {'; '.join(error_warnings)}")
            
            return parsed.normalized_datetime
            
        except DateParsingError as e:
            raise DateNormalizationError(f"Failed to normalize date '{date_str}': {str(e)}")
    
    def compare_dates(self, date1_str: str, date2_str: str) -> int:
        """
        Compare two date strings for sorting.
        
        Args:
            date1_str: First date string
            date2_str: Second date string
            
        Returns:
            -1 if date1 < date2, 0 if equal, 1 if date1 > date2
            
        Raises:
            DateNormalizationError: If either date cannot be normalized
        """
        try:
            dt1 = self.normalize_for_sorting(date1_str)
            dt2 = self.normalize_for_sorting(date2_str)
            
            if dt1 < dt2:
                return -1
            elif dt1 > dt2:
                return 1
            else:
                return 0
                
        except Exception as e:
            raise DateNormalizationError(f"Failed to compare dates '{date1_str}' and '{date2_str}': {str(e)}")
    
    def batch_parse_dates(
        self, 
        date_strings: List[str], 
        continue_on_error: bool = True
    ) -> Dict[str, ParsedDate]:
        """
        Parse multiple dates efficiently with error collection.
        
        Args:
            date_strings: List of date strings to parse
            continue_on_error: Whether to continue parsing after individual errors
            
        Returns:
            Dictionary mapping original date strings to ParsedDate objects
        """
        results = {}
        error_collector = ErrorCollector()
        
        self.logger.info(f"Batch parsing {len(date_strings)} date strings")
        
        for date_str in date_strings:
            try:
                parsed = self.parse_date(date_str, strict_validation=False)
                results[date_str] = parsed
                
                # Log significant warnings
                error_warnings = [w for w in parsed.validation_warnings if 'ERROR' in w.upper()]
                if error_warnings:
                    self.logger.warning(f"Date '{date_str}' has validation errors: {'; '.join(error_warnings)}")
                    
            except Exception as e:
                error_collector.add_error(e, {"date_string": date_str})
                
                if not continue_on_error:
                    raise DateParsingError(f"Batch parsing failed at '{date_str}': {str(e)}")
                
                # Create a fallback ParsedDate for failed parsing
                results[date_str] = ParsedDate(
                    original_string=date_str,
                    normalized_datetime=datetime(1900, 1, 1),  # Fallback date
                    precision='unknown',
                    is_valid=False,
                    validation_warnings=[f"ERROR: Parsing failed: {str(e)}"]
                )
        
        if error_collector.has_errors():
            summary = error_collector.get_summary()
            self.logger.warning(
                f"Batch parsing completed with {summary['total_errors']} errors out of {len(date_strings)} dates"
            )
        else:
            self.logger.info(f"Successfully batch parsed all {len(date_strings)} dates")
        
        return results


# Global date parser instance for reuse
_date_parser_instance = None

def get_date_parser() -> SpotifyDateParser:
    """Get a shared instance of the SpotifyDateParser."""
    global _date_parser_instance
    if _date_parser_instance is None:
        _date_parser_instance = SpotifyDateParser()
    return _date_parser_instance


# Convenience functions for date operations
def parse_spotify_date(date_str: str, strict: bool = False) -> ParsedDate:
    """
    Convenience function to parse a single Spotify date string.
    
    Args:
        date_str: Date string to parse
        strict: Whether to use strict validation
        
    Returns:
        ParsedDate object with parsing results
        
    Raises:
        DateParsingError: If parsing fails
    """
    parser = get_date_parser()
    return parser.parse_date(date_str, strict_validation=strict)


def normalize_spotify_date(date_str: str) -> Optional[datetime]:
    """
    Convenience function to normalize a Spotify date for sorting.
    
    Args:
        date_str: Date string to normalize
        
    Returns:
        Normalized datetime object or None if invalid
    """
    try:
        parser = get_date_parser()
        return parser.normalize_for_sorting(date_str)
    except DateNormalizationError:
        return None


def compare_spotify_dates(date1_str: str, date2_str: str) -> Optional[int]:
    """
    Convenience function to compare two Spotify date strings.
    
    Args:
        date1_str: First date string
        date2_str: Second date string
        
    Returns:
        -1 if date1 < date2, 0 if equal, 1 if date1 > date2, None if comparison fails
    """
    try:
        parser = get_date_parser()
        return parser.compare_dates(date1_str, date2_str)
    except DateNormalizationError:
        return None


def validate_spotify_date(date_str: str, strict: bool = True) -> Tuple[bool, List[str]]:
    """
    Convenience function to validate a Spotify date string.
    
    Args:
        date_str: Date string to validate
        strict: Whether to use strict validation
        
    Returns:
        Tuple of (is_valid, list_of_warnings)
    """
    try:
        parsed = parse_spotify_date(date_str, strict=strict)
        return parsed.is_valid, parsed.validation_warnings
    except DateParsingError as e:
        return False, [f"ERROR: {str(e)}"]


def batch_validate_spotify_dates(
    date_strings: List[str], 
    strict: bool = False
) -> Dict[str, Tuple[bool, List[str]]]:
    """
    Validate multiple Spotify date strings efficiently.
    
    Args:
        date_strings: List of date strings to validate
        strict: Whether to use strict validation
        
    Returns:
        Dictionary mapping date strings to (is_valid, warnings) tuples
    """
    results = {}
    parser = get_date_parser()
    
    logger.info(f"Batch validating {len(date_strings)} date strings")
    
    for date_str in date_strings:
        try:
            parsed = parser.parse_date(date_str, strict_validation=strict)
            results[date_str] = (parsed.is_valid, parsed.validation_warnings)
        except Exception as e:
            results[date_str] = (False, [f"ERROR: {str(e)}"])
    
    valid_count = sum(1 for is_valid, _ in results.values() if is_valid)
    logger.info(f"Batch validation complete: {valid_count}/{len(date_strings)} valid dates")
    
    return results


def get_date_edge_case_info(date_str: str) -> Dict[str, Any]:
    """
    Get detailed information about potential edge cases in a date string.
    
    Args:
        date_str: Date string to analyze
        
    Returns:
        Dictionary with edge case analysis
    """
    try:
        parsed = parse_spotify_date(date_str, strict=False)
        current_year = datetime.now().year
        
        edge_cases = {
            "is_leap_year_date": False,
            "is_future_date": False,
            "is_very_old_date": False,
            "is_historically_impossible": False,
            "precision_warnings": [],
            "validation_warnings": parsed.validation_warnings.copy()
        }
        
        # Check for leap year dates
        if parsed.normalized_datetime.month == 2 and parsed.normalized_datetime.day == 29:
            edge_cases["is_leap_year_date"] = True
            
        # Check for future dates
        if parsed.normalized_datetime.year > current_year:
            edge_cases["is_future_date"] = True
            
        # Check for very old dates
        if parsed.normalized_datetime.year < 1950:
            edge_cases["is_very_old_date"] = True
            
        # Check for historically impossible dates
        if parsed.normalized_datetime.year < 1877:
            edge_cases["is_historically_impossible"] = True
            
        # Add precision-specific warnings
        if parsed.precision == 'year_only':
            edge_cases["precision_warnings"].append("Date normalized to January 1st for year-only format")
        elif parsed.precision == 'year_month':
            edge_cases["precision_warnings"].append("Date normalized to first day of month")
            
        return edge_cases
        
    except Exception as e:
        return {
            "error": str(e),
            "is_leap_year_date": False,
            "is_future_date": False,
            "is_very_old_date": False,
            "is_historically_impossible": False,
            "precision_warnings": [],
            "validation_warnings": [f"ERROR: Failed to analyze date: {str(e)}"]
        }


def analyze_album_date_quality(albums: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze the quality and characteristics of release dates in a collection of albums.
    
    Args:
        albums: List of album dictionaries
        
    Returns:
        Dictionary with comprehensive date quality analysis
    """
    if not albums:
        return {"error": "Empty album list provided"}
    
    analysis = {
        "total_albums": len(albums),
        "albums_with_dates": 0,
        "albums_without_dates": 0,
        "valid_dates": 0,
        "invalid_dates": 0,
        "date_precision_counts": {"full_date": 0, "year_month": 0, "year_only": 0, "unknown": 0},
        "edge_case_counts": {
            "leap_year_dates": 0,
            "future_dates": 0,
            "very_old_dates": 0,
            "historically_impossible": 0
        },
        "date_range": {"earliest": None, "latest": None},
        "common_warnings": {},
        "sample_invalid_dates": []
    }
    
    date_parser = get_date_parser()
    all_dates = []
    
    logger.info(f"Analyzing date quality for {len(albums)} albums")
    
    for album in albums:
        album_id = album.get('id', 'unknown')
        release_date = album.get('release_date')
        
        if not release_date:
            analysis["albums_without_dates"] += 1
            continue
            
        analysis["albums_with_dates"] += 1
        
        try:
            parsed = date_parser.parse_date(release_date, strict_validation=False)
            
            if parsed.is_valid:
                analysis["valid_dates"] += 1
                analysis["date_precision_counts"][parsed.precision] += 1
                all_dates.append(parsed.normalized_datetime)
                
                # Check for edge cases
                edge_info = get_date_edge_case_info(release_date)
                for edge_type, is_edge in edge_info.items():
                    if edge_type.startswith("is_") and is_edge:
                        key = edge_type[3:]  # Remove "is_" prefix
                        if key in analysis["edge_case_counts"]:
                            analysis["edge_case_counts"][key] += 1
            else:
                analysis["invalid_dates"] += 1
                if len(analysis["sample_invalid_dates"]) < 5:
                    analysis["sample_invalid_dates"].append({
                        "album_id": album_id,
                        "date_string": release_date,
                        "warnings": parsed.validation_warnings
                    })
            
            # Collect warning frequencies
            for warning in parsed.validation_warnings:
                if warning not in analysis["common_warnings"]:
                    analysis["common_warnings"][warning] = 0
                analysis["common_warnings"][warning] += 1
                
        except Exception as e:
            analysis["invalid_dates"] += 1
            if len(analysis["sample_invalid_dates"]) < 5:
                analysis["sample_invalid_dates"].append({
                    "album_id": album_id,
                    "date_string": release_date,
                    "error": str(e)
                })
    
    # Calculate date range
    if all_dates:
        analysis["date_range"]["earliest"] = min(all_dates).isoformat()
        analysis["date_range"]["latest"] = max(all_dates).isoformat()
    
    # Calculate quality percentages
    if analysis["albums_with_dates"] > 0:
        analysis["valid_date_percentage"] = (analysis["valid_dates"] / analysis["albums_with_dates"]) * 100
    else:
        analysis["valid_date_percentage"] = 0
    
    analysis["overall_date_coverage"] = (analysis["albums_with_dates"] / analysis["total_albums"]) * 100
    
    logger.info(
        f"Date quality analysis complete: {analysis['valid_dates']}/{analysis['albums_with_dates']} "
        f"valid dates ({analysis['valid_date_percentage']:.1f}%), "
        f"{analysis['overall_date_coverage']:.1f}% coverage"
    )
    
    return analysis


class SortingError(Exception):
    """Base exception for sorting-related errors."""
    pass


class InvalidStrategyError(SortingError):
    """Raised when an invalid sorting strategy is requested."""
    pass


class DataValidationError(SortingError):
    """Raised when album data validation fails."""
    pass


class DateParsingError(SortingError):
    """Raised when date parsing fails."""
    pass


class DateValidationError(SortingError):
    """Raised when date validation fails."""
    pass


class DateNormalizationError(SortingError):
    """Raised when date normalization fails."""
    pass


class PerformanceError(SortingError):
    """Raised when sorting operations exceed performance thresholds."""
    pass


def performance_monitor(func):
    """Decorator to monitor sorting performance and log warnings."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Log performance metrics
            if execution_time > SORTING_PERFORMANCE["performance_timeout_ms"]:
                logger.warning(
                    f"Sorting operation {func.__name__} took {execution_time:.2f}ms, "
                    f"exceeding threshold of {SORTING_PERFORMANCE['performance_timeout_ms']}ms"
                )
            else:
                logger.debug(f"Sorting operation {func.__name__} completed in {execution_time:.2f}ms")
            
            return result
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(
                f"Sorting operation {func.__name__} failed after {execution_time:.2f}ms: {str(e)}"
            )
            raise
    return wrapper


def error_handler(func):
    """Decorator to handle and log sorting errors consistently."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except SortingError:
            # Re-raise sorting-specific errors
            raise
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Full traceback: {traceback.format_exc()}")
            
            # Convert to SortingError for consistent handling
            raise SortingError(f"Error in {func.__name__}: {str(e)}") from e
    return wrapper


class ErrorCollector:
    """Collects and manages errors during batch sorting operations."""
    
    def __init__(self, max_errors: int = None):
        self.max_errors = max_errors or SORTING_ERROR_CONFIG["max_logged_errors"]
        self.errors = []
        self.error_count = 0
    
    def add_error(self, error: Exception, context: Dict[str, Any] = None):
        """Add an error to the collection."""
        self.error_count += 1
        
        if len(self.errors) < self.max_errors:
            self.errors.append({
                "error": error,
                "context": context or {},
                "timestamp": datetime.now().isoformat()
            })
        
        # Log individual errors if configured
        if SORTING_ERROR_CONFIG["log_individual_errors"]:
            context_str = f" (context: {context})" if context else ""
            logger.warning(f"Sorting error #{self.error_count}: {str(error)}{context_str}")
    
    def has_errors(self) -> bool:
        """Check if any errors were collected."""
        return self.error_count > 0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of collected errors."""
        return {
            "total_errors": self.error_count,
            "logged_errors": len(self.errors),
            "errors": self.errors
        }


class SortOrder(Enum):
    """Enumeration for sort order directions."""
    ASCENDING = "asc"
    DESCENDING = "desc"


class SortingStrategy(ABC):
    """Abstract base class for album sorting strategies."""
    
    @abstractmethod
    def get_sort_key(self, album: Dict[str, Any]) -> Any:
        """
        Extract the sort key from an album object.
        
        Args:
            album: Album dictionary containing Spotify album data
            
        Returns:
            The value to use for sorting this album
        """
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return the name of this sorting strategy."""
        pass
    
    def handle_missing_data(self, album: Dict[str, Any]) -> Any:
        """
        Handle albums with missing or invalid sort data.
        
        Args:
            album: Album dictionary
            
        Returns:
            Default value to use when sort key is missing/invalid
        """
        return None


class ReleaseDateSortingStrategy(SortingStrategy):
    """Enhanced sorting strategy for album release dates with comprehensive parsing and validation."""
    
    def __init__(self, preserve_original: bool = True):
        """
        Initialize the release date sorting strategy.
        
        Args:
            preserve_original: Whether to preserve original date formats in album data
        """
        self.preserve_original = preserve_original
        self.date_parser = get_date_parser()
        self._parsed_dates_cache = {}  # Cache for parsed dates to improve performance
    
    def get_sort_key(self, album: Dict[str, Any]) -> Optional[datetime]:
        """
        Extract release date from album for sorting with comprehensive parsing.
        
        Args:
            album: Album dictionary containing Spotify album data
            
        Returns:
            datetime object for sorting, or None if invalid/missing
        """
        if not self._validate_album_data(album):
            return None
            
        release_date = album.get("release_date")
        if not release_date:
            logger.debug(f"No release date found for album {album.get('id', 'unknown')}")
            return None
            
        try:
            # Use cached result if available
            if release_date in self._parsed_dates_cache:
                parsed = self._parsed_dates_cache[release_date]
            else:
                parsed = self.date_parser.parse_date(release_date, strict_validation=False)
                self._parsed_dates_cache[release_date] = parsed
            
            # Log any validation warnings
            if parsed.validation_warnings:
                warning_msg = "; ".join(parsed.validation_warnings)
                logger.debug(
                    f"Date parsing warnings for album {album.get('id', 'unknown')} "
                    f"(date: '{release_date}'): {warning_msg}"
                )
            
            # Only return valid dates for sorting
            if parsed.is_valid:
                # Store parsed date metadata in album if preserving original
                if self.preserve_original and "release_date_parsed" not in album:
                    album["release_date_parsed"] = {
                        "original": parsed.original_string,
                        "normalized": parsed.normalized_datetime.isoformat(),
                        "precision": parsed.precision,
                        "warnings": parsed.validation_warnings
                    }
                
                return parsed.normalized_datetime
            else:
                error_warnings = [w for w in parsed.validation_warnings if 'ERROR' in w.upper()]
                logger.warning(
                    f"Invalid release date '{release_date}' for album {album.get('id', 'unknown')}: "
                    f"{'; '.join(error_warnings)}"
                )
                return None
                
        except (DateParsingError, DateNormalizationError) as e:
            logger.warning(
                f"Failed to parse release date '{release_date}' for album {album.get('id', 'unknown')}: {e}"
            )
            return None
        except Exception as e:
            logger.error(
                f"Unexpected error parsing release date '{release_date}' "
                f"for album {album.get('id', 'unknown')}: {e}"
            )
            return None
    
    def has_valid_release_date(self, album: Dict[str, Any]) -> bool:
        """
        Check if an album has a valid release date.
        
        Args:
            album: Album dictionary containing Spotify album data
            
        Returns:
            True if album has valid release date, False otherwise
        """
        if not self._validate_album_data(album):
            return False
            
        release_date = album.get("release_date")
        if not release_date or not isinstance(release_date, str) or not release_date.strip():
            return False
            
        try:
            # Use cached result if available
            if release_date in self._parsed_dates_cache:
                parsed = self._parsed_dates_cache[release_date]
            else:
                parsed = self.date_parser.parse_date(release_date, strict_validation=False)
                self._parsed_dates_cache[release_date] = parsed
            
            return parsed.is_valid
            
        except Exception as e:
            logger.debug(f"Date validation failed for album {album.get('id', 'unknown')}: {e}")
            return False
    
    def identify_missing_invalid_dates(self, albums: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Identify albums with missing or invalid release dates.
        
        Args:
            albums: List of album dictionaries to analyze
            
        Returns:
            Dictionary with 'valid', 'missing', and 'invalid' keys containing album lists
        """
        result = {
            "valid": [],
            "missing": [],
            "invalid": []
        }
        
        logger.info(f"Analyzing release dates for {len(albums)} albums")
        
        for album in albums:
            if not isinstance(album, dict):
                result["invalid"].append(album)
                continue
                
            release_date = album.get("release_date")
            
            # Check for missing release date
            if not release_date or not isinstance(release_date, str) or not release_date.strip():
                result["missing"].append(album)
                logger.debug(f"Missing release date for album {album.get('id', 'unknown')}")
                continue
            
            # Check for invalid release date
            try:
                if release_date in self._parsed_dates_cache:
                    parsed = self._parsed_dates_cache[release_date]
                else:
                    parsed = self.date_parser.parse_date(release_date, strict_validation=False)
                    self._parsed_dates_cache[release_date] = parsed
                
                if parsed.is_valid:
                    result["valid"].append(album)
                else:
                    result["invalid"].append(album)
                    error_warnings = [w for w in parsed.validation_warnings if 'ERROR' in w.upper()]
                    logger.debug(f"Invalid release date '{release_date}' for album {album.get('id', 'unknown')}: {'; '.join(error_warnings)}")
                    
            except Exception as e:
                result["invalid"].append(album)
                logger.debug(f"Date parsing failed for album {album.get('id', 'unknown')} with date '{release_date}': {e}")
        
        logger.info(
            f"Date analysis complete: {len(result['valid'])} valid, "
            f"{len(result['missing'])} missing, {len(result['invalid'])} invalid"
        )
        
        return result
    
    def _validate_album_data(self, album: Dict[str, Any]) -> bool:
        """Validate that album data is in expected format."""
        if not isinstance(album, dict):
            logger.error(f"Expected album dict, got {type(album)}")
            return False
        
        if "id" not in album:
            logger.warning("Album missing required 'id' field")
            # Don't fail completely, just log warning
        
        return True
    
    def get_parsed_date_info(self, album: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Get detailed parsing information for an album's release date.
        
        Args:
            album: Album dictionary
            
        Returns:
            Dictionary with parsing details or None if not available
        """
        release_date = album.get("release_date")
        if not release_date:
            return None
        
        try:
            if release_date in self._parsed_dates_cache:
                parsed = self._parsed_dates_cache[release_date]
            else:
                parsed = self.date_parser.parse_date(release_date, strict_validation=False)
                self._parsed_dates_cache[release_date] = parsed
            
            return {
                "original_string": parsed.original_string,
                "normalized_datetime": parsed.normalized_datetime,
                "precision": parsed.precision,
                "is_valid": parsed.is_valid,
                "validation_warnings": parsed.validation_warnings,
                "sort_key": parsed.normalized_datetime if parsed.is_valid else None
            }
            
        except Exception as e:
            logger.warning(f"Failed to get parsing info for date '{release_date}': {e}")
            return None
    
    def normalize_date_for_comparison(self, date_str: str) -> Optional[datetime]:
        """
        Normalize a date string for comparison operations.
        
        This method is used for date comparison utilities and preserves
        the original date format in the data.
        
        Args:
            date_str: Date string to normalize
            
        Returns:
            Normalized datetime for comparison or None if invalid
        """
        try:
            return self.date_parser.normalize_for_sorting(date_str)
        except DateNormalizationError as e:
            logger.warning(f"Failed to normalize date '{date_str}' for comparison: {e}")
            return None
    
    def compare_release_dates(self, album1: Dict[str, Any], album2: Dict[str, Any]) -> int:
        """
        Compare release dates of two albums.
        
        Args:
            album1: First album dictionary
            album2: Second album dictionary
            
        Returns:
            -1 if album1 release date < album2, 0 if equal, 1 if album1 > album2
            None values are treated as "far future" for consistent sorting
        """
        date1 = self.get_sort_key(album1)
        date2 = self.get_sort_key(album2)
        
        # Handle None values (missing dates)
        if date1 is None and date2 is None:
            return 0
        elif date1 is None:
            return 1  # None sorts last
        elif date2 is None:
            return -1  # None sorts last
        
        # Compare normalized dates
        if date1 < date2:
            return -1
        elif date1 > date2:
            return 1
        else:
            return 0
    
    def clear_cache(self):
        """Clear the internal date parsing cache."""
        self._parsed_dates_cache.clear()
        logger.debug("Cleared release date parsing cache")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get statistics about the parsing cache."""
        return {
            "cached_dates": len(self._parsed_dates_cache),
            "valid_dates": len([p for p in self._parsed_dates_cache.values() if p.is_valid]),
            "invalid_dates": len([p for p in self._parsed_dates_cache.values() if not p.is_valid])
        }
    
    def get_strategy_name(self) -> str:
        """Return the name of this sorting strategy."""
        return "release_date"
    
    def handle_missing_data(self, album: Dict[str, Any]) -> datetime:
        """
        Handle albums with missing release dates.
        
        Returns a far future date to sort unknown dates last when ascending,
        or far past date when descending.
        """
        return datetime(9999, 12, 31)


class AlbumNameSortingStrategy(SortingStrategy):
    """Sorting strategy for album names."""
    
    def get_sort_key(self, album: Dict[str, Any]) -> str:
        """
        Extract album name for sorting.
        
        Args:
            album: Album dictionary containing Spotify album data
            
        Returns:
            Album name in lowercase for case-insensitive sorting
        """
        name = album.get("name", "")
        return name.lower().strip() if name else ""
    
    def get_strategy_name(self) -> str:
        """Return the name of this sorting strategy."""
        return "album_name"
    
    def handle_missing_data(self, album: Dict[str, Any]) -> str:
        """Handle albums with missing names."""
        return "zzz_unknown"  # Sort unknown names last


class AlbumIdSortingStrategy(SortingStrategy):
    """Sorting strategy for album IDs (used as secondary sort key)."""
    
    def get_sort_key(self, album: Dict[str, Any]) -> str:
        """
        Extract album ID for sorting.
        
        Args:
            album: Album dictionary containing Spotify album data
            
        Returns:
            Album ID for stable sorting
        """
        return album.get("id", "")
    
    def get_strategy_name(self) -> str:
        """Return the name of this sorting strategy."""
        return "album_id"
    
    def handle_missing_data(self, album: Dict[str, Any]) -> str:
        """Handle albums with missing IDs."""
        return "zzz_unknown_id"


class PopularitySortingStrategy(SortingStrategy):
    """Sorting strategy for album popularity scores."""
    
    def get_sort_key(self, album: Dict[str, Any]) -> int:
        """
        Extract popularity score for sorting.
        
        Args:
            album: Album dictionary containing Spotify album data
            
        Returns:
            Popularity score (0-100) for sorting
        """
        if not isinstance(album, dict):
            logger.error(f"Expected album dict, got {type(album)}")
            return -1
            
        popularity = album.get("popularity")
        if popularity is None:
            logger.debug(f"No popularity score for album {album.get('id', 'unknown')}")
            return -1  # Sort unknown popularity last
        
        try:
            score = int(popularity)
            if not (0 <= score <= 100):
                logger.warning(
                    f"Popularity score {score} outside valid range [0, 100] "
                    f"for album {album.get('id', 'unknown')}"
                )
                return max(0, min(100, score))  # Clamp to valid range
            return score
        except (ValueError, TypeError) as e:
            logger.warning(
                f"Invalid popularity value '{popularity}' for album {album.get('id', 'unknown')}: {e}"
            )
            return -1
    
    def get_strategy_name(self) -> str:
        """Return the name of this sorting strategy."""
        return "popularity"
    
    def handle_missing_data(self, album: Dict[str, Any]) -> int:
        """Handle albums with missing popularity scores."""
        return -1


class TrackCountSortingStrategy(SortingStrategy):
    """Sorting strategy for number of tracks in an album."""
    
    def get_sort_key(self, album: Dict[str, Any]) -> int:
        """
        Extract track count for sorting.
        
        Args:
            album: Album dictionary containing Spotify album data
            
        Returns:
            Number of tracks for sorting
        """
        if not isinstance(album, dict):
            logger.error(f"Expected album dict, got {type(album)}")
            return 0
            
        total_tracks = album.get("total_tracks")
        if total_tracks is None:
            logger.debug(f"No track count for album {album.get('id', 'unknown')}")
            return 0
        
        try:
            count = int(total_tracks)
            if count < 0:
                logger.warning(
                    f"Negative track count {count} for album {album.get('id', 'unknown')}, using 0"
                )
                return 0
            return count
        except (ValueError, TypeError) as e:
            logger.warning(
                f"Invalid track count '{total_tracks}' for album {album.get('id', 'unknown')}: {e}"
            )
            return 0
    
    def get_strategy_name(self) -> str:
        """Return the name of this sorting strategy."""
        return "track_count"
    
    def handle_missing_data(self, album: Dict[str, Any]) -> int:
        """Handle albums with missing track counts."""
        return 0


class ArtistNameSortingStrategy(SortingStrategy):
    """Sorting strategy for primary artist name."""
    
    def get_sort_key(self, album: Dict[str, Any]) -> str:
        """
        Extract primary artist name for sorting.
        
        Args:
            album: Album dictionary containing Spotify album data
            
        Returns:
            Primary artist name in lowercase for case-insensitive sorting
        """
        artists = album.get("artists", [])
        if not artists or not isinstance(artists, list):
            return ""
        
        # Use the first artist as primary
        first_artist = artists[0]
        if isinstance(first_artist, dict):
            name = first_artist.get("name", "")
        else:
            name = str(first_artist) if first_artist else ""
        
        return name.lower().strip() if name else ""
    
    def get_strategy_name(self) -> str:
        """Return the name of this sorting strategy."""
        return "artist_name"
    
    def handle_missing_data(self, album: Dict[str, Any]) -> str:
        """Handle albums with missing artist names."""
        return "zzz_unknown_artist"


class ReverseSortString:
    """Helper class for reverse string sorting in composite strategies."""
    
    def __init__(self, value: str):
        self.value = value
    
    def __lt__(self, other):
        if isinstance(other, ReverseSortString):
            return self.value > other.value
        return self.value > str(other)
    
    def __le__(self, other):
        if isinstance(other, ReverseSortString):
            return self.value >= other.value
        return self.value >= str(other)
    
    def __gt__(self, other):
        if isinstance(other, ReverseSortString):
            return self.value < other.value
        return self.value < str(other)
    
    def __ge__(self, other):
        if isinstance(other, ReverseSortString):
            return self.value <= other.value
        return self.value <= str(other)
    
    def __eq__(self, other):
        if isinstance(other, ReverseSortString):
            return self.value == other.value
        return self.value == str(other)
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __str__(self):
        return self.value
    
    def __repr__(self):
        return f"ReverseSortString({self.value!r})"


class AlbumSorter:
    """
    Main class for sorting albums using configurable strategies.
    
    This class implements the strategy pattern to allow flexible sorting
    of album collections by different criteria.
    """
    
    def __init__(self, preserve_original_dates: bool = True):
        """
        Initialize the AlbumSorter with available strategies.
        
        Args:
            preserve_original_dates: Whether to preserve original date formats in album data
        """
        self._strategies = {
            "release_date": ReleaseDateSortingStrategy(preserve_original=preserve_original_dates),
            "album_name": AlbumNameSortingStrategy(),
            "album_id": AlbumIdSortingStrategy(),
            "popularity": PopularitySortingStrategy(),
            "track_count": TrackCountSortingStrategy(),
            "artist_name": ArtistNameSortingStrategy(),
        }
    
    def register_strategy(self, name: str, strategy: SortingStrategy) -> None:
        """
        Register a new sorting strategy.
        
        Args:
            name: Name identifier for the strategy
            strategy: SortingStrategy instance
        """
        self._strategies[name] = strategy
        logger.info(f"Registered new sorting strategy: {name}")
    
    def get_available_strategies(self) -> List[str]:
        """Return list of available sorting strategy names."""
        return list(self._strategies.keys())
    
    def get_strategy_info(self) -> Dict[str, str]:
        """
        Return detailed information about available strategies.
        
        Returns:
            Dictionary mapping strategy names to their descriptions
        """
        return {
            "release_date": "Sort by album release date (newest first by default)",
            "album_name": "Sort alphabetically by album name",
            "album_id": "Sort by Spotify album ID (for stable sorting)",
            "popularity": "Sort by popularity score (0-100, highest first by default)",
            "track_count": "Sort by number of tracks in album",
            "artist_name": "Sort alphabetically by primary artist name"
        }
    
    @error_handler
    def create_composite_strategy(self, strategies: List[Tuple[str, SortOrder]]) -> Callable[[Dict[str, Any]], Tuple]:
        """
        Create a composite sorting strategy that combines multiple strategies.
        
        Args:
            strategies: List of tuples (strategy_name, sort_order) for multi-level sorting
            
        Returns:
            A function that returns a tuple of sort keys for the given strategies
            
        Raises:
            InvalidStrategyError: If any strategy name is invalid
            DataValidationError: If strategies list is invalid
        """
        if not strategies:
            raise DataValidationError("Strategies list cannot be empty")
        
        if len(strategies) > SORTING_CONFIG["max_strategies_per_sort"]:
            raise DataValidationError(
                f"Too many strategies ({len(strategies)}). "
                f"Maximum allowed: {SORTING_CONFIG['max_strategies_per_sort']}"
            )
        
        # Validate all strategies exist
        for strategy_name, order in strategies:
            if strategy_name not in self._strategies:
                raise InvalidStrategyError(f"Unknown strategy: {strategy_name}")
            if not isinstance(order, SortOrder):
                raise DataValidationError(f"Invalid sort order for {strategy_name}: {order}")
        
        logger.info(f"Creating composite strategy with {len(strategies)} sorting levels")
        
        def composite_key_func(album: Dict[str, Any]) -> Tuple:
            keys = []
            error_collector = ErrorCollector()
            
            for strategy_name, order in strategies:
                strategy = self._strategies[strategy_name]
                try:
                    key = strategy.get_sort_key(album)
                    if key is None:
                        key = strategy.handle_missing_data(album)
                        logger.debug(
                            f"Using fallback value for {strategy_name} "
                            f"on album {album.get('id', 'unknown')}"
                        )
                    
                    # Handle different data types for reverse sorting
                    if order == SortOrder.DESCENDING:
                        if isinstance(key, (int, float)):
                            # For numeric values, multiply by -1
                            key = -key
                        elif isinstance(key, datetime):
                            # For datetime, use negative timestamp
                            key = -key.timestamp()
                        elif isinstance(key, str):
                            # For strings, use reverse sort wrapper
                            key = ReverseSortString(key)
                        else:
                            # For other types, convert to string and reverse
                            key = ReverseSortString(str(key))
                    
                    keys.append(key)
                except Exception as e:
                    error_collector.add_error(e, {
                        "strategy": strategy_name,
                        "album_id": album.get('id', 'unknown'),
                        "album_name": album.get('name', 'unknown')
                    })
                    
                    # Use fallback value for failed strategy
                    try:
                        fallback_key = strategy.handle_missing_data(album)
                        if order == SortOrder.DESCENDING and isinstance(fallback_key, str):
                            fallback_key = ReverseSortString(fallback_key)
                        keys.append(fallback_key)
                    except Exception as fallback_error:
                        logger.error(f"Fallback failed for {strategy_name}: {fallback_error}")
                        keys.append("zzz_error_fallback")
            
            # Log error summary if errors occurred
            if error_collector.has_errors():
                summary = error_collector.get_summary()
                logger.warning(
                    f"Composite key generation had {summary['total_errors']} errors "
                    f"for album {album.get('id', 'unknown')}"
                )
            
            return tuple(keys)
        
        return composite_key_func
    
    @performance_monitor
    @error_handler
    def sort_albums_multi(
        self,
        albums: List[Dict[str, Any]],
        strategies: List[Tuple[str, SortOrder]],
        separate_unknown: bool = True,
        preserve_unknown_order: bool = True
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Sort albums using multiple strategies in order of priority.
        
        Args:
            albums: List of album dictionaries to sort
            strategies: List of tuples (strategy_name, sort_order) in priority order
            separate_unknown: Whether to separate albums with missing primary sort data
            preserve_unknown_order: Whether to preserve original order for unknown albums
            
        Returns:
            Dictionary with 'known' and 'unknown' keys containing sorted album lists
            
        Raises:
            DataValidationError: If input validation fails
            InvalidStrategyError: If any strategy names are invalid
            PerformanceError: If operation exceeds performance thresholds
        """
        # Input validation
        if not isinstance(albums, list):
            raise DataValidationError(f"Expected list of albums, got {type(albums)}")
        
        if not albums:
            logger.debug("Empty album list provided for sorting")
            return {"known": [], "unknown": []}
        
        if not strategies:
            raise DataValidationError("At least one sorting strategy must be provided")
        
        # Performance warning for large datasets
        if len(albums) > SORTING_PERFORMANCE["performance_warning_threshold"]:
            logger.warning(
                f"Sorting large dataset ({len(albums)} albums). "
                f"This may impact performance."
            )
        
        primary_strategy_name = strategies[0][0]
        if primary_strategy_name not in self._strategies:
            raise InvalidStrategyError(f"Unknown primary strategy: {primary_strategy_name}")
        
        primary_strategy = self._strategies[primary_strategy_name]
        
        logger.info(
            f"Multi-sorting {len(albums)} albums with {len(strategies)} strategies: "
            f"{[s[0] for s in strategies]}"
        )
        
        error_collector = ErrorCollector()
        
        # Separate albums with valid vs missing primary sort data, preserving original order
        known_albums = []
        unknown_albums = []
        
        for i, album in enumerate(albums):
            try:
                if not isinstance(album, dict):
                    error_collector.add_error(
                        DataValidationError(f"Album at index {i} is not a dictionary"),
                        {"index": i, "type": type(album)}
                    )
                    # Add original index for order preservation
                    album_with_index = {"_original_index": i, "_album_data": album}
                    unknown_albums.append(album_with_index)
                    continue
                
                # Add original index to preserve order if needed
                album_with_index = dict(album)  # Create a copy
                album_with_index["_original_index"] = i
                
                primary_key = primary_strategy.get_sort_key(album)
                fallback_key = primary_strategy.handle_missing_data(album)
                
                if primary_key is not None and primary_key != fallback_key:
                    known_albums.append(album_with_index)
                else:
                    unknown_albums.append(album_with_index)
                    if primary_key is None:
                        logger.debug(
                            f"Album {album.get('id', 'unknown')} has missing primary sort data"
                        )
                        
            except Exception as e:
                error_collector.add_error(e, {
                    "album_id": album.get('id', 'unknown') if isinstance(album, dict) else 'invalid',
                    "album_index": i,
                    "strategy": primary_strategy_name
                })
                # Add original index for order preservation
                album_with_index = dict(album) if isinstance(album, dict) else {"_album_data": album}
                album_with_index["_original_index"] = i
                unknown_albums.append(album_with_index)
        
        # Create composite sorting function
        try:
            composite_key_func = self.create_composite_strategy(strategies)
        except Exception as e:
            error_collector.add_error(e, {"context": "composite_strategy_creation"})
            logger.error(f"Failed to create composite strategy: {e}")
            if SORTING_ERROR_CONFIG["fallback_to_original_order"]:
                logger.info("Falling back to original album order")
                return {"known": albums, "unknown": []}
            raise
        
        # Sort albums with known primary data
        try:
            logger.debug(f"Sorting {len(known_albums)} albums with known primary data")
            known_albums.sort(key=composite_key_func)
        except Exception as e:
            error_collector.add_error(e, {"context": "known_albums_sorting"})
            logger.error(f"Error sorting known albums: {e}")
            
            if SORTING_ERROR_CONFIG["fallback_to_original_order"]:
                logger.info("Falling back to original order for known albums")
                # Return original order if sorting fails
                return {"known": albums, "unknown": []}
            raise SortingError(f"Failed to sort known albums: {e}")
        
        # Handle unknown albums - preserve original order or sort by secondary strategies
        if unknown_albums:
            if preserve_unknown_order:
                # Sort by original index to preserve order
                logger.debug(f"Preserving original order for {len(unknown_albums)} albums with unknown primary data")
                unknown_albums.sort(key=lambda x: x.get("_original_index", 0))
            elif len(strategies) > 1:
                try:
                    secondary_strategies = strategies[1:]
                    logger.debug(
                        f"Sorting {len(unknown_albums)} albums with unknown primary data "
                        f"using {len(secondary_strategies)} secondary strategies"
                    )
                    secondary_key_func = self.create_composite_strategy(secondary_strategies)
                    unknown_albums.sort(key=secondary_key_func)
                except Exception as e:
                    error_collector.add_error(e, {"context": "unknown_albums_sorting"})
                    logger.warning(f"Error sorting unknown albums: {e}. Preserving original order.")
                    # Fallback to original order for unknown albums
                    unknown_albums.sort(key=lambda x: x.get("_original_index", 0))
        
        # Clean up the _original_index field from results
        def clean_album(album_data):
            if isinstance(album_data, dict):
                cleaned = {k: v for k, v in album_data.items() if k != "_original_index"}
                # Handle wrapped album data
                if "_album_data" in cleaned:
                    return cleaned["_album_data"]
                return cleaned
            return album_data
        
        known_albums = [clean_album(album) for album in known_albums]
        unknown_albums = [clean_album(album) for album in unknown_albums]
        
        # Log error summary
        if error_collector.has_errors():
            summary = error_collector.get_summary()
            logger.warning(
                f"Sorting completed with {summary['total_errors']} errors. "
                f"Known albums: {len(known_albums)}, Unknown albums: {len(unknown_albums)}"
            )
        else:
            logger.info(
                f"Sorting completed successfully. "
                f"Known albums: {len(known_albums)}, Unknown albums: {len(unknown_albums)}"
            )
        
        if separate_unknown:
            return {"known": known_albums, "unknown": unknown_albums}
        else:
            return {"known": known_albums + unknown_albums, "unknown": []}
    
    @performance_monitor
    @error_handler
    def sort_albums(
        self,
        albums: List[Dict[str, Any]],
        primary_strategy: str = "release_date",
        secondary_strategy: str = "album_id",
        order: SortOrder = SortOrder.DESCENDING,
        separate_unknown: bool = True,
        preserve_unknown_order: bool = True
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Sort albums using the specified strategies.
        
        Args:
            albums: List of album dictionaries to sort
            primary_strategy: Name of primary sorting strategy
            secondary_strategy: Name of secondary sorting strategy  
            order: Sort order (ascending or descending)
            separate_unknown: Whether to separate albums with missing primary sort data
            preserve_unknown_order: Whether to preserve original order for unknown albums
            
        Returns:
            Dictionary with 'known' and 'unknown' keys containing sorted album lists
            
        Raises:
            DataValidationError: If input validation fails
            InvalidStrategyError: If strategy names are invalid
        """
        # Input validation
        if not isinstance(albums, list):
            raise DataValidationError(f"Expected list of albums, got {type(albums)}")
            
        if not albums:
            logger.debug("Empty album list provided for sorting")
            return {"known": [], "unknown": []}
        
        # Validate strategies
        if primary_strategy not in self._strategies:
            raise InvalidStrategyError(f"Unknown primary strategy: {primary_strategy}")
        if secondary_strategy not in self._strategies:
            raise InvalidStrategyError(f"Unknown secondary strategy: {secondary_strategy}")
        
        if not isinstance(order, SortOrder):
            raise DataValidationError(f"Invalid sort order: {order}")
        
        # Performance warning for large datasets
        if len(albums) > SORTING_PERFORMANCE["performance_warning_threshold"]:
            logger.warning(
                f"Sorting large dataset ({len(albums)} albums) with strategies "
                f"{primary_strategy} -> {secondary_strategy}"
            )
        
        primary_strat = self._strategies[primary_strategy]
        secondary_strat = self._strategies[secondary_strategy]
        
        logger.info(
            f"Sorting {len(albums)} albums by {primary_strategy} ({order.value}), "
            f"then by {secondary_strategy}"
        )
        
        error_collector = ErrorCollector()
        
        # Separate albums with valid vs missing primary sort data, preserving original order
        known_albums = []
        unknown_albums = []
        
        for i, album in enumerate(albums):
            try:
                if not isinstance(album, dict):
                    error_collector.add_error(
                        DataValidationError(f"Album at index {i} is not a dictionary"),
                        {"index": i, "type": type(album)}
                    )
                    # Add original index for order preservation
                    album_with_index = {"_original_index": i, "_album_data": album}
                    unknown_albums.append(album_with_index)
                    continue
                
                # Add original index to preserve order if needed
                album_with_index = dict(album)  # Create a copy
                album_with_index["_original_index"] = i
                
                primary_key = primary_strat.get_sort_key(album)
                fallback_key = primary_strat.handle_missing_data(album)
                
                if primary_key is not None and primary_key != fallback_key:
                    known_albums.append(album_with_index)
                else:
                    unknown_albums.append(album_with_index)
                    if primary_key is None:
                        logger.debug(
                            f"Album {album.get('id', 'unknown')} has missing primary sort data"
                        )
                        
            except Exception as e:
                error_collector.add_error(e, {
                    "album_id": album.get('id', 'unknown') if isinstance(album, dict) else 'invalid',
                    "album_index": i,
                    "strategy": primary_strategy
                })
                # Add original index for order preservation
                album_with_index = dict(album) if isinstance(album, dict) else {"_album_data": album}
                album_with_index["_original_index"] = i
                unknown_albums.append(album_with_index)
        
        # Sort albums with known primary data
        try:
            logger.debug(f"Sorting {len(known_albums)} albums with known primary data")
            
            def safe_sort_key(album):
                try:
                    primary_key = primary_strat.get_sort_key(album) or primary_strat.handle_missing_data(album)
                    secondary_key = secondary_strat.get_sort_key(album) or secondary_strat.handle_missing_data(album)
                    return (primary_key, secondary_key)
                except Exception as e:
                    logger.warning(f"Error getting sort key for album {album.get('id', 'unknown')}: {e}")
                    return (primary_strat.handle_missing_data(album), secondary_strat.handle_missing_data(album))
            
            known_albums.sort(
                key=safe_sort_key,
                reverse=(order == SortOrder.DESCENDING)
            )
        except Exception as e:
            error_collector.add_error(e, {"context": "known_albums_sorting"})
            logger.error(f"Error sorting known albums: {e}")
            
            if SORTING_ERROR_CONFIG["fallback_to_original_order"]:
                logger.info("Falling back to original order for known albums")
                return {"known": albums, "unknown": []}
            raise SortingError(f"Failed to sort known albums: {e}")
        
        # Handle unknown albums - preserve original order or sort by secondary strategy
        if unknown_albums:
            if preserve_unknown_order:
                # Sort by original index to preserve order
                logger.debug(f"Preserving original order for {len(unknown_albums)} albums with unknown primary data")
                unknown_albums.sort(key=lambda x: x.get("_original_index", 0))
            else:
                try:
                    logger.debug(f"Sorting {len(unknown_albums)} albums with unknown primary data by secondary strategy")
                    
                    def safe_secondary_sort_key(album):
                        try:
                            return secondary_strat.get_sort_key(album) or secondary_strat.handle_missing_data(album)
                        except Exception as e:
                            logger.warning(f"Error getting secondary sort key for album {album.get('id', 'unknown')}: {e}")
                            return secondary_strat.handle_missing_data(album)
                    
                    unknown_albums.sort(key=safe_secondary_sort_key)
                except Exception as e:
                    error_collector.add_error(e, {"context": "unknown_albums_sorting"})
                    logger.warning(f"Error sorting unknown albums: {e}. Preserving original order.")
                    # Fallback to original order for unknown albums
                    unknown_albums.sort(key=lambda x: x.get("_original_index", 0))
        
        # Clean up the _original_index field from results
        def clean_album(album_data):
            if isinstance(album_data, dict):
                cleaned = {k: v for k, v in album_data.items() if k != "_original_index"}
                # Handle wrapped album data
                if "_album_data" in cleaned:
                    return cleaned["_album_data"]
                return cleaned
            return album_data
        
        known_albums = [clean_album(album) for album in known_albums]
        unknown_albums = [clean_album(album) for album in unknown_albums]
        
        # Log error summary
        if error_collector.has_errors():
            summary = error_collector.get_summary()
            logger.warning(
                f"Sorting completed with {summary['total_errors']} errors. "
                f"Known albums: {len(known_albums)}, Unknown albums: {len(unknown_albums)}"
            )
        else:
            logger.info(
                f"Sorting completed successfully. "
                f"Known albums: {len(known_albums)}, Unknown albums: {len(unknown_albums)}"
            )
        
        if separate_unknown:
            return {"known": known_albums, "unknown": unknown_albums}
        else:
            # Combine lists with known albums first
            return {"known": known_albums + unknown_albums, "unknown": []}
    
    @performance_monitor
    @error_handler
    def sort_albums_by_type(
        self,
        albums_by_type: Dict[str, List[Dict[str, Any]]],
        primary_strategy: str = "release_date",
        secondary_strategy: str = "album_id",
        order: SortOrder = SortOrder.DESCENDING,
        separate_unknown: bool = True
    ) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """
        Sort albums grouped by type (album, single, compilation).
        
        Args:
            albums_by_type: Dictionary mapping album types to album lists
            primary_strategy: Name of primary sorting strategy
            secondary_strategy: Name of secondary sorting strategy
            order: Sort order (ascending or descending)
            separate_unknown: Whether to separate albums with missing primary sort data
            
        Returns:
            Dictionary mapping album types to sorted results
            
        Raises:
            DataValidationError: If input validation fails
            InvalidStrategyError: If strategy names are invalid
        """
        if not isinstance(albums_by_type, dict):
            raise DataValidationError(f"Expected dictionary, got {type(albums_by_type)}")
        
        if not albums_by_type:
            logger.debug("Empty albums_by_type dictionary provided")
            return {}
        
        result = {}
        error_collector = ErrorCollector()
        total_albums = sum(len(albums) for albums in albums_by_type.values())
        
        logger.info(
            f"Sorting {total_albums} albums across {len(albums_by_type)} types "
            f"using {primary_strategy} -> {secondary_strategy}"
        )
        
        for album_type, albums in albums_by_type.items():
            try:
                if not isinstance(albums, list):
                    error_collector.add_error(
                        DataValidationError(f"Albums for type '{album_type}' is not a list"),
                        {"album_type": album_type, "type": type(albums)}
                    )
                    result[album_type] = {"known": [], "unknown": []}
                    continue
                
                result[album_type] = self.sort_albums(
                    albums, primary_strategy, secondary_strategy, order, separate_unknown
                )
                
                # Log success for this type
                sorted_result = result[album_type]
                total_sorted = len(sorted_result["known"]) + len(sorted_result["unknown"])
                logger.debug(
                    f"Sorted {total_sorted} {album_type}s "
                    f"(known: {len(sorted_result['known'])}, unknown: {len(sorted_result['unknown'])})"
                )
                
            except Exception as e:
                error_collector.add_error(e, {
                    "album_type": album_type,
                    "album_count": len(albums) if isinstance(albums, list) else "unknown"
                })
                
                if SORTING_ERROR_CONFIG["fallback_to_original_order"]:
                    logger.warning(f"Error sorting {album_type} albums: {e}. Using original order.")
                    result[album_type] = {"known": albums if isinstance(albums, list) else [], "unknown": []}
                else:
                    raise SortingError(f"Failed to sort {album_type} albums: {e}")
        
        # Log final summary
        if error_collector.has_errors():
            summary = error_collector.get_summary()
            logger.warning(
                f"Album type sorting completed with {summary['total_errors']} errors "
                f"across {len(albums_by_type)} album types"
            )
        else:
            logger.info(f"Successfully sorted all {len(albums_by_type)} album types")
        
        return result
    
    @performance_monitor
    @error_handler
    def sort_albums_by_type_multi(
        self,
        albums_by_type: Dict[str, List[Dict[str, Any]]],
        strategies: List[Tuple[str, SortOrder]],
        separate_unknown: bool = True
    ) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """
        Sort albums grouped by type using multiple strategies.
        
        Args:
            albums_by_type: Dictionary mapping album types to album lists
            strategies: List of tuples (strategy_name, sort_order) in priority order
            separate_unknown: Whether to separate albums with missing primary sort data
            
        Returns:
            Dictionary mapping album types to sorted results
            
        Raises:
            DataValidationError: If input validation fails
            InvalidStrategyError: If strategy names are invalid
        """
        if not isinstance(albums_by_type, dict):
            raise DataValidationError(f"Expected dictionary, got {type(albums_by_type)}")
        
        if not albums_by_type:
            logger.debug("Empty albums_by_type dictionary provided")
            return {}
        
        if not strategies:
            raise DataValidationError("At least one sorting strategy must be provided")
        
        result = {}
        error_collector = ErrorCollector()
        total_albums = sum(len(albums) for albums in albums_by_type.values())
        
        logger.info(
            f"Multi-sorting {total_albums} albums across {len(albums_by_type)} types "
            f"using {len(strategies)} strategies: {[s[0] for s in strategies]}"
        )
        
        for album_type, albums in albums_by_type.items():
            try:
                if not isinstance(albums, list):
                    error_collector.add_error(
                        DataValidationError(f"Albums for type '{album_type}' is not a list"),
                        {"album_type": album_type, "type": type(albums)}
                    )
                    result[album_type] = {"known": [], "unknown": []}
                    continue
                
                result[album_type] = self.sort_albums_multi(
                    albums, strategies, separate_unknown
                )
                
                # Log success for this type
                sorted_result = result[album_type]
                total_sorted = len(sorted_result["known"]) + len(sorted_result["unknown"])
                logger.debug(
                    f"Multi-sorted {total_sorted} {album_type}s with {len(strategies)} strategies "
                    f"(known: {len(sorted_result['known'])}, unknown: {len(sorted_result['unknown'])})"
                )
                
            except Exception as e:
                error_collector.add_error(e, {
                    "album_type": album_type,
                    "album_count": len(albums) if isinstance(albums, list) else "unknown",
                    "strategies": [s[0] for s in strategies]
                })
                
                if SORTING_ERROR_CONFIG["fallback_to_original_order"]:
                    logger.warning(f"Error multi-sorting {album_type} albums: {e}. Using original order.")
                    result[album_type] = {"known": albums if isinstance(albums, list) else [], "unknown": []}
                else:
                    raise SortingError(f"Failed to multi-sort {album_type} albums: {e}")
        
        # Log final summary
        if error_collector.has_errors():
            summary = error_collector.get_summary()
            logger.warning(
                f"Multi-strategy album type sorting completed with {summary['total_errors']} errors "
                f"across {len(albums_by_type)} album types"
            )
        else:
            logger.info(f"Successfully multi-sorted all {len(albums_by_type)} album types")
        
        return result


# Convenience functions for common use cases
@error_handler
def sort_albums_by_release_date(
    albums: List[Dict[str, Any]], 
    descending: bool = True,
    separate_unknown: bool = True,
    preserve_original_dates: bool = True
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Convenience function to sort albums by release date with enhanced parsing.
    
    Args:
        albums: List of album dictionaries
        descending: If True, sort newest first; if False, sort oldest first
        separate_unknown: Whether to separate albums with missing release dates
        preserve_original_dates: Whether to preserve original date formats in album data
        
    Returns:
        Dictionary with 'known' and 'unknown' keys containing sorted albums
        
    Raises:
        DataValidationError: If input validation fails
        SortingError: If sorting operation fails
    """
    if not isinstance(albums, list):
        raise DataValidationError(f"Expected list of albums, got {type(albums)}")
    
    logger.debug(f"Sorting {len(albums)} albums by release date (descending={descending})")
    
    try:
        sorter = AlbumSorter(preserve_original_dates=preserve_original_dates)
        order = SortOrder.DESCENDING if descending else SortOrder.ASCENDING
        return sorter.sort_albums(albums, "release_date", "album_id", order, separate_unknown)
    except Exception as e:
        logger.error(f"Failed to sort albums by release date: {e}")
        raise SortingError(f"Release date sorting failed: {e}") from e


@error_handler
def sort_albums_by_name(
    albums: List[Dict[str, Any]], 
    descending: bool = False
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Convenience function to sort albums alphabetically by name.
    
    Args:
        albums: List of album dictionaries  
        descending: If True, sort Z-A; if False, sort A-Z
        
    Returns:
        Dictionary with 'known' and 'unknown' keys containing sorted albums
        
    Raises:
        DataValidationError: If input validation fails
        SortingError: If sorting operation fails
    """
    if not isinstance(albums, list):
        raise DataValidationError(f"Expected list of albums, got {type(albums)}")
    
    logger.debug(f"Sorting {len(albums)} albums by name (descending={descending})")
    
    try:
        sorter = AlbumSorter()
        order = SortOrder.DESCENDING if descending else SortOrder.ASCENDING
        return sorter.sort_albums(albums, "album_name", "album_id", order, separate_unknown=False)
    except Exception as e:
        logger.error(f"Failed to sort albums by name: {e}")
        raise SortingError(f"Name sorting failed: {e}") from e


@error_handler
def sort_albums_by_popularity(
    albums: List[Dict[str, Any]], 
    descending: bool = True,
    separate_unknown: bool = True
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Convenience function to sort albums by popularity score.
    
    Args:
        albums: List of album dictionaries
        descending: If True, sort highest popularity first; if False, sort lowest first
        separate_unknown: Whether to separate albums with missing popularity
        
    Returns:
        Dictionary with 'known' and 'unknown' keys containing sorted albums
        
    Raises:
        DataValidationError: If input validation fails
        SortingError: If sorting operation fails
    """
    if not isinstance(albums, list):
        raise DataValidationError(f"Expected list of albums, got {type(albums)}")
    
    logger.debug(f"Sorting {len(albums)} albums by popularity (descending={descending})")
    
    try:
        sorter = AlbumSorter()
        order = SortOrder.DESCENDING if descending else SortOrder.ASCENDING
        return sorter.sort_albums(albums, "popularity", "album_id", order, separate_unknown)
    except Exception as e:
        logger.error(f"Failed to sort albums by popularity: {e}")
        raise SortingError(f"Popularity sorting failed: {e}") from e


@error_handler
def sort_albums_multi_criteria(
    albums: List[Dict[str, Any]],
    primary_sort: str = "release_date",
    secondary_sort: str = "album_name",
    tertiary_sort: str = "album_id",
    primary_desc: bool = True,
    secondary_desc: bool = False,
    tertiary_desc: bool = False,
    separate_unknown: bool = True
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Convenience function for multi-criteria sorting with up to 3 strategies.
    
    Args:
        albums: List of album dictionaries
        primary_sort: Primary sorting strategy name
        secondary_sort: Secondary sorting strategy name
        tertiary_sort: Tertiary sorting strategy name
        primary_desc: Whether primary sort should be descending
        secondary_desc: Whether secondary sort should be descending
        tertiary_desc: Whether tertiary sort should be descending
        separate_unknown: Whether to separate albums with missing primary sort data
        
    Returns:
        Dictionary with 'known' and 'unknown' keys containing sorted albums
        
    Raises:
        DataValidationError: If input validation fails
        InvalidStrategyError: If strategy names are invalid
        SortingError: If sorting operation fails
    """
    if not isinstance(albums, list):
        raise DataValidationError(f"Expected list of albums, got {type(albums)}")
    
    logger.debug(
        f"Multi-criteria sorting {len(albums)} albums: "
        f"{primary_sort} -> {secondary_sort} -> {tertiary_sort}"
    )
    
    try:
        sorter = AlbumSorter()
        strategies = [
            (primary_sort, SortOrder.DESCENDING if primary_desc else SortOrder.ASCENDING),
            (secondary_sort, SortOrder.DESCENDING if secondary_desc else SortOrder.ASCENDING),
            (tertiary_sort, SortOrder.DESCENDING if tertiary_desc else SortOrder.ASCENDING)
        ]
        return sorter.sort_albums_multi(albums, strategies, separate_unknown)
    except Exception as e:
        logger.error(f"Failed multi-criteria sorting: {e}")
        raise SortingError(f"Multi-criteria sorting failed: {e}") from e


def safe_sort_albums_with_recovery(
    albums: List[Dict[str, Any]],
    primary_strategy: str = "release_date",
    secondary_strategy: str = "album_id",
    order: SortOrder = SortOrder.DESCENDING,
    separate_unknown: bool = True,
    preserve_unknown_order: bool = True,
    fallback_to_original: bool = True
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Safely sort albums with comprehensive error recovery and graceful degradation.
    
    This function provides maximum resilience for sorting operations by:
    1. Attempting full sorting with error handling
    2. Falling back to simpler sorting strategies if needed
    3. Preserving original order as last resort
    4. Logging all errors and recovery attempts
    
    Args:
        albums: List of album dictionaries to sort
        primary_strategy: Name of primary sorting strategy
        secondary_strategy: Name of secondary sorting strategy
        order: Sort order (ascending or descending)
        separate_unknown: Whether to separate albums with missing primary sort data
        preserve_unknown_order: Whether to preserve original order for unknown albums
        fallback_to_original: Whether to fallback to original order on complete failure
        
    Returns:
        Dictionary with 'known' and 'unknown' keys containing sorted album lists
        
    Note:
        This function will never raise exceptions and always returns a valid result
    """
    if not isinstance(albums, list):
        logger.error(f"Invalid input: expected list, got {type(albums)}")
        return {"known": [], "unknown": []}
    
    if not albums:
        logger.debug("Empty album list provided")
        return {"known": [], "unknown": []}
    
    original_count = len(albums)
    logger.info(f"Starting safe sort for {original_count} albums with strategy {primary_strategy}")
    
    # Strategy 1: Try full sorting with all error handling
    try:
        logger.debug("Attempting full sorting with error handling")
        sorter = AlbumSorter(preserve_original_dates=True)
        result = sorter.sort_albums(
            albums, 
            primary_strategy, 
            secondary_strategy, 
            order, 
            separate_unknown,
            preserve_unknown_order
        )
        
        # Validate result
        result_count = len(result.get("known", [])) + len(result.get("unknown", []))
        if result_count == original_count:
            logger.info(f"Full sorting successful: {len(result['known'])} known, {len(result['unknown'])} unknown")
            return result
        else:
            logger.warning(f"Full sorting lost albums: {original_count} -> {result_count}")
            
    except Exception as e:
        logger.warning(f"Full sorting failed: {e}")
    
    # Strategy 2: Try simpler sorting with just primary strategy
    try:
        logger.debug("Attempting simplified sorting with primary strategy only")
        sorter = AlbumSorter(preserve_original_dates=False)  # Disable metadata to reduce complexity
        
        # Manually separate known/unknown and sort known albums only
        release_date_strategy = ReleaseDateSortingStrategy(preserve_original=False)
        known_albums = []
        unknown_albums = []
        
        for i, album in enumerate(albums):
            try:
                if isinstance(album, dict) and release_date_strategy.has_valid_release_date(album):
                    known_albums.append(album)
                else:
                    unknown_albums.append(album)
            except Exception as album_error:
                logger.debug(f"Error checking album {i}: {album_error}")
                unknown_albums.append(album)
        
        # Sort known albums only
        if known_albums:
            try:
                def safe_sort_key(album):
                    try:
                        sort_key = release_date_strategy.get_sort_key(album)
                        return sort_key if sort_key is not None else datetime(1900, 1, 1)
                    except Exception:
                        return datetime(1900, 1, 1)
                
                known_albums.sort(key=safe_sort_key, reverse=(order == SortOrder.DESCENDING))
                logger.info(f"Simplified sorting successful: {len(known_albums)} known, {len(unknown_albums)} unknown")
                
                if separate_unknown:
                    return {"known": known_albums, "unknown": unknown_albums}
                else:
                    return {"known": known_albums + unknown_albums, "unknown": []}
                    
            except Exception as sort_error:
                logger.warning(f"Simplified sorting failed: {sort_error}")
        else:
            logger.info("No albums with valid dates found, all albums marked as unknown")
            if separate_unknown:
                return {"known": [], "unknown": albums}
            else:
                return {"known": albums, "unknown": []}
    
    except Exception as e:
        logger.warning(f"Simplified sorting attempt failed: {e}")
    
    # Strategy 3: Try basic separation without sorting
    try:
        logger.debug("Attempting basic separation without sorting")
        known_albums = []
        unknown_albums = []
        
        for album in albums:
            try:
                if isinstance(album, dict):
                    release_date = album.get("release_date")
                    if release_date and isinstance(release_date, str) and release_date.strip():
                        # Basic validation - just check if it looks like a date
                        if len(release_date.strip()) >= 4 and release_date.strip()[:4].isdigit():
                            known_albums.append(album)
                        else:
                            unknown_albums.append(album)
                    else:
                        unknown_albums.append(album)
                else:
                    unknown_albums.append(album)
            except Exception:
                unknown_albums.append(album)
        
        logger.info(f"Basic separation successful: {len(known_albums)} known, {len(unknown_albums)} unknown")
        
        if separate_unknown:
            return {"known": known_albums, "unknown": unknown_albums}
        else:
            return {"known": known_albums + unknown_albums, "unknown": []}
            
    except Exception as e:
        logger.error(f"Basic separation failed: {e}")
    
    # Strategy 4: Last resort - return original order
    if fallback_to_original:
        logger.warning("All sorting strategies failed, returning original order")
        return {"known": albums, "unknown": []}
    else:
        logger.error("All sorting strategies failed and fallback disabled")
        return {"known": [], "unknown": albums}


def diagnose_sorting_issues(albums: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Diagnose potential issues with album data that might cause sorting problems.
    
    Args:
        albums: List of album dictionaries to diagnose
        
    Returns:
        Dictionary with diagnostic information about potential sorting issues
    """
    if not isinstance(albums, list):
        return {"error": f"Expected list, got {type(albums)}"}
    
    if not albums:
        return {"error": "Empty album list"}
    
    diagnosis = {
        "total_albums": len(albums),
        "data_type_issues": [],
        "missing_required_fields": [],
        "date_issues": [],
        "duplicate_ids": [],
        "memory_usage_mb": 0,
        "recommendations": []
    }
    
    try:
        # Check basic data types and structure
        non_dict_albums = 0
        missing_id_albums = 0
        seen_ids = set()
        duplicate_ids = []
        
        for i, album in enumerate(albums):
            try:
                if not isinstance(album, dict):
                    non_dict_albums += 1
                    diagnosis["data_type_issues"].append(f"Album {i}: not a dictionary")
                    continue
                
                # Check for required ID field
                album_id = album.get("id")
                if not album_id:
                    missing_id_albums += 1
                    diagnosis["missing_required_fields"].append(f"Album {i}: missing ID")
                elif album_id in seen_ids:
                    duplicate_ids.append(album_id)
                    diagnosis["duplicate_ids"].append(f"Album {i}: duplicate ID {album_id}")
                else:
                    seen_ids.add(album_id)
                
                # Check release date issues
                release_date = album.get("release_date")
                if release_date is not None:
                    if not isinstance(release_date, str):
                        diagnosis["date_issues"].append(f"Album {i} ({album_id}): release_date is not a string")
                    elif not release_date.strip():
                        diagnosis["date_issues"].append(f"Album {i} ({album_id}): empty release_date")
                    else:
                        # Try basic date parsing
                        try:
                            parse_spotify_date(release_date, strict=False)
                        except Exception as date_error:
                            diagnosis["date_issues"].append(f"Album {i} ({album_id}): invalid date '{release_date}' - {date_error}")
                
            except Exception as album_error:
                diagnosis["data_type_issues"].append(f"Album {i}: error processing - {album_error}")
        
        # Calculate approximate memory usage
        try:
            import sys
            diagnosis["memory_usage_mb"] = sys.getsizeof(albums) / (1024 * 1024)
        except Exception:
            diagnosis["memory_usage_mb"] = 0
        
        # Generate recommendations
        if non_dict_albums > 0:
            diagnosis["recommendations"].append(f"Fix {non_dict_albums} albums that are not dictionaries")
        
        if missing_id_albums > 0:
            diagnosis["recommendations"].append(f"Add IDs to {missing_id_albums} albums missing ID field")
        
        if len(duplicate_ids) > 0:
            diagnosis["recommendations"].append(f"Remove {len(duplicate_ids)} duplicate album IDs")
        
        if len(diagnosis["date_issues"]) > 0:
            diagnosis["recommendations"].append(f"Fix {len(diagnosis['date_issues'])} albums with date issues")
        
        if len(albums) > 1000:
            diagnosis["recommendations"].append("Consider sorting in smaller batches for better performance")
        
        if len(diagnosis["date_issues"]) > len(albums) * 0.5:
            diagnosis["recommendations"].append("High percentage of date issues - consider using fallback sorting")
        
        logger.info(f"Sorting diagnosis complete for {len(albums)} albums")
        
    except Exception as e:
        diagnosis["error"] = f"Diagnosis failed: {str(e)}"
        logger.error(f"Failed to diagnose sorting issues: {e}")
    
    return diagnosis