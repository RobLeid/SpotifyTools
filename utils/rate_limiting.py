"""
Rate limiting and exponential backoff utilities for the Spotify ISRC Finder application.
"""

import time
import random
import threading
from typing import Optional, Dict, Callable, Any
import streamlit as st
from .constants import (
    INITIAL_BACKOFF_DELAY,
    MAX_BACKOFF_DELAY,
    BACKOFF_MULTIPLIER,
    JITTER_RANGE,
    RATE_LIMIT_DELAY,
)


class RateLimitExceeded(Exception):
    """Custom exception for when rate limits are exceeded."""

    pass


class ExponentialBackoff:
    """
    Implements exponential backoff with jitter for retry logic.
    """

    def __init__(
        self,
        initial_delay: float = INITIAL_BACKOFF_DELAY,
        max_delay: float = MAX_BACKOFF_DELAY,
        multiplier: float = BACKOFF_MULTIPLIER,
        jitter: float = JITTER_RANGE,
    ):
        """
        Initialize exponential backoff configuration.

        Args:
            initial_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
            multiplier: Backoff multiplier
            jitter: Maximum jitter range in seconds
        """
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.multiplier = multiplier
        self.jitter = jitter
        self.current_delay = initial_delay

    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for a given attempt with jitter.

        Args:
            attempt: Current attempt number (0-based)

        Returns:
            Delay in seconds
        """
        # Calculate exponential delay
        delay = self.initial_delay * (self.multiplier**attempt)

        # Cap at maximum delay
        delay = min(delay, self.max_delay)

        # Add jitter to prevent thundering herd
        jitter = random.uniform(0, self.jitter)

        return delay + jitter

    def wait(self, attempt: int) -> None:
        """
        Wait for the calculated delay.

        Args:
            attempt: Current attempt number (0-based)
        """
        delay = self.calculate_delay(attempt)
        time.sleep(delay)

    def reset(self) -> None:
        """Reset the backoff to initial state."""
        self.current_delay = self.initial_delay


class RateLimiter:
    """
    Simple rate limiter to prevent API abuse.
    """

    def __init__(self, min_interval: float = RATE_LIMIT_DELAY):
        """
        Initialize rate limiter.

        Args:
            min_interval: Minimum time between requests in seconds
        """
        self.min_interval = min_interval
        self.last_request_time = 0.0
        self.lock = threading.Lock()

    def wait_if_needed(self) -> None:
        """Wait if needed to respect rate limits."""
        with self.lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time

            if time_since_last < self.min_interval:
                sleep_time = self.min_interval - time_since_last
                time.sleep(sleep_time)

            self.last_request_time = time.time()


class RetryHandler:
    """
    Handles retry logic with exponential backoff and rate limiting.
    """

    def __init__(
        self,
        max_retries: int = 5,
        backoff: Optional[ExponentialBackoff] = None,
        rate_limiter: Optional[RateLimiter] = None,
    ):
        """
        Initialize retry handler.

        Args:
            max_retries: Maximum number of retry attempts
            backoff: Exponential backoff configuration
            rate_limiter: Rate limiter instance
        """
        self.max_retries = max_retries
        self.backoff = backoff or ExponentialBackoff()
        self.rate_limiter = rate_limiter or RateLimiter()

    def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function with retry logic.

        Args:
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Function result

        Raises:
            RateLimitExceeded: If max retries exceeded
        """
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                # Apply rate limiting
                self.rate_limiter.wait_if_needed()

                # Execute function
                result = func(*args, **kwargs)

                # Reset backoff on success
                self.backoff.reset()

                return result

            except Exception as e:
                last_exception = e

                # Check if it's a rate limit error
                if hasattr(e, "response") and hasattr(e.response, "status_code"):
                    if e.response.status_code == 429:
                        # Handle 429 rate limit
                        retry_after = e.response.headers.get("Retry-After")
                        if retry_after:
                            wait_time = min(int(retry_after), MAX_BACKOFF_DELAY)
                            st.warning(
                                f"⏱️ Rate limit hit. Waiting {wait_time} seconds... (attempt {attempt + 1}/{self.max_retries})"
                            )
                            time.sleep(wait_time)
                        else:
                            st.warning(
                                f"⏱️ Rate limit hit. Using exponential backoff... (attempt {attempt + 1}/{self.max_retries})"
                            )
                            self.backoff.wait(attempt)
                        continue

                # For other errors, use exponential backoff
                if attempt < self.max_retries - 1:
                    delay = self.backoff.calculate_delay(attempt)
                    st.warning(
                        f"⏱️ Request failed, retrying in {delay:.1f}s... (attempt {attempt + 1}/{self.max_retries})"
                    )
                    self.backoff.wait(attempt)
                else:
                    # Last attempt failed
                    break

        # All attempts failed
        raise RateLimitExceeded(
            f"Max retries ({self.max_retries}) exceeded. Last error: {last_exception}"
        )


def handle_spotify_response_errors(response) -> None:
    """
    Handle common Spotify API response errors.

    Args:
        response: HTTP response object

    Raises:
        Various exceptions based on response status
    """
    if response.status_code == 200:
        return

    elif response.status_code == 429:
        # Rate limit exceeded
        retry_after = response.headers.get("Retry-After", "60")
        raise RateLimitExceeded(
            f"Rate limit exceeded. Retry after {retry_after} seconds"
        )

    elif response.status_code == 401:
        raise Exception("Authentication failed. Please check your access token.")

    elif response.status_code == 404:
        raise Exception("Resource not found.")

    elif response.status_code == 403:
        raise Exception("Forbidden. Check your permissions.")

    elif response.status_code >= 500:
        raise Exception(f"Server error: {response.status_code}")

    else:
        raise Exception(f"HTTP error: {response.status_code}")


# Global instances for reuse
default_retry_handler = RetryHandler()
default_rate_limiter = RateLimiter()
