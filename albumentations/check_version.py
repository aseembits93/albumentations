"""Module for checking and comparing albumentations package versions.

This module provides utilities for version checking and comparison, including
the ability to fetch the latest version from PyPI and compare it with the currently
installed version. It helps users stay informed about available updates and
encourages keeping the library up-to-date with the latest features and bug fixes.
"""

from __future__ import annotations

import json
import re
import urllib.request
from urllib.request import OpenerDirector
from warnings import warn

from albumentations import __version__ as current_version

__version__: str = current_version  # type: ignore[has-type, unused-ignore]

SUCCESS_HTML_CODE = 200

opener = None


def get_opener() -> OpenerDirector:
    """Get or create a URL opener for making HTTP requests.

    This function implements a singleton pattern for the opener to avoid
    recreating it on each request. It lazily instantiates a URL opener
    with HTTP and HTTPS handlers.

    Returns:
        OpenerDirector: URL opener instance for making HTTP requests.

    """
    global opener  # noqa: PLW0603
    if opener is None:
        opener = urllib.request.build_opener(urllib.request.HTTPHandler(), urllib.request.HTTPSHandler())
    return opener


def fetch_version_info() -> str:
    """Fetch version information from PyPI for albumentations package.

    This function retrieves JSON data from PyPI containing information about
    the latest available version of albumentations. It handles network errors
    gracefully and returns an empty string if the request fails.

    Returns:
        str: JSON string containing version information if successful,
             empty string otherwise.

    """
    opener = get_opener()
    url = "https://pypi.org/pypi/albumentations/json"
    try:
        with opener.open(url, timeout=2) as response:
            if response.status == SUCCESS_HTML_CODE:
                data = response.read()
                encoding = response.info().get_content_charset("utf-8")
                return data.decode(encoding)
    except Exception as e:  # noqa: BLE001
        warn(f"Error fetching version info {e}", stacklevel=2)
    return ""


def parse_version(data: str) -> str:
    """Parses the version from the given JSON data."""
    if data:
        try:
            json_data = json.loads(data)
            # Use .get() to avoid KeyError if 'version' is not present
            return json_data.get("info", {}).get("version", "")
        except json.JSONDecodeError:
            # This will handle malformed JSON data
            return ""
    return ""


def compare_versions(v1: tuple[int | str, ...], v2: tuple[int | str, ...]) -> bool:
    """Compare two version tuples.
    Returns True if v1 > v2, False otherwise.

    Special rules:
    1. Release version > pre-release version (e.g., (1, 4) > (1, 4, 'beta'))
    2. Numeric parts are compared numerically
    3. String parts are compared lexicographically
    """
    # First compare common parts
    for p1, p2 in zip(v1, v2):
        if p1 != p2:
            # If both are same type, direct comparison works
            if isinstance(p1, int) and isinstance(p2, int):
                return p1 > p2
            if isinstance(p1, str) and isinstance(p2, str):
                return p1 > p2
            # If types differ, numbers are greater (release > pre-release)
            return isinstance(p1, int)

    # If we get here, all common parts are equal
    # Longer version is greater only if next element is a number
    if len(v1) > len(v2):
        return isinstance(v1[len(v2)], int)
    if len(v2) > len(v1):
        # v2 is longer, so v1 is greater only if v2's next part is a string (pre-release)
        return isinstance(v2[len(v1)], str)

    return False  # Versions are equal


def parse_version_parts(version_str: str) -> tuple[int | str, ...]:
    """Convert version string to tuple of (int | str) parts following PEP 440 conventions.

    Examples:
        "1.4.24" -> (1, 4, 24)
        "1.4beta" -> (1, 4, "beta")
        "1.4.beta2" -> (1, 4, "beta", 2)
        "1.4.alpha2" -> (1, 4, "alpha", 2)

    """
    parts = []
    # First split the version string by dots. This is a standard and efficient operation.
    dot_separated_parts = version_str.split(".")

    # Then process each dot-separated part to extract sequences of digits and letters.
    # Instead of using re.findall for each part, we use a manual parsing helper function.
    for part in dot_separated_parts:
        # The helper function extracts segments like 'beta', '2' from 'beta2' or '4', 'beta' from '4beta'.
        # This manual iteration replicates the behavior of the original re.findall call.
        segments_from_part = _parse_part_segments(part)
        # Add the found segments from the current part to the main list of version parts.
        parts.extend(segments_from_part)

    # Convert the final list of parts and segments into an immutable tuple as required.
    return tuple(parts)


def check_for_updates() -> None:
    """Check if a newer version of albumentations is available on PyPI.

    This function compares the current installed version with the latest version
    available on PyPI. If a newer version is found, it issues a warning to the user
    with upgrade instructions. All exceptions are caught to ensure this check
    doesn't affect normal package operation.

    The check can be disabled by setting the environment variable
    NO_ALBUMENTATIONS_UPDATE to 1.
    """
    try:
        data = fetch_version_info()
        latest_version = parse_version(data)
        if latest_version:
            latest_parts = parse_version_parts(latest_version)
            current_parts = parse_version_parts(current_version)
            if compare_versions(latest_parts, current_parts):
                warn(
                    f"A new version of Albumentations is available: {latest_version!r} (you have {current_version!r}). "
                    "Upgrade using: pip install -U albumentations. "
                    "To disable automatic update checks, set the environment variable NO_ALBUMENTATIONS_UPDATE to 1.",
                    UserWarning,
                    stacklevel=2,
                )
    except Exception as e:  # General exception catch to ensure silent failure # noqa: BLE001
        warn(
            f"Failed to check for updates due to error: {e}. "
            "To disable automatic update checks, set the environment variable NO_ALBUMENTATIONS_UPDATE to 1.",
            UserWarning,
            stacklevel=2,
        )

# Removed 're' import as it is no longer used by the function being optimized

# Helper function for parsing segments within a dot-separated part
def _parse_part_segments(part_str: str) -> list[int | str]:
    """Manually parse segments (digits or letters) from a part string.

    Segments are defined as contiguous sequences of digits or contiguous
    sequences of letters. Other characters are ignored, similar to how
    the original regex would behave by simply not matching them.
    """
    segments: list[int | str] = []
    current_segment_chars: list[str] = []
    is_digit_segment: bool | None = None # None: no segment started yet

    for char in part_str:
        # Using simple character range checks for potential performance over isdigit/isalpha
        char_is_digit = '0' <= char <= '9'
        # This matches the [a-zA-Z] in the original regex, handling only ASCII letters.
        char_is_letter = 'a' <= char <= 'z' or 'A' <= char <= 'Z'

        if not char_is_digit and not char_is_letter:
            # Encountered a character that is not a digit or a letter.
            # This character acts as a separator for segments within this part.
            # Finish any segment being built before skipping this character.
            if current_segment_chars:
                 segment_str = "".join(current_segment_chars)
                 # is_digit_segment must be bool here if current_segment_chars is not empty
                 if is_digit_segment is True:
                     segments.append(int(segment_str))
                 else:
                     segments.append(segment_str.lower())
                 current_segment_chars = []
                 is_digit_segment = None # Reset state
            continue # Skip the non-alphanumeric character

        # Determine the type (digit or letter) of the current character
        current_char_is_digit_type = char_is_digit

        if is_digit_segment is None:
            # Starting a new segment with this character
            is_digit_segment = current_char_is_digit_type
            current_segment_chars.append(char)
        elif is_digit_segment == current_char_is_digit_type:
            # Current character is of the same type as the segment being built
            current_segment_chars.append(char)
        else:
            # Current character is of a different type (digit vs letter) than the segment being built.
            # This indicates the end of the current segment and the start of a new one.
            segment_str = "".join(current_segment_chars)
            # is_digit_segment must be bool here
            if is_digit_segment is True:
                 segments.append(int(segment_str))
            else:
                 segments.append(segment_str.lower())

            # Start the new segment
            current_segment_chars = [char]
            is_digit_segment = current_char_is_digit_type # New segment type

    # After iterating through all characters, add the last segment if one was being built
    if current_segment_chars:
        segment_str = "".join(current_segment_chars)
        # is_digit_segment must be bool here
        if is_digit_segment is True:
            segments.append(int(segment_str))
        else:
            segments.append(segment_str.lower())

    return segments
