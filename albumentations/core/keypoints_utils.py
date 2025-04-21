"""Module for handling keypoint operations during augmentation.

This module provides utilities for working with keypoints in various formats during
the augmentation process. It includes functions for converting between coordinate systems,
filtering keypoints based on visibility, validating keypoint data, and applying
transformations to keypoints. The module supports different keypoint formats including
xy, yx, and those with additional angle or size information.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np

from albumentations.core.type_definitions import NUM_KEYPOINTS_COLUMNS_IN_ALBUMENTATIONS

from .utils import DataProcessor, Params, ShapeType

__all__ = [
    "KeypointParams",
    "KeypointsProcessor",
    "angle_to_2pi_range",
    "check_keypoints",
    "convert_keypoints_from_albumentations",
    "convert_keypoints_to_albumentations",
    "filter_keypoints",
]

keypoint_formats = {"xy", "yx", "xya", "xys", "xyas", "xysa", "xyz"}


def angle_to_2pi_range(angles: np.ndarray) -> np.ndarray:
    """Convert angles to the range [0, 2π).

    This function takes an array of angles and ensures they are all within
    the range of 0 to 2π (exclusive) by applying modulo 2π.

    Args:
        angles (np.ndarray): Array of angle values in radians.

    Returns:
        np.ndarray: Array of the same shape as input with angles normalized to [0, 2π).

    """
    # np.mod is efficient for arrays, no significant optimization needed here.
    return np.mod(angles, 2 * np.pi)


class KeypointParams(Params):
    """Parameters of keypoints

    Args:
        format (str): format of keypoints. Should be 'xy', 'yx', 'xya', 'xys', 'xyas', 'xysa', 'xyz'.

            x - X coordinate,

            y - Y coordinate

            z - Z coordinate (for 3D keypoints)

            s - Keypoint scale

            a - Keypoint orientation in radians or degrees (depending on KeypointParams.angle_in_degrees)

        label_fields (list[str]): list of fields that are joined with keypoints, e.g labels.
            Should be same type as keypoints.
        remove_invisible (bool): to remove invisible points after transform or not
        angle_in_degrees (bool): angle in degrees or radians in 'xya', 'xyas', 'xysa' keypoints
        check_each_transform (bool): if `True`, then keypoints will be checked after each dual transform.
            Default: `True`

    Note:
        The internal Albumentations format is [x, y, z, angle, scale]. For 2D formats (xy, yx, xya, xys, xyas, xysa),
        z coordinate is set to 0. For formats without angle or scale, these values are set to 0.

    """

    def __init__(
        self,
        format: str,  # noqa: A002
        label_fields: Sequence[str] | None = None,
        remove_invisible: bool = True,
        angle_in_degrees: bool = True,
        check_each_transform: bool = True,
    ):
        super().__init__(format, label_fields)
        self.remove_invisible = remove_invisible
        self.angle_in_degrees = angle_in_degrees
        self.check_each_transform = check_each_transform

    def to_dict_private(self) -> dict[str, Any]:
        """Get the private dictionary representation of keypoint parameters.

        Returns:
            dict[str, Any]: Dictionary containing the keypoint parameters.

        """
        data = super().to_dict_private()
        data.update(
            {
                "remove_invisible": self.remove_invisible,
                "angle_in_degrees": self.angle_in_degrees,
                "check_each_transform": self.check_each_transform,
            },
        )
        return data

    @classmethod
    def is_serializable(cls) -> bool:
        """Check if the keypoint parameters are serializable.

        Returns:
            bool: Always returns True as KeypointParams is serializable.

        """
        return True

    @classmethod
    def get_class_fullname(cls) -> str:
        """Get the full name of the class.

        Returns:
            str: The string "KeypointParams".

        """
        return "KeypointParams"

    def __repr__(self) -> str:
        return (
            f"KeypointParams(format={self.format}, label_fields={self.label_fields},"
            f" remove_invisible={self.remove_invisible}, angle_in_degrees={self.angle_in_degrees},"
            f" check_each_transform={self.check_each_transform})"
        )


class KeypointsProcessor(DataProcessor):
    """Processor for keypoint data transformation.

    This class handles the conversion, validation, and filtering of keypoints
    during transformations. It ensures keypoints are correctly formatted and
    processed according to the specified keypoint parameters.

    Args:
        params (KeypointParams): Parameters for keypoint processing.
        additional_targets (dict[str, str] | None): Dictionary mapping additional target names to their types.

    """

    def __init__(self, params: KeypointParams, additional_targets: dict[str, str] | None = None):
        super().__init__(params, additional_targets)

    @property
    def default_data_name(self) -> str:
        """Get the default name for keypoint data.

        Returns:
            str: The string "keypoints".

        """
        return "keypoints"

    def ensure_data_valid(self, data: dict[str, Any]) -> None:
        """Ensure the provided data dictionary contains all required label fields.

        Args:
            data (dict[str, Any]): The data dictionary to validate.

        Raises:
            ValueError: If any label field specified in params is missing from the data.

        """
        if self.params.label_fields and not all(i in data for i in self.params.label_fields):
            msg = "Your 'label_fields' are not valid - them must have same names as params in 'keypoint_params' dict"
            raise ValueError(msg)

    def filter(
        self,
        data: np.ndarray,
        shape: ShapeType,
    ) -> np.ndarray:
        """Filter keypoints based on visibility within given shape.

        Args:
            data (np.ndarray): Keypoints in [x, y, z, angle, scale] format
            shape (ShapeType): Shape to check against as {'height': height, 'width': width, 'depth': depth}

        Returns:
            np.ndarray: Filtered keypoints

        """
        self.params: KeypointParams
        return filter_keypoints(data, shape, remove_invisible=self.params.remove_invisible)

    def check(self, data: np.ndarray, shape: ShapeType) -> None:
        """Check if keypoints are valid within the given shape.

        Args:
            data (np.ndarray): Keypoints to validate.
            shape (ShapeType): Shape to check against.

        """
        check_keypoints(data, shape)

    def convert_from_albumentations(
        self,
        data: np.ndarray,
        shape: ShapeType,
    ) -> np.ndarray:
        """Convert keypoints from internal Albumentations format to the specified format.

        Args:
            data (np.ndarray): Keypoints in Albumentations format.
            shape (ShapeType): Shape information for validation.

        Returns:
            np.ndarray: Converted keypoints in the target format.

        """
        if not data.size:
            return data

        params = self.params
        return convert_keypoints_from_albumentations(
            data,
            params.format,
            shape,
            check_validity=params.remove_invisible,
            angle_in_degrees=params.angle_in_degrees,
        )

    def convert_to_albumentations(
        self,
        data: np.ndarray,
        shape: ShapeType,
    ) -> np.ndarray:
        """Convert keypoints from the specified format to internal Albumentations format.

        Args:
            data (np.ndarray): Keypoints in source format.
            shape (ShapeType): Shape information for validation.

        Returns:
            np.ndarray: Converted keypoints in Albumentations format.

        """
        if not data.size:
            return data
        params = self.params
        return convert_keypoints_to_albumentations(
            data,
            params.format,
            shape,
            check_validity=params.remove_invisible,
            angle_in_degrees=params.angle_in_degrees,
        )


def check_keypoints(keypoints: np.ndarray, shape: ShapeType) -> None:
    """Check if keypoint coordinates are within valid ranges for the given shape.

    This function validates that:
    1. All x-coordinates are within [0, width)
    2. All y-coordinates are within [0, height)
    3. If depth is provided in shape, z-coordinates are within [0, depth)
    4. Angles are within the range [0, 2π)

    Args:
        keypoints (np.ndarray): Array of keypoints with shape (N, 5+), where N is the number of keypoints.
            - First 2 columns are always x, y
            - Column 3 (if present) is z
            - Column 4 (if present) is angle
            - Column 5+ (if present) are additional attributes
        shape (ShapeType): The shape of the image/volume:
                           - For 2D: {'height': int, 'width': int}
                           - For 3D: {'height': int, 'width': int, 'depth': int}

    Raises:
        ValueError: If any keypoint coordinate is outside the valid range, or if angles are invalid.
                   The error message will detail which keypoints are invalid and why.

    Note:
        - The function assumes that keypoint coordinates are in absolute pixel values, not normalized
        - Angles are in radians
        - Z-coordinates are only checked if 'depth' is present in shape

    """
    height, width = shape["height"], shape["width"]
    has_depth = "depth" in shape
    num_cols = keypoints.shape[1]

    error_messages = []

    # Use boolean masks to find invalid keypoints efficiently in a vectorized manner.
    invalid_x_mask = (keypoints[:, 0] < 0) | (keypoints[:, 0] >= width)
    invalid_y_mask = (keypoints[:, 1] < 0) | (keypoints[:, 1] >= height)

    overall_invalid_mask = invalid_x_mask | invalid_y_mask

    invalid_z_mask = None
    if has_depth and num_cols > 2:
        z = keypoints[:, 2]
        depth = shape["depth"]
        invalid_z_mask = (z < 0) | (z >= depth)
        overall_invalid_mask |= invalid_z_mask

    invalid_angle_mask = None
    if num_cols > 3:
        angles = keypoints[:, 3]
        invalid_angle_mask = (angles < 0) | (angles >= 2 * math.pi)
        overall_invalid_mask |= invalid_angle_mask

    # Find the indices of all keypoints that are invalid for *any* reason.
    # This avoids repeated calls to np.where and set operations used in the original code.
    if np.any(overall_invalid_mask):
        invalid_indices = np.where(overall_invalid_mask)[0]
        # Sort indices for consistent error message order
        invalid_indices.sort()

        # Iterate through the identified invalid keypoints and generate specific messages.
        # Checking the pre-computed masks is faster than re-evaluating conditions or searching sets.
        for idx in invalid_indices:
            if invalid_x_mask[idx]:
                error_messages.append(
                    f"Expected x for keypoint {keypoints[idx]} to be in range [0, {width}), got {keypoints[idx, 0]}",
                )
            if invalid_y_mask[idx]:
                error_messages.append(
                    f"Expected y for keypoint {keypoints[idx]} to be in range [0, {height}), got {keypoints[idx, 1]}",
                )
            if invalid_z_mask is not None and invalid_z_mask[idx]:
                 error_messages.append(
                    f"Expected z for keypoint {keypoints[idx]} to be in range [0, {depth}), got {keypoints[idx, 2]}"
                 )
            if invalid_angle_mask is not None and invalid_angle_mask[idx]:
                 error_messages.append(
                    f"Expected angle for keypoint {keypoints[idx]} to be in range [0, 2π), got {keypoints[idx, 3]}"
                 )


    if error_messages:
        raise ValueError("\n".join(error_messages))


def filter_keypoints(
    keypoints: np.ndarray,
    shape: ShapeType,
    remove_invisible: bool,
) -> np.ndarray:
    """Filter keypoints to remove those outside the boundaries.

    Args:
        keypoints (np.ndarray): A numpy array of shape (N, 5+) where N is the number of keypoints.
                               Each row represents a keypoint (x, y, z, angle, scale, ...).
        shape (ShapeType): Shape to check against as {'height': height, 'width': width, 'depth': depth}.
        remove_invisible (bool): If True, remove keypoints outside the boundaries.

    Returns:
        np.ndarray: Filtered keypoints.

    """
    if not remove_invisible:
        return keypoints

    if not keypoints.size:
        return keypoints

    height, width, depth = shape["height"], shape["width"], shape.get("depth", None)

    # Create boolean mask for visible keypoints
    x, y, z = keypoints[:, 0], keypoints[:, 1], keypoints[:, 2]
    visible = (x >= 0) & (x < width) & (y >= 0) & (y < height)

    if depth is not None:
        visible &= (z >= 0) & (z < depth)

    # Apply the mask to filter keypoints
    return keypoints[visible]


def convert_keypoints_to_albumentations(
    keypoints: np.ndarray,
    source_format: Literal["xy", "yx", "xya", "xys", "xyas", "xysa", "xyz"],
    shape: ShapeType,
    check_validity: bool = False,
    angle_in_degrees: bool = True,
) -> np.ndarray:
    """Convert keypoints from various formats to the Albumentations format.

    This function takes keypoints in different formats and converts them to the standard
    Albumentations format: [x, y, z, angle, scale]. For 2D formats, z is set to 0.
    For formats without angle or scale, these values are set to 0.

    Args:
        keypoints (np.ndarray): Array of keypoints with shape (N, 2+), where N is the number of keypoints.
                                The number of columns depends on the source_format.
        source_format (Literal["xy", "yx", "xya", "xys", "xyas", "xysa", "xyz"]): The format of the input keypoints.
            - "xy": [x, y]
            - "yx": [y, x]
            - "xya": [x, y, angle]
            - "xys": [x, y, scale]
            - "xyas": [x, y, angle, scale]
            - "xysa": [x, y, scale, angle]
            - "xyz": [x, y, z]
        shape (ShapeType): The shape of the image {'height': height, 'width': width, 'depth': depth}.
        check_validity (bool, optional): If True, check if the converted keypoints are within the image boundaries.
                                         Defaults to False.
        angle_in_degrees (bool, optional): If True, convert input angles from degrees to radians.
                                           Defaults to True.

    Returns:
        np.ndarray: Array of keypoints in Albumentations format [x, y, z, angle, scale] with shape (N, 5+).
                    Any additional columns from the input keypoints are preserved and appended after the
                    first 5 columns.

    Raises:
        ValueError: If the source_format is not one of the supported formats.

    Note:
        - For 2D formats (xy, yx, xya, xys, xyas, xysa), z coordinate is set to 0
        - Angles are converted to the range [0, 2π) radians
        - If the input keypoints have additional columns beyond what's specified in the source_format,
          these columns are preserved in the output

    """
    if source_format not in keypoint_formats:
        raise ValueError(f"Unknown source_format {source_format}. Supported formats are: {keypoint_formats}")

    # Mapping from Albumentations format column index (0-4 for x, y, z, angle, scale)
    # to the input keypoints column index, or None if not present in the source format.
    # Note: NUM_KEYPOINTS_COLUMNS_IN_ALBUMENTATIONS is 5 (x, y, z, angle, scale).
    format_to_albumentations_indices: dict[str, list[int | None]] = {
        "xy": [0, 1, None, None, None],
        "yx": [1, 0, None, None, None],
        "xya": [0, 1, None, 2, None],
        "xys": [0, 1, None, None, 2],
        "xyas": [0, 1, None, 2, 3],
        "xysa": [0, 1, None, 3, 2],
        "xyz": [0, 1, 2, None, None],
    }

    input_indices_map: list[int | None] = format_to_albumentations_indices[source_format]
    num_input_cols = len(source_format) # This is correct based on how additional columns are handled later.
    N = keypoints.shape[0]

    # Create the base array for the 5 Albumentations columns. Initialize with zeros.
    processed_keypoints_base = np.zeros((N, NUM_KEYPOINTS_COLUMNS_IN_ALBUMENTATIONS), dtype=np.float32)

    # Use vectorized assignment to populate the base columns.
    # This replaces the explicit Python loop over columns in the original implementation.
    # x (col 0)
    processed_keypoints_base[:, 0] = keypoints[:, input_indices_map[0]]
    # y (col 1)
    processed_keypoints_base[:, 1] = keypoints[:, input_indices_map[1]]

    # z (col 2) - default is 0 from np.zeros
    if input_indices_map[2] is not None:
        processed_keypoints_base[:, 2] = keypoints[:, input_indices_map[2]]

    # angle (col 3) - default is 0 from np.zeros
    if input_indices_map[3] is not None:
        angle_col = keypoints[:, input_indices_map[3]]
        if angle_in_degrees:
            angle_col = np.radians(angle_col)
        processed_keypoints_base[:, 3] = angle_to_2pi_range(angle_col)

    # scale (col 4) - default is 0 from np.zeros
    if input_indices_map[4] is not None:
        processed_keypoints_base[:, 4] = keypoints[:, input_indices_map[4]]

    # Append any additional columns present in the input that are not part of the source_format.
    if keypoints.shape[1] > num_input_cols:
        additional_cols = keypoints[:, num_input_cols:]
        processed_keypoints = np.column_stack((processed_keypoints_base, additional_cols))
    else:
        processed_keypoints = processed_keypoints_base

    if check_validity:
        # This call benefits from the optimization made to check_keypoints
        check_keypoints(processed_keypoints, shape)

    return processed_keypoints


def convert_keypoints_from_albumentations(
    keypoints: np.ndarray,
    target_format: Literal["xy", "yx", "xya", "xys", "xyas", "xysa", "xyz"],
    shape: ShapeType,
    check_validity: bool = False,
    angle_in_degrees: bool = True,
) -> np.ndarray:
    """Convert keypoints from Albumentations format to various other formats.

    This function takes keypoints in the standard Albumentations format [x, y, z, angle, scale]
    and converts them to the specified target format.

    Args:
        keypoints (np.ndarray): Array of keypoints in Albumentations format with shape (N, 5+),
                                where N is the number of keypoints. Each row represents a keypoint
                                [x, y, z, angle, scale, ...].
        target_format (Literal["xy", "yx", "xya", "xys", "xyas", "xysa", "xyz"]): The desired output format.
            - "xy": [x, y]
            - "yx": [y, x]
            - "xya": [x, y, angle]
            - "xys": [x, y, scale]
            - "xyas": [x, y, angle, scale]
            - "xysa": [x, y, scale, angle]
            - "xyz": [x, y, z]
        shape (ShapeType): The shape of the image {'height': height, 'width': width, 'depth': depth}.
        check_validity (bool, optional): If True, check if the keypoints are within the image boundaries.
                                         Defaults to False.
        angle_in_degrees (bool, optional): If True, convert output angles to degrees.
                                           If False, angles remain in radians.
                                           Defaults to True.

    Returns:
        np.ndarray: Array of keypoints in the specified target format with shape (N, 2+).
                    Any additional columns from the input keypoints beyond the first 5
                    are preserved and appended after the converted columns.

    Raises:
        ValueError: If the target_format is not one of the supported formats.

    Note:
        - Input angles are assumed to be in the range [0, 2π) radians
        - If the input keypoints have additional columns beyond the first 5,
          these columns are preserved in the output

    """
    if target_format not in keypoint_formats:
        raise ValueError(f"Unknown target_format {target_format}. Supported formats are: {keypoint_formats}")

    x, y, z, angle, scale = keypoints[:, 0], keypoints[:, 1], keypoints[:, 2], keypoints[:, 3], keypoints[:, 4]
    angle = angle_to_2pi_range(angle)

    if check_validity:
        check_keypoints(np.column_stack((x, y, z, angle, scale)), shape)

    if angle_in_degrees:
        angle = np.degrees(angle)

    format_to_columns = {
        "xy": [x, y],
        "yx": [y, x],
        "xya": [x, y, angle],
        "xys": [x, y, scale],
        "xyas": [x, y, angle, scale],
        "xysa": [x, y, scale, angle],
        "xyz": [x, y, z],
    }

    result = np.column_stack(format_to_columns[target_format])

    # Add any additional columns from the original keypoints
    if keypoints.shape[1] > NUM_KEYPOINTS_COLUMNS_IN_ALBUMENTATIONS:
        return np.column_stack((result, keypoints[:, NUM_KEYPOINTS_COLUMNS_IN_ALBUMENTATIONS:]))

    return result
