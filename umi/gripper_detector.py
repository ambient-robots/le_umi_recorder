"""
Gripper State Detection Module

.. deprecated::
    This module is deprecated in favor of the new pipeline-based gripper
    detection using Google Gemini API. The new approach provides:
    - Semantic understanding of gripper state (not just color detection)
    - Both "measured" (actual) and "commanded" (intended) gripper values
    - Better handling of occlusion and lighting variations

    Use `umi.pipeline.stage4_gripper.GripperDetector` instead:

        from umi.pipeline import GripperDetector
        detector = GripperDetector(Stage4Config())
        await detector.process_episode(stage1_dir, output_dir)

    This legacy module is kept for backward compatibility with existing
    calibration files but will be removed in a future version.

Fast vision-based gripper state detection using HSV color thresholding
to track orange gripper tips.

Usage:
    detector = GripperDetector("gripper_calibration.yaml")
    result = detector.get_gripper_state(frame)
    print(f"Gripper state: {result.state:.2f}")  # 0=open, 1=closed
"""

import warnings

warnings.warn(
    "umi.gripper_detector is deprecated. Use umi.pipeline.stage4_gripper instead.",
    DeprecationWarning,
    stacklevel=2,
)

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import yaml


@dataclass
class GripperDetectionResult:
    """Result from gripper state detection."""

    state: float  # 0.0 (open) to 1.0 (closed)
    raw_distance: Optional[float]  # Distance in pixels (None if detection failed)
    blob_count: int  # Number of orange blobs detected
    confidence: float  # Detection confidence (0-1)


class GripperDetector:
    """
    Fast gripper state detection using HSV color thresholding.

    Detects orange gripper tips and calculates normalized state (0=open, 1=closed).

    Args:
        calibration_path: Path to gripper_calibration.yaml
        roi: Optional region of interest (x, y, width, height) for performance
        downscale_factor: Factor to downscale image (1.0 = no downscale, 2.0 = half size)
    """

    def __init__(
        self,
        calibration_path: str,
        roi: Optional[Tuple[int, int, int, int]] = None,
        downscale_factor: float = 1.0,
    ):
        self.calibration = self._load_calibration(calibration_path)
        self.roi = roi
        self.downscale_factor = downscale_factor

        # Pre-compute HSV bounds as numpy arrays for faster comparison
        self._hsv_lower = np.array(self.calibration["hsv_lower"], dtype=np.uint8)
        self._hsv_upper = np.array(self.calibration["hsv_upper"], dtype=np.uint8)

        # Pre-create morphological kernels
        self._erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # Calibration values
        self._min_distance = float(self.calibration["min_distance"])
        self._max_distance = float(self.calibration["max_distance"])
        self._min_contour_area = int(self.calibration.get("min_contour_area", 100))

        # Cache for last valid state (for smoothing/fallback)
        self._last_valid_state: float = 0.5

    def _load_calibration(self, path: str) -> dict:
        """Load calibration from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Calibration file not found: {path}")

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        # Validate required fields
        required = ["min_distance", "max_distance", "hsv_lower", "hsv_upper"]
        for field in required:
            if field not in data:
                raise ValueError(f"Calibration missing required field: {field}")

        return data

    def get_gripper_state(self, frame: np.ndarray) -> GripperDetectionResult:
        """
        Detect gripper state from camera frame.

        Args:
            frame: RGB image as numpy array (H, W, 3)

        Returns:
            GripperDetectionResult with state (0=open, 1=closed)
        """
        working_frame = frame

        # Apply ROI if specified
        if self.roi is not None:
            x, y, w, h = self.roi
            working_frame = working_frame[y : y + h, x : x + w]

        # Downscale for performance if specified
        if self.downscale_factor > 1.0:
            new_size = (
                int(working_frame.shape[1] / self.downscale_factor),
                int(working_frame.shape[0] / self.downscale_factor),
            )
            working_frame = cv2.resize(
                working_frame, new_size, interpolation=cv2.INTER_AREA
            )

        # Detect blobs and compute state
        blobs = self._detect_blobs(working_frame)
        distance = self._compute_distance(blobs)

        if distance is None:
            # Detection failed - return cached state with low confidence
            return GripperDetectionResult(
                state=self._last_valid_state,
                raw_distance=None,
                blob_count=len(blobs),
                confidence=0.0,
            )

        # Adjust distance for downscaling
        if self.downscale_factor > 1.0:
            distance *= self.downscale_factor

        # Normalize distance to 0-1 state
        # min_distance = closed (state=1), max_distance = open (state=0)
        normalized = (distance - self._min_distance) / (
            self._max_distance - self._min_distance
        )
        state = 1.0 - np.clip(normalized, 0.0, 1.0)

        # Compute confidence based on blob quality
        confidence = self._compute_confidence(blobs)

        # Cache valid state
        self._last_valid_state = float(state)

        return GripperDetectionResult(
            state=float(state),
            raw_distance=float(distance),
            blob_count=len(blobs),
            confidence=confidence,
        )

    def _detect_blobs(
        self, frame: np.ndarray
    ) -> List[Tuple[float, float, float]]:
        """
        Detect orange blobs in frame.

        Returns:
            List of (centroid_x, centroid_y, area) tuples
        """
        # Convert RGB to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

        # Threshold for orange color
        mask = cv2.inRange(hsv, self._hsv_lower, self._hsv_upper)

        # Morphological cleanup
        mask = cv2.erode(mask, self._erode_kernel, iterations=1)
        mask = cv2.dilate(mask, self._dilate_kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Extract blob centroids and areas
        blobs: List[Tuple[float, float, float]] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= self._min_contour_area:
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                    blobs.append((cx, cy, area))

        return blobs

    def _compute_distance(
        self, blobs: List[Tuple[float, float, float]]
    ) -> Optional[float]:
        """Compute distance between two largest blobs."""
        if len(blobs) < 2:
            return None

        # Sort by area and take two largest
        sorted_blobs = sorted(blobs, key=lambda b: b[2], reverse=True)[:2]
        (x1, y1, _), (x2, y2, _) = sorted_blobs

        return float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))

    def _compute_confidence(
        self, blobs: List[Tuple[float, float, float]]
    ) -> float:
        """Compute detection confidence (0-1)."""
        if len(blobs) < 2:
            return 0.0
        if len(blobs) == 2:
            return 1.0
        # More than 2 blobs reduces confidence
        return max(0.5, 1.0 - (len(blobs) - 2) * 0.1)

    def get_debug_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Get the HSV threshold mask for debugging.

        Args:
            frame: RGB image as numpy array (H, W, 3)

        Returns:
            Binary mask as numpy array (H, W)
        """
        working_frame = frame

        if self.roi is not None:
            x, y, w, h = self.roi
            working_frame = working_frame[y : y + h, x : x + w]

        if self.downscale_factor > 1.0:
            new_size = (
                int(working_frame.shape[1] / self.downscale_factor),
                int(working_frame.shape[0] / self.downscale_factor),
            )
            working_frame = cv2.resize(
                working_frame, new_size, interpolation=cv2.INTER_AREA
            )

        hsv = cv2.cvtColor(working_frame, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, self._hsv_lower, self._hsv_upper)
        mask = cv2.erode(mask, self._erode_kernel, iterations=1)
        mask = cv2.dilate(mask, self._dilate_kernel, iterations=1)

        return mask
