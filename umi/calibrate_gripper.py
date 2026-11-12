"""
Gripper Calibration Script

Interactive calibration tool to determine min/max gripper distances
by recording while the user opens and closes the gripper.

Uses Rerun for real-time visualization of:
- Camera feed with detected blobs overlaid
- HSV threshold mask
- Distance time series with min/max markers

Usage:
    python -m umi.calibrate_gripper --camera 0 --duration 10 --output gripper_calibration.yaml
"""

import argparse
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import rerun as rr
import yaml


@dataclass
class GripperCalibration:
    """Calibration data for gripper detection."""

    min_distance: float
    max_distance: float
    hsv_lower: Tuple[int, int, int]
    hsv_upper: Tuple[int, int, int]
    camera_width: int
    camera_height: int
    min_contour_area: int
    calibration_timestamp: str


class GripperCalibrator:
    """
    Calibration tool for gripper state detection.

    Records video while user opens/closes gripper, tracks orange blob
    distances, and saves calibration data.
    """

    # Default HSV range for orange (H: 5-25, S: 100-255, V: 100-255)
    DEFAULT_HSV_LOWER = (5, 100, 100)
    DEFAULT_HSV_UPPER = (25, 255, 255)
    DEFAULT_MIN_CONTOUR_AREA = 100

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        duration_seconds: float = 10.0,
        output_path: str = "gripper_calibration.yaml",
        hsv_lower: Optional[Tuple[int, int, int]] = None,
        hsv_upper: Optional[Tuple[int, int, int]] = None,
        min_contour_area: int = DEFAULT_MIN_CONTOUR_AREA,
        video_path: Optional[str] = None,
    ):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.duration_seconds = duration_seconds
        self.output_path = Path(output_path)
        self.video_path = Path(video_path) if video_path else None

        self.hsv_lower = hsv_lower or self.DEFAULT_HSV_LOWER
        self.hsv_upper = hsv_upper or self.DEFAULT_HSV_UPPER
        self.min_contour_area = min_contour_area

        # Pre-create morphological kernels
        self._erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # Pre-compute HSV bounds as numpy arrays
        self._hsv_lower_np = np.array(self.hsv_lower, dtype=np.uint8)
        self._hsv_upper_np = np.array(self.hsv_upper, dtype=np.uint8)

    def run_calibration(self) -> Optional[GripperCalibration]:
        """
        Main calibration loop with Rerun visualization.

        Returns:
            GripperCalibration if successful, None if failed
        """
        # Initialize Rerun
        rr.init("gripper_calibration", spawn=True)

        # Determine source type
        using_video = self.video_path is not None

        # Log static info
        source_info = f"Video: {self.video_path}" if using_video else f"Camera {self.camera_index}"
        rr.log(
            "instructions",
            rr.TextDocument(
                f"""# Gripper Calibration

**Source:** {source_info}

**Instructions:**
1. {'Watch video playback' if using_video else 'Position gripper in camera view'}
2. {'Video shows gripper opening/closing' if using_video else 'Slowly open and close gripper several times'}
3. {'Processing entire video' if using_video else f'Recording duration: {self.duration_seconds}s'}

**Orange Detection HSV Range:**
- Lower: {self.hsv_lower}
- Upper: {self.hsv_upper}

Press Ctrl+C to abort.
""",
                media_type=rr.MediaType.MARKDOWN,
            ),
            static=True,
        )

        # Open video source (camera or video file)
        if using_video:
            if not self.video_path.exists():
                print(f"Error: Video file not found: {self.video_path}")
                return None
            cap = cv2.VideoCapture(str(self.video_path))
            if not cap.isOpened():
                print(f"Error: Could not open video file: {self.video_path}")
                return None
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"Video opened: {self.video_path.name}")
            print(f"  Frames: {total_frames}, FPS: {video_fps:.1f}")
        else:
            cap = cv2.VideoCapture(self.camera_index)
            if not cap.isOpened():
                print(f"Error: Could not open camera {self.camera_index}")
                return None
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # Get actual resolution
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Resolution: {actual_width}x{actual_height}")

        # Tracking data
        all_distances: List[float] = []
        min_distance_so_far = float("inf")
        max_distance_so_far = 0.0

        start_time = time.time()
        frame_idx = 0

        try:
            if using_video:
                print(f"\nProcessing video ({total_frames} frames)...")
            else:
                print(f"\nRecording for {self.duration_seconds} seconds...")
                print("Open and close the gripper slowly and fully.\n")

            while True:
                # Check exit condition
                if using_video:
                    # Video mode: process until end of file
                    elapsed = frame_idx / video_fps if video_fps > 0 else 0
                else:
                    # Camera mode: process for duration_seconds
                    elapsed = time.time() - start_time
                    if elapsed >= self.duration_seconds:
                        break

                ret, frame_bgr = cap.read()
                if not ret:
                    if using_video:
                        # End of video file
                        print(f"\nEnd of video reached at frame {frame_idx}")
                        break
                    else:
                        print("Warning: Failed to read frame")
                        continue

                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                # Detect orange blobs
                blobs, mask = self._detect_orange_blobs(frame_rgb)
                distance = self._compute_tip_distance(blobs)

                # Update tracking
                if distance is not None:
                    all_distances.append(distance)
                    min_distance_so_far = min(min_distance_so_far, distance)
                    max_distance_so_far = max(max_distance_so_far, distance)

                # Visualize in Rerun
                self._visualize_frame(
                    frame_idx,
                    frame_rgb,
                    mask,
                    blobs,
                    distance,
                    min_distance_so_far,
                    max_distance_so_far,
                    elapsed,
                )

                frame_idx += 1

                # Print progress
                if frame_idx % 30 == 0:
                    if using_video:
                        progress = f"[{frame_idx}/{total_frames}]"
                    else:
                        progress = f"[{elapsed:.1f}s]"

                    if distance is not None:
                        print(
                            f"  {progress} Distance: {distance:.1f}px "
                            f"(min: {min_distance_so_far:.1f}, max: {max_distance_so_far:.1f})"
                        )
                    else:
                        print(
                            f"  {progress} No valid detection (blobs: {len(blobs)})"
                        )

        except KeyboardInterrupt:
            print("\nCalibration aborted by user.")
            return None
        finally:
            cap.release()

        # Validate results
        if len(all_distances) < 10:
            print(f"\nError: Not enough valid detections ({len(all_distances)})")
            print("Ensure gripper tips are visible and orange is detected.")
            return None

        if max_distance_so_far <= min_distance_so_far:
            print("\nError: min_distance >= max_distance")
            print("Ensure you fully opened AND closed the gripper.")
            return None

        # Create calibration
        calibration = GripperCalibration(
            min_distance=float(min_distance_so_far),
            max_distance=float(max_distance_so_far),
            hsv_lower=self.hsv_lower,
            hsv_upper=self.hsv_upper,
            camera_width=actual_width,
            camera_height=actual_height,
            min_contour_area=self.min_contour_area,
            calibration_timestamp=datetime.now().isoformat(),
        )

        print(f"\n Calibration complete!")
        print(f"  Min distance (closed): {calibration.min_distance:.1f}px")
        print(f"  Max distance (open):   {calibration.max_distance:.1f}px")
        print(f"  Valid frames: {len(all_distances)}")

        return calibration

    def _detect_orange_blobs(
        self, frame: np.ndarray
    ) -> Tuple[List[Tuple[float, float, float]], np.ndarray]:
        """
        Detect orange blobs in frame.

        Returns:
            Tuple of (list of (cx, cy, area), binary mask)
        """
        # Convert RGB to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

        # Threshold for orange color
        mask = cv2.inRange(hsv, self._hsv_lower_np, self._hsv_upper_np)

        # Morphological cleanup
        mask = cv2.erode(mask, self._erode_kernel, iterations=1)
        mask = cv2.dilate(mask, self._dilate_kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Extract blob centroids and areas
        blobs: List[Tuple[float, float, float]] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= self.min_contour_area:
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                    blobs.append((cx, cy, area))

        return blobs, mask

    def _compute_tip_distance(
        self, blobs: List[Tuple[float, float, float]]
    ) -> Optional[float]:
        """Compute distance between two largest blobs."""
        if len(blobs) < 2:
            return None

        # Sort by area and take two largest
        sorted_blobs = sorted(blobs, key=lambda b: b[2], reverse=True)[:2]
        (x1, y1, _), (x2, y2, _) = sorted_blobs

        return float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))

    def _visualize_frame(
        self,
        frame_idx: int,
        frame: np.ndarray,
        mask: np.ndarray,
        blobs: List[Tuple[float, float, float]],
        distance: Optional[float],
        min_dist: float,
        max_dist: float,
        elapsed: float,
    ):
        """Log frame data to Rerun for visualization."""
        rr.set_time("frame", sequence=frame_idx)

        # Log camera image
        rr.log("camera/image", rr.Image(frame))

        # Log HSV mask
        rr.log("camera/mask", rr.Image(mask))

        # Log detected blobs as points
        if blobs:
            # Sort by area to identify the two largest
            sorted_blobs = sorted(blobs, key=lambda b: b[2], reverse=True)
            positions = [[b[0], b[1]] for b in sorted_blobs]
            radii = [np.sqrt(b[2] / np.pi) for b in sorted_blobs]

            # Color: green for top 2 blobs, yellow for others
            colors = []
            for i in range(len(sorted_blobs)):
                if i < 2:
                    colors.append([0, 255, 0, 255])  # Green for main tips
                else:
                    colors.append([255, 255, 0, 255])  # Yellow for extras

            rr.log(
                "camera/blobs",
                rr.Points2D(
                    positions=positions,
                    radii=radii,
                    colors=colors,
                ),
            )

            # Draw line between the two largest blobs
            if len(sorted_blobs) >= 2:
                p1 = sorted_blobs[0][:2]
                p2 = sorted_blobs[1][:2]
                rr.log(
                    "camera/gripper_line",
                    rr.LineStrips2D(
                        strips=[[p1, p2]],
                        colors=[[0, 255, 0, 255]],
                        radii=[2.0],
                    ),
                )
        else:
            # Clear previous blobs
            rr.log("camera/blobs", rr.Clear(recursive=False))
            rr.log("camera/gripper_line", rr.Clear(recursive=False))

        # Log distance metrics
        if distance is not None:
            rr.log("metrics/distance", rr.Scalars(distance))

        if min_dist < float("inf"):
            rr.log("metrics/min_distance", rr.Scalars(min_dist))
        if max_dist > 0:
            rr.log("metrics/max_distance", rr.Scalars(max_dist))

        # Log status text
        remaining = max(0, self.duration_seconds - elapsed)
        status = f"""## Calibration Status

**Time remaining:** {remaining:.1f}s
**Blobs detected:** {len(blobs)}
**Current distance:** {f'{distance:.1f}px' if distance else 'N/A'}
**Min distance:** {f'{min_dist:.1f}px' if min_dist < float('inf') else 'N/A'}
**Max distance:** {f'{max_dist:.1f}px' if max_dist > 0 else 'N/A'}
"""
        rr.log("status", rr.TextDocument(status, media_type=rr.MediaType.MARKDOWN))

    def save_calibration(self, calibration: GripperCalibration) -> None:
        """Save calibration to YAML file."""
        data = {
            "min_distance": calibration.min_distance,
            "max_distance": calibration.max_distance,
            "hsv_lower": list(calibration.hsv_lower),
            "hsv_upper": list(calibration.hsv_upper),
            "camera_width": calibration.camera_width,
            "camera_height": calibration.camera_height,
            "min_contour_area": calibration.min_contour_area,
            "calibration_timestamp": calibration.calibration_timestamp,
        }

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        print(f"\nCalibration saved to: {self.output_path}")


def list_cameras(max_cameras: int = 10) -> List[int]:
    """List available camera indices."""
    available = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate gripper state detection by recording gripper movement"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config YAML file (reads camera settings from it)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Camera index (default: 0, or from config)",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to MP4 video file (use instead of camera)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Camera width (default: 640, or from config)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Camera height (default: 480, or from config)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Recording duration in seconds (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="gripper_calibration.yaml",
        help="Output YAML file path (default: gripper_calibration.yaml)",
    )
    parser.add_argument(
        "--hsv-lower",
        type=int,
        nargs=3,
        default=None,
        metavar=("H", "S", "V"),
        help="HSV lower bound (default: 5 100 100)",
    )
    parser.add_argument(
        "--hsv-upper",
        type=int,
        nargs=3,
        default=None,
        metavar=("H", "S", "V"),
        help="HSV upper bound (default: 25 255 255)",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=100,
        help="Minimum contour area in pixels (default: 100)",
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="List available cameras and exit",
    )

    args = parser.parse_args()

    if args.list_cameras:
        cameras = list_cameras()
        if cameras:
            print(f"Available cameras: {cameras}")
        else:
            print("No cameras found")
        return

    # Load config file if provided
    camera_index = args.camera
    width = args.width
    height = args.height

    if args.config:
        try:
            with open(args.config, "r") as f:
                config = yaml.safe_load(f) or {}
            print(f"Loaded config from: {args.config}")

            # Parse camera string from config (format: INDEX:NAME:WxH)
            camera_str = config.get("camera")
            if camera_str and camera_index is None:
                parts = camera_str.split(":")
                if len(parts) >= 1:
                    camera_index = int(parts[0])
                if len(parts) >= 3:
                    resolution = parts[2].lower().split("x")
                    if len(resolution) == 2:
                        if width is None:
                            width = int(resolution[0])
                        if height is None:
                            height = int(resolution[1])
                print(f"  Camera from config: index={camera_index}, {width}x{height}")
        except Exception as e:
            print(f"Warning: Could not load config file: {e}")

    # Apply defaults
    camera_index = camera_index if camera_index is not None else 0
    width = width if width is not None else 640
    height = height if height is not None else 480

    hsv_lower = tuple(args.hsv_lower) if args.hsv_lower else None
    hsv_upper = tuple(args.hsv_upper) if args.hsv_upper else None

    calibrator = GripperCalibrator(
        camera_index=camera_index,
        width=width,
        height=height,
        duration_seconds=args.duration,
        output_path=args.output,
        hsv_lower=hsv_lower,
        hsv_upper=hsv_upper,
        min_contour_area=args.min_area,
        video_path=args.video,
    )

    calibration = calibrator.run_calibration()
    if calibration:
        calibrator.save_calibration(calibration)


if __name__ == "__main__":
    main()
