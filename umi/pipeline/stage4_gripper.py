"""
Stage 4: Gripper Width Detection

Uses Gemini API to semantically analyze gripper states throughout the video,
detecting both measured (actual) and commanded (intended) gripper openings.

Input:
    - Stage 1 output: raw_video.mp4, video_timestamps.parquet

Output:
    - gripper_states.parquet: Gripper states per frame
    - metadata.json: Processing metadata
"""

import argparse
import asyncio
import json
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import pyarrow.parquet as pq
import rerun as rr

from .schemas import (
    Stage4Config,
    StageMetadata,
    GripperState,
    gripper_states_to_table,
    PipelineConfig,
)
from .gemini_client import (
    GeminiClient,
    get_gripper_prompt,
    GENAI_AVAILABLE,
)


class GripperDetector:
    """
    Detects gripper states from video using Gemini's semantic understanding.

    Outputs two values per frame:
    - measured_state (0-100): Actual physical gripper opening observed
    - commanded_state (0-100): Intended/target gripper opening

    Example scenario:
    - Robot attempts to grasp object
    - commanded=0 (fully close), measured=30 (object prevents full close)
    """

    def __init__(self, config: Stage4Config):
        """
        Initialize gripper detector.

        Args:
            config: Stage 4 configuration
        """
        self.config = config
        self.client: Optional[GeminiClient] = None
        self._rerun_initialized = False

    def _setup_rerun(self):
        """Initialize Rerun visualization."""
        rr.init("stage4_gripper_detector", spawn=True)
        rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        self._rerun_initialized = True
        print("Rerun visualization started")

    def _visualize_gripper_results(
        self,
        gripper_states: List[GripperState],
        video_path: Path,
    ):
        """Visualize gripper detection results with video playback."""
        if not self._rerun_initialized:
            return

        # Open video for frame extraction
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Warning: Could not open video for visualization: {video_path}")
            return

        total_frames = len(gripper_states)
        fps = cap.get(cv2.CAP_PROP_FPS) or 60

        print(f"  Visualizing {total_frames} frames...")

        for i, state in enumerate(gripper_states):
            # Set Rerun time
            rr.set_time_sequence("frame", i)
            rr.set_time_seconds("time", i / fps)

            # Read video frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret and frame is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rr.log("video/frame", rr.Image(frame_rgb))

            # Log gripper states
            rr.log("gripper/measured", rr.Scalar(state.measured_state))
            rr.log("gripper/commanded", rr.Scalar(state.commanded_state))
            rr.log("gripper/confidence", rr.Scalar(state.confidence))

            # Log annotation as text
            if state.annotation:
                rr.log("gripper/annotation", rr.TextLog(state.annotation))

            # Log difference (commanded - measured)
            diff = state.commanded_state - state.measured_state
            rr.log("gripper/difference", rr.Scalar(diff))

            if (i + 1) % 100 == 0:
                print(f"    Visualized {i + 1}/{total_frames} frames")

        cap.release()
        print(f"  Visualization complete")

    async def setup(self):
        """Initialize Gemini client."""
        if not GENAI_AVAILABLE:
            raise ImportError(
                "google-generativeai package not installed. "
                "Install with: pip install google-generativeai"
            )

        self.client = GeminiClient(
            model=self.config.model,
            api_key_env=self.config.api_key_env,
        )
        print(f"Gemini client initialized with model: {self.config.model}")

    async def detect_gripper_states(
        self,
        video_path: Path,
        video_timestamps: List[int],
    ) -> List[GripperState]:
        """
        Detect gripper states throughout the video.

        Args:
            video_path: Path to episode video
            video_timestamps: List of timestamp_ns for each video frame

        Returns:
            List of GripperState for each video frame
        """
        if self.client is None:
            await self.setup()

        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        print(f"Analyzing gripper states in video: {video_path.name}")

        # Get prompt template
        prompt = get_gripper_prompt("v1")

        # Analyze video
        response = await self.client.analyze_video(
            video_path,
            prompt,
            json_response=True,
        )

        # Parse response
        data = self.client.parse_json_response(response)

        # Get sampled frames from Gemini response
        frames_data = data.get("frames", [])
        total_duration = data.get("total_duration_seconds", 0)

        if not frames_data:
            print("Warning: No frames detected by Gemini. Using default values.")
            # Return default states
            return [
                GripperState(
                    frame_index=i,
                    timestamp_ns=ts,
                    measured_state=50.0,
                    commanded_state=50.0,
                    confidence=0.0,
                    annotation="unknown",
                )
                for i, ts in enumerate(video_timestamps)
            ]

        # Convert sampled frames to per-frame states
        # Gemini samples at ~1 second intervals, we need to interpolate
        gripper_states = self._interpolate_gripper_states(
            frames_data,
            video_timestamps,
            total_duration,
        )

        print(f"  Detected {len(gripper_states)} gripper states")
        print(f"  Summary: {data.get('summary', 'N/A')}")

        return gripper_states

    def _interpolate_gripper_states(
        self,
        frames_data: List[dict],
        video_timestamps: List[int],
        total_duration: float,
    ) -> List[GripperState]:
        """
        Interpolate sampled gripper states to all video frames.

        Gemini samples at ~1 second intervals, but we need values
        for every video frame (60Hz).
        """
        if not frames_data:
            return []

        # Sort frames by time
        frames_data = sorted(frames_data, key=lambda x: x.get("time_seconds", 0))

        # Extract sample times and values
        sample_times = np.array([f.get("time_seconds", 0) for f in frames_data])
        sample_measured = np.array([f.get("measured", 50) for f in frames_data])
        sample_commanded = np.array([f.get("commanded", 50) for f in frames_data])
        sample_annotations = [f.get("annotation", "unknown") for f in frames_data]

        # Calculate video duration and frame times
        if total_duration > 0:
            duration_ns = int(total_duration * 1e9)
        elif len(video_timestamps) >= 2:
            duration_ns = video_timestamps[-1] - video_timestamps[0]
        else:
            duration_ns = int(1e9)  # Default 1 second

        start_ns = video_timestamps[0] if video_timestamps else 0

        # Create gripper states for each frame
        gripper_states = []

        for i, ts_ns in enumerate(video_timestamps):
            # Convert frame timestamp to seconds from start
            frame_time_s = (ts_ns - start_ns) / 1e9

            # Find interpolation indices
            idx_after = np.searchsorted(sample_times, frame_time_s)

            if idx_after == 0:
                # Before first sample
                measured = sample_measured[0]
                commanded = sample_commanded[0]
                annotation = sample_annotations[0]
            elif idx_after >= len(sample_times):
                # After last sample
                measured = sample_measured[-1]
                commanded = sample_commanded[-1]
                annotation = sample_annotations[-1]
            else:
                # Interpolate between samples
                idx_before = idx_after - 1
                t0 = sample_times[idx_before]
                t1 = sample_times[idx_after]

                if t1 == t0:
                    alpha = 0
                else:
                    alpha = (frame_time_s - t0) / (t1 - t0)

                measured = (
                    sample_measured[idx_before] * (1 - alpha)
                    + sample_measured[idx_after] * alpha
                )
                commanded = (
                    sample_commanded[idx_before] * (1 - alpha)
                    + sample_commanded[idx_after] * alpha
                )

                # Use annotation from nearest sample
                if alpha < 0.5:
                    annotation = sample_annotations[idx_before]
                else:
                    annotation = sample_annotations[idx_after]

            gripper_states.append(
                GripperState(
                    frame_index=i,
                    timestamp_ns=ts_ns,
                    measured_state=float(measured),
                    commanded_state=float(commanded),
                    confidence=0.8,  # Reasonable confidence for interpolated values
                    annotation=annotation,
                )
            )

        return gripper_states

    async def process_episode(self, stage1_dir: Path, output_dir: Path) -> Path:
        """
        Process a single episode from Stage 1.

        Args:
            stage1_dir: Path to Stage 1 output directory
            output_dir: Output directory for Stage 4 results

        Returns:
            Path to output directory
        """
        stage1_dir = Path(stage1_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing episode: {stage1_dir}")

        # Find files
        video_path = stage1_dir / "raw_video.mp4"
        timestamps_path = stage1_dir / "video_timestamps.parquet"

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        if not timestamps_path.exists():
            raise FileNotFoundError(f"Timestamps file not found: {timestamps_path}")

        # Load video timestamps
        timestamps_table = pq.read_table(timestamps_path)
        video_timestamps = timestamps_table["timestamp_ns"].to_pylist()

        print(f"  Loaded {len(video_timestamps)} video timestamps")

        # Initialize Rerun visualization
        self._setup_rerun()

        # Detect gripper states
        gripper_states = await self.detect_gripper_states(video_path, video_timestamps)

        # Visualize results with video playback
        self._visualize_gripper_results(gripper_states, video_path)

        # Save results
        gripper_table = gripper_states_to_table(gripper_states)
        pq.write_table(gripper_table, output_dir / "gripper_states.parquet")

        # Calculate statistics
        measured_values = [s.measured_state for s in gripper_states]
        commanded_values = [s.commanded_state for s in gripper_states]

        # Save metadata
        metadata = StageMetadata(
            stage=4,
            version="1.0.0",
            extra={
                "model": self.config.model,
                "processing_mode": "batch",
                "frames_analyzed": len(gripper_states),
                "statistics": {
                    "measured_mean": float(np.mean(measured_values)),
                    "measured_std": float(np.std(measured_values)),
                    "measured_min": float(np.min(measured_values)),
                    "measured_max": float(np.max(measured_values)),
                    "commanded_mean": float(np.mean(commanded_values)),
                    "commanded_std": float(np.std(commanded_values)),
                    "commanded_min": float(np.min(commanded_values)),
                    "commanded_max": float(np.max(commanded_values)),
                },
            },
        )
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        print(f"  Saved {len(gripper_states)} gripper states")
        print(f"  Measured range: [{np.min(measured_values):.1f}, {np.max(measured_values):.1f}]")
        print(f"  Commanded range: [{np.min(commanded_values):.1f}, {np.max(commanded_values):.1f}]")

        return output_dir

    async def cleanup(self):
        """Cleanup resources."""
        if self.client:
            await self.client.cleanup()


async def main():
    """CLI entry point for Stage 4 gripper detection."""
    parser = argparse.ArgumentParser(
        description="Stage 4: Gripper Detection - Detect gripper states with Gemini"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Path to Stage 1 output directory",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output directory (default: input/../stage4)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.0-flash",
        help="Gemini model to use (default: gemini-2.0-flash)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to pipeline config YAML file",
    )

    args = parser.parse_args()

    # Build config
    if args.config and args.config.exists():
        config = PipelineConfig.from_yaml(args.config)
        stage4_config = config.stage4
    else:
        stage4_config = Stage4Config(model=args.model)

    # Determine paths
    input_dir = Path(args.input)
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = input_dir.parent / "stage4"

    # Process
    detector = GripperDetector(stage4_config)
    try:
        await detector.setup()
        await detector.process_episode(input_dir, output_dir)
    finally:
        await detector.cleanup()

    print(f"\nStage 4 complete. Output: {output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
