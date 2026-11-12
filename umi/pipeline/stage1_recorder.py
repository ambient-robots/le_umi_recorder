"""
Stage 1: Raw Recording

Records Quest tracker trajectory at 50Hz and video at 60Hz with precise
nanosecond timestamps for synchronization.

Output:
    - raw_trajectory.parquet: Controller poses with timestamps
    - raw_video.mp4: Video footage
    - video_timestamps.parquet: Frame-level timestamps for sync
    - metadata.json: Recording metadata
"""

import argparse
import asyncio
import json
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import rerun as rr
from scipy.spatial.transform import Rotation

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'backend'))
from api import ControllerTrackingClient, PoseData, ControllerState

from .schemas import (
    Stage1Config,
    StageMetadata,
    TrajectoryFrame,
    VideoTimestamp,
    trajectory_frames_to_table,
    video_timestamps_to_table,
    PipelineConfig,
)


class RawRecorder:
    """
    Records raw trajectory and video data without any processing.

    Key features:
    - 50Hz trajectory capture from Quest controller
    - 60Hz video capture from USB camera
    - Nanosecond timestamps for precise sync
    - Arrow/Parquet intermediate format
    """

    def __init__(self, config: Stage1Config, output_dir: Path):
        """
        Initialize the raw recorder.

        Args:
            config: Stage 1 configuration
            output_dir: Base output directory for recordings
        """
        self.config = config
        self.output_dir = Path(output_dir)

        # Recording state
        self.is_recording = False
        self.episode_count = 0

        # Data buffers
        self.trajectory_buffer: List[TrajectoryFrame] = []
        self.video_timestamps: List[VideoTimestamp] = []

        # Timing
        self.trajectory_interval_ns = int(1e9 / config.trajectory_fps)
        self.video_interval_ns = int(1e9 / config.video_fps)

        # Video writer
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.video_frame_count = 0

        # Camera
        self.camera: Optional[cv2.VideoCapture] = None

        # Tracking client
        self.client = ControllerTrackingClient(config.server_url)
        self.latest_pose: Optional[PoseData] = None

        # Thread control
        self._stop_event = threading.Event()
        self._trajectory_thread: Optional[threading.Thread] = None
        self._video_thread: Optional[threading.Thread] = None

        # Keyboard state
        self._pending_action: Optional[str] = None
        self._quit_requested = False
        self._lock = threading.Lock()

        # Rerun visualization
        self._rerun_initialized = False
        self._latest_frame: Optional[np.ndarray] = None

    def _setup_rerun(self):
        """Initialize Rerun visualization."""
        rr.init("stage1_raw_recorder", spawn=True)
        rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

        # Log world origin
        rr.log("world_origin", rr.Transform3D(axis_length=0.1), static=True)

        self._rerun_initialized = True
        print("Rerun visualization started")

    def _visualize_controller(self, controller: ControllerState, hand: str):
        """Visualize controller pose in Rerun."""
        if not self._rerun_initialized:
            return

        pos = np.array(controller.position)
        quat = np.array(controller.orientation)  # xyzw

        # Log controller transform
        rr.log(
            f"controller/{hand}",
            rr.Transform3D(
                translation=pos,
                rotation=rr.Quaternion(xyzw=quat.tolist()),
                axis_length=0.05,
            ),
        )

        # Log as point for trajectory visualization
        rr.log(
            f"controller/{hand}/position",
            rr.Points3D([pos], radii=0.01),
        )

        # Log trigger/grip values
        buttons = controller.buttons or {}
        trigger = buttons.get("trigger", {}).get("value", 0.0)
        grip = buttons.get("grip", {}).get("value", 0.0)

        rr.log("buttons/trigger", rr.Scalar(trigger))
        rr.log("buttons/grip", rr.Scalar(grip))

    def _visualize_camera_frame(self, frame: np.ndarray):
        """Visualize camera frame in Rerun."""
        if not self._rerun_initialized:
            return

        # Convert BGR to RGB for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rr.log("camera/image", rr.Image(frame_rgb))

    def _visualize_status(self):
        """Visualize recording status in Rerun."""
        if not self._rerun_initialized:
            return

        status = "RECORDING" if self.is_recording else "IDLE"
        color = [255, 0, 0] if self.is_recording else [128, 128, 128]

        rr.log(
            "status/recording",
            rr.TextLog(
                f"Episode {self.episode_count} - {status} "
                f"[Traj: {len(self.trajectory_buffer)}, Video: {self.video_frame_count}]"
            ),
        )

        if self.is_recording:
            rr.log("recording/trajectory_frames", rr.Scalar(len(self.trajectory_buffer)))
            rr.log("recording/video_frames", rr.Scalar(self.video_frame_count))

    async def setup(self):
        """Initialize camera and server connection."""
        print(f"Setting up Stage 1 Raw Recorder...")
        print(f"  Trajectory FPS: {self.config.trajectory_fps}")
        print(f"  Video FPS: {self.config.video_fps}")
        print(f"  Controller: {self.config.controller_hand}")

        # Connect camera
        cam_cfg = self.config.camera
        print(f"\nConnecting to camera {cam_cfg.name} (index {cam_cfg.index})...")

        self.camera = cv2.VideoCapture(cam_cfg.index)
        if not self.camera.isOpened():
            raise RuntimeError(f"Could not open camera {cam_cfg.index}")

        # Configure camera
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg.width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg.height)
        self.camera.set(cv2.CAP_PROP_FPS, cam_cfg.fps)

        actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
        actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"  Camera ready: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS")

        # Check server
        try:
            status = await self.client.get_status()
            print(f"\nServer status: {status}")
            if not status.get('has_pose_data'):
                print("Warning: No pose data available. Make sure Quest is connected.")
        except Exception as e:
            print(f"Warning: Could not connect to server: {e}")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Rerun visualization
        self._setup_rerun()

        # Print controls
        self._print_controls()

    def _print_controls(self):
        """Print available controls."""
        print("\n" + "=" * 50)
        print("STAGE 1: RAW RECORDING")
        print("=" * 50)
        print("Controls:")
        print("  [s] Start recording episode")
        print("  [e] End episode (save)")
        print("  [a] Abort episode (discard)")
        print("  [q] Quit")
        print("=" * 50 + "\n")

    def _start_keyboard_listener(self):
        """Start keyboard listener thread."""
        try:
            from pynput import keyboard

            def on_press(key):
                try:
                    char = key.char.lower() if hasattr(key, 'char') and key.char else None
                    if char:
                        with self._lock:
                            if char == 's':
                                self._pending_action = 'start'
                            elif char == 'e':
                                self._pending_action = 'end'
                            elif char == 'a':
                                self._pending_action = 'abort'
                            elif char == 'q':
                                self._quit_requested = True
                except Exception:
                    pass

            listener = keyboard.Listener(on_press=on_press)
            listener.daemon = True
            listener.start()
            print("Keyboard listener started")
        except ImportError:
            print("Warning: pynput not available, keyboard control disabled")

    def _get_pending_action(self) -> Optional[str]:
        """Get and clear pending action."""
        with self._lock:
            action = self._pending_action
            self._pending_action = None
            return action

    def _handle_action(self, action: str):
        """Handle keyboard action."""
        if action == 'start' and not self.is_recording:
            self._start_episode()
        elif action == 'end' and self.is_recording:
            self._end_episode()
        elif action == 'abort' and self.is_recording:
            self._abort_episode()

    def _start_episode(self):
        """Start recording a new episode."""
        episode_dir = self.output_dir / f"episode_{self.episode_count:06d}" / "stage1"
        episode_dir.mkdir(parents=True, exist_ok=True)

        # Initialize video writer
        video_path = episode_dir / "raw_video.mp4"
        cam_cfg = self.config.camera
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            str(video_path),
            fourcc,
            self.config.video_fps,
            (cam_cfg.width, cam_cfg.height),
        )

        # Clear buffers
        self.trajectory_buffer.clear()
        self.video_timestamps.clear()
        self.video_frame_count = 0

        self.is_recording = True
        self._current_episode_dir = episode_dir

        print(f"\n>>> Recording episode {self.episode_count} started")
        print(f"    Output: {episode_dir}")

    def _end_episode(self):
        """End and save current episode."""
        if not self.is_recording:
            return

        self.is_recording = False
        episode_dir = self._current_episode_dir

        # Close video writer
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None

        # Save trajectory data
        if self.trajectory_buffer:
            trajectory_table = trajectory_frames_to_table(self.trajectory_buffer)
            pq.write_table(
                trajectory_table,
                episode_dir / "raw_trajectory.parquet",
            )
            print(f"    Saved {len(self.trajectory_buffer)} trajectory frames")

        # Save video timestamps
        if self.video_timestamps:
            timestamps_table = video_timestamps_to_table(self.video_timestamps)
            pq.write_table(
                timestamps_table,
                episode_dir / "video_timestamps.parquet",
            )
            print(f"    Saved {len(self.video_timestamps)} video timestamps")

        # Save metadata
        metadata = StageMetadata(
            stage=1,
            version="1.0.0",
            extra={
                "trajectory_fps": self.config.trajectory_fps,
                "video_fps": self.config.video_fps,
                "trajectory_frames": len(self.trajectory_buffer),
                "video_frames": self.video_frame_count,
                "camera_config": {
                    "name": self.config.camera.name,
                    "width": self.config.camera.width,
                    "height": self.config.camera.height,
                },
                "controller_hand": self.config.controller_hand,
            },
        )
        with open(episode_dir / "metadata.json", "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        print(f">>> Episode {self.episode_count} saved")
        print(f"    Trajectory: {len(self.trajectory_buffer)} frames @ {self.config.trajectory_fps}Hz")
        print(f"    Video: {self.video_frame_count} frames @ {self.config.video_fps}Hz")

        self.episode_count += 1

    def _abort_episode(self):
        """Abort current episode without saving."""
        if not self.is_recording:
            return

        self.is_recording = False

        # Close video writer
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None

        # Delete episode directory
        import shutil
        if hasattr(self, '_current_episode_dir') and self._current_episode_dir.exists():
            shutil.rmtree(self._current_episode_dir)

        print(f">>> Episode {self.episode_count} aborted")

    def _capture_trajectory_frame(self):
        """Capture a single trajectory frame."""
        if self.latest_pose is None:
            return

        # Get controller data
        hand = self.config.controller_hand
        controller = self.latest_pose.right if hand == "right" else self.latest_pose.left

        if controller is None:
            return

        timestamp_ns = time.time_ns()
        frame_index = len(self.trajectory_buffer)

        # Get button values
        buttons = controller.buttons or {}
        trigger = buttons.get("trigger", {}).get("value", 0.0)
        grip = buttons.get("grip", {}).get("value", 0.0)

        frame = TrajectoryFrame(
            timestamp_ns=timestamp_ns,
            frame_index=frame_index,
            position_x=controller.position[0],
            position_y=controller.position[1],
            position_z=controller.position[2],
            orientation_x=controller.orientation[0],
            orientation_y=controller.orientation[1],
            orientation_z=controller.orientation[2],
            orientation_w=controller.orientation[3],
            button_trigger=trigger,
            button_grip=grip,
            controller_hand=hand,
        )
        self.trajectory_buffer.append(frame)

    def _capture_video_frame(self) -> Optional[np.ndarray]:
        """Capture a single video frame."""
        if self.camera is None:
            return None

        ret, frame = self.camera.read()
        if not ret or frame is None:
            return None

        timestamp_ns = time.time_ns()

        # Write to video
        if self.video_writer:
            self.video_writer.write(frame)

        # Find closest trajectory frame
        closest_traj_idx = len(self.trajectory_buffer) - 1 if self.trajectory_buffer else 0

        # Record timestamp
        video_ts = VideoTimestamp(
            frame_index=self.video_frame_count,
            timestamp_ns=timestamp_ns,
            closest_trajectory_idx=closest_traj_idx,
        )
        self.video_timestamps.append(video_ts)
        self.video_frame_count += 1

        # Store for visualization
        self._latest_frame = frame
        return frame

    def on_pose_update(self, pose: PoseData):
        """Callback for pose updates from server."""
        self.latest_pose = pose

    async def run(self):
        """Main recording loop."""
        await self.setup()
        self._start_keyboard_listener()

        # Start WebSocket stream for pose updates
        stream_task = asyncio.create_task(
            self.client.connect_stream(self.on_pose_update)
        )

        print("\nRecording loop started. Press 's' to start, 'q' to quit.")

        # Timing
        trajectory_interval = 1.0 / self.config.trajectory_fps
        video_interval = 1.0 / self.config.video_fps

        last_trajectory_time = 0.0
        last_video_time = 0.0
        last_status_time = time.time()

        try:
            while not self._quit_requested:
                current_time = time.time()

                # Check for keyboard actions
                action = self._get_pending_action()
                if action:
                    self._handle_action(action)

                if self.is_recording:
                    # Capture trajectory at 50Hz
                    if current_time - last_trajectory_time >= trajectory_interval:
                        self._capture_trajectory_frame()
                        last_trajectory_time = current_time

                    # Capture video at 60Hz
                    if current_time - last_video_time >= video_interval:
                        frame = self._capture_video_frame()
                        if frame is not None:
                            self._visualize_camera_frame(frame)
                        last_video_time = current_time

                # Visualize controller pose (always, even when not recording)
                if self.latest_pose:
                    hand = self.config.controller_hand
                    controller = self.latest_pose.right if hand == "right" else self.latest_pose.left
                    if controller:
                        self._visualize_controller(controller, hand)

                # Status update every 2 seconds
                if current_time - last_status_time >= 2.0:
                    self._visualize_status()
                    if self.is_recording:
                        print(
                            f"\r  Recording: {len(self.trajectory_buffer)} traj, "
                            f"{self.video_frame_count} video frames",
                            end="",
                            flush=True,
                        )
                    last_status_time = current_time

                # Small sleep to prevent busy loop
                await asyncio.sleep(0.001)

        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        finally:
            # Cleanup
            stream_task.cancel()
            try:
                await stream_task
            except asyncio.CancelledError:
                pass

            if self.is_recording:
                print("\nSaving current episode before exit...")
                self._end_episode()

            if self.camera:
                self.camera.release()

            await self.client.close()
            print("\nRecorder shutdown complete.")


async def main():
    """CLI entry point for Stage 1 recording."""
    parser = argparse.ArgumentParser(
        description="Stage 1: Raw Recording - Capture Quest trajectory and video"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to pipeline config YAML file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("./recordings") / datetime.now().strftime("%Y%m%d_%H%M%S"),
        help="Output directory for recordings",
    )
    parser.add_argument(
        "--trajectory-fps",
        type=int,
        default=50,
        help="Trajectory capture FPS (default: 50)",
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=60,
        help="Video capture FPS (default: 60)",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Camera index (default: 0)",
    )
    parser.add_argument(
        "--camera-name",
        type=str,
        default="wrist",
        help="Camera name (default: wrist)",
    )
    parser.add_argument(
        "--server",
        type=str,
        default="https://localhost:8000",
        help="Quest tracking server URL",
    )
    parser.add_argument(
        "--hand",
        type=str,
        choices=["left", "right"],
        default="right",
        help="Controller hand to track (default: right)",
    )

    args = parser.parse_args()

    # Build config
    if args.config and args.config.exists():
        config = PipelineConfig.from_yaml(args.config)
        stage1_config = config.stage1
    else:
        from .schemas import CameraConfig as CamCfg
        stage1_config = Stage1Config(
            trajectory_fps=args.trajectory_fps,
            video_fps=args.video_fps,
            camera=CamCfg(
                name=args.camera_name,
                index=args.camera_index,
            ),
            controller_hand=args.hand,
            server_url=args.server,
        )

    recorder = RawRecorder(stage1_config, args.output)
    await recorder.run()


if __name__ == "__main__":
    asyncio.run(main())
