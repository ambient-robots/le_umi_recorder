"""
Final Assembly: Combine All Stages into LeRobot v2.1 Dataset

Reads outputs from all pipeline stages and produces a complete
LeRobot-compatible dataset for training with Physical Intelligence
PI 0.5/0.6 models.

Input:
    - Stage 1: raw_video.mp4, video_timestamps.parquet
    - Stage 2: joints.parquet
    - Stage 3: task_prompt.json
    - Stage 4: gripper_states.parquet

Output:
    - LeRobot v2.1 dataset structure:
        - meta/info.json
        - meta/stats.json
        - meta/episodes.jsonl
        - data/episode_NNNNNN.parquet
        - videos/observation.images.{camera}/episode_NNNNNN.mp4
"""

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .schemas import (
    AssemblyConfig,
    PipelineConfig,
)


class PipelineAssembler:
    """
    Combines outputs from all pipeline stages into LeRobot v2.1 format.

    Dataset format:
    - action: [joint_angles (6), gripper_commanded (1)] = 7D
    - observation.state: [joint_angles (6), gripper_measured (1)] = 7D
    - observation.images.{camera}: Video frames (stored as MP4)
    - task: Generated task string per episode
    """

    # Joint names in order
    JOINT_NAMES = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "elbow_roll",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    ]

    def __init__(self, config: AssemblyConfig, output_dir: Path):
        """
        Initialize assembler.

        Args:
            config: Assembly configuration
            output_dir: Output directory for dataset
        """
        self.config = config
        self.output_dir = Path(output_dir)

        # Dataset state
        self.episode_index = 0
        self.total_frames = 0

        # Accumulate stats across episodes
        self.stats_accumulators: Dict[str, Dict[str, List[float]]] = {}

    def setup(self):
        """Set up output directory structure."""
        print(f"Setting up LeRobot v2.1 dataset: {self.output_dir}")

        # Create directory structure
        (self.output_dir / "meta").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "data").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "videos").mkdir(parents=True, exist_ok=True)

        # Initialize episodes file
        episodes_file = self.output_dir / "meta" / "episodes.jsonl"
        if episodes_file.exists():
            # Count existing episodes
            with open(episodes_file) as f:
                self.episode_index = sum(1 for _ in f)
            print(f"  Continuing from episode {self.episode_index}")

    def assemble_episode(
        self,
        episode_dir: Path,
        camera_name: str = "wrist",
    ) -> int:
        """
        Assemble a single episode from pipeline outputs.

        Args:
            episode_dir: Directory containing stage outputs
            camera_name: Name of the camera for video observations

        Returns:
            Number of frames in the episode
        """
        episode_dir = Path(episode_dir)
        print(f"\nAssembling episode from: {episode_dir}")

        # Locate stage directories
        stage1_dir = episode_dir / "stage1"
        stage2_dir = episode_dir / "stage2"
        stage3_dir = episode_dir / "stage3"
        stage4_dir = episode_dir / "stage4"

        # Verify required files exist
        required_files = [
            stage1_dir / "raw_video.mp4",
            stage1_dir / "video_timestamps.parquet",
            stage2_dir / "joints.parquet",
            stage4_dir / "gripper_states.parquet",
        ]
        for f in required_files:
            if not f.exists():
                raise FileNotFoundError(f"Required file not found: {f}")

        # Load data from all stages
        joints_table = pq.read_table(stage2_dir / "joints.parquet")
        gripper_table = pq.read_table(stage4_dir / "gripper_states.parquet")
        timestamps_table = pq.read_table(stage1_dir / "video_timestamps.parquet")

        # Load task prompt if available
        task_prompt_file = stage3_dir / "task_prompt.json"
        if task_prompt_file.exists():
            with open(task_prompt_file) as f:
                task_data = json.load(f)
                task = task_data.get("task_description", "Unknown task")
        else:
            task = "Unknown task"
            print("  Warning: No task prompt found, using default")

        num_frames = len(joints_table)
        print(f"  Frames: {num_frames}")
        print(f"  Task: {task}")

        # Build episode data
        episode_data = self._build_episode_data(
            joints_table,
            gripper_table,
            timestamps_table,
            self.episode_index,
        )

        # Save episode parquet
        episode_parquet_path = (
            self.output_dir / "data" / f"episode_{self.episode_index:06d}.parquet"
        )
        pq.write_table(episode_data, episode_parquet_path)
        print(f"  Saved: {episode_parquet_path.name}")

        # Copy video to dataset
        video_dir = self.output_dir / "videos" / f"observation.images.{camera_name}"
        video_dir.mkdir(parents=True, exist_ok=True)
        video_dest = video_dir / f"episode_{self.episode_index:06d}.mp4"
        shutil.copy(stage1_dir / "raw_video.mp4", video_dest)
        print(f"  Copied video to: {video_dest.relative_to(self.output_dir)}")

        # Update episodes.jsonl
        duration_s = num_frames / self.config.fps
        episode_meta = {
            "episode_index": self.episode_index,
            "num_frames": num_frames,
            "task": task,
            "length_s": duration_s,
        }
        with open(self.output_dir / "meta" / "episodes.jsonl", "a") as f:
            f.write(json.dumps(episode_meta) + "\n")

        # Update stats accumulators
        self._update_stats(episode_data)

        self.episode_index += 1
        self.total_frames += num_frames

        return num_frames

    def _build_episode_data(
        self,
        joints_table: pa.Table,
        gripper_table: pa.Table,
        timestamps_table: pa.Table,
        episode_index: int,
    ) -> pa.Table:
        """Build episode parquet data from stage outputs."""
        num_frames = len(joints_table)

        # Extract joint angles (6D)
        joint_angles = np.column_stack([
            joints_table["shoulder_pan"].to_numpy(),
            joints_table["shoulder_lift"].to_numpy(),
            joints_table["elbow_flex"].to_numpy(),
            joints_table["elbow_roll"].to_numpy(),
            joints_table["wrist_flex"].to_numpy(),
            joints_table["wrist_roll"].to_numpy(),
        ])

        # Extract gripper states
        # Scale from 0-100 to 0-1 for dataset
        gripper_measured = gripper_table["measured_state"].to_numpy() / 100.0
        gripper_commanded = gripper_table["commanded_state"].to_numpy() / 100.0

        # Build action: [joint_angles, gripper_commanded]
        action = np.column_stack([joint_angles, gripper_commanded])

        # Build observation.state: [joint_angles, gripper_measured]
        observation_state = np.column_stack([joint_angles, gripper_measured])

        # Build timestamps
        frame_indices = np.arange(num_frames, dtype=np.int64)
        episode_indices = np.full(num_frames, episode_index, dtype=np.int64)
        timestamps = frame_indices.astype(np.float64) / self.config.fps

        # Create table
        data = {
            "frame_index": frame_indices,
            "episode_index": episode_indices,
            "timestamp": timestamps,
            "action": [row.tolist() for row in action.astype(np.float32)],
            "observation.state": [
                row.tolist() for row in observation_state.astype(np.float32)
            ],
        }

        return pa.Table.from_pydict(data)

    def _update_stats(self, episode_data: pa.Table):
        """Update running statistics from episode data."""
        # Extract arrays
        actions = np.array(episode_data["action"].to_pylist())
        observations = np.array(episode_data["observation.state"].to_pylist())

        for name, arr in [("action", actions), ("observation.state", observations)]:
            if name not in self.stats_accumulators:
                self.stats_accumulators[name] = {
                    "values": [],
                    "mins": [],
                    "maxs": [],
                }
            self.stats_accumulators[name]["values"].append(arr)

    def finalize(self, camera_name: str = "wrist", camera_shape: tuple = (480, 640, 3)):
        """
        Finalize dataset by computing statistics and writing info.json.

        Args:
            camera_name: Camera name used in the dataset
            camera_shape: Shape of camera frames (H, W, C)
        """
        print(f"\nFinalizing dataset...")

        # Compute statistics
        stats = self._compute_stats()

        # Write stats.json
        stats_file = self.output_dir / "meta" / "stats.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  Saved: meta/stats.json")

        # Build info.json
        info = {
            "codebase_version": "v2.1",
            "robot_type": "umi",
            "fps": self.config.fps,
            "features": {
                "action": {
                    "dtype": "float32",
                    "shape": [7],
                    "names": {"axes": self.JOINT_NAMES},
                },
                "observation.state": {
                    "dtype": "float32",
                    "shape": [7],
                    "names": {"axes": self.JOINT_NAMES},
                },
                f"observation.images.{camera_name}": {
                    "dtype": "video",
                    "shape": list(camera_shape),
                    "names": ["height", "width", "channel"],
                },
            },
            "total_episodes": self.episode_index,
            "total_frames": self.total_frames,
            "repo_id": self.config.repo_id or f"user/{self.output_dir.name}",
            "created_at": datetime.now().isoformat(),
        }

        # Write info.json
        info_file = self.output_dir / "meta" / "info.json"
        with open(info_file, "w") as f:
            json.dump(info, f, indent=2)
        print(f"  Saved: meta/info.json")

        print(f"\nDataset complete:")
        print(f"  Episodes: {self.episode_index}")
        print(f"  Total frames: {self.total_frames}")
        print(f"  Output: {self.output_dir}")

    def _compute_stats(self) -> Dict[str, Any]:
        """Compute dataset statistics for normalization."""
        stats = {}

        for name, accumulator in self.stats_accumulators.items():
            if not accumulator["values"]:
                continue

            # Concatenate all values
            all_values = np.concatenate(accumulator["values"], axis=0)

            # Compute stats per dimension
            stats[name] = {
                "mean": all_values.mean(axis=0).tolist(),
                "std": all_values.std(axis=0).tolist(),
                "min": all_values.min(axis=0).tolist(),
                "max": all_values.max(axis=0).tolist(),
            }

        return stats


def assemble_session(
    recordings_dir: Path,
    output_dir: Path,
    config: AssemblyConfig,
    camera_name: str = "wrist",
):
    """
    Assemble all episodes in a recording session.

    Args:
        recordings_dir: Directory containing episode subdirectories
        output_dir: Output directory for dataset
        config: Assembly configuration
        camera_name: Camera name for video observations
    """
    recordings_dir = Path(recordings_dir)
    output_dir = Path(output_dir)

    # Find all episode directories
    episode_dirs = sorted(
        [d for d in recordings_dir.iterdir() if d.is_dir() and d.name.startswith("episode_")]
    )

    if not episode_dirs:
        print(f"No episode directories found in: {recordings_dir}")
        return

    print(f"Found {len(episode_dirs)} episodes to assemble")

    # Initialize assembler
    assembler = PipelineAssembler(config, output_dir)
    assembler.setup()

    # Process each episode
    for episode_dir in episode_dirs:
        try:
            assembler.assemble_episode(episode_dir, camera_name)
        except FileNotFoundError as e:
            print(f"  Skipping {episode_dir.name}: {e}")
            continue
        except Exception as e:
            print(f"  Error processing {episode_dir.name}: {e}")
            continue

    # Finalize
    assembler.finalize(camera_name)


def main():
    """CLI entry point for dataset assembly."""
    parser = argparse.ArgumentParser(
        description="Assemble pipeline outputs into LeRobot v2.1 dataset"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Path to recordings directory (containing episode_* subdirs)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output directory for dataset",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="wrist",
        help="Camera name for video observations (default: wrist)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Dataset FPS (default: 60)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        help="Repository ID for dataset (e.g., user/dataset_name)",
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
        assembly_config = config.assembly
    else:
        assembly_config = AssemblyConfig(
            fps=args.fps,
            repo_id=args.repo_id or "",
        )

    # Assemble
    assemble_session(
        args.input,
        args.output,
        assembly_config,
        args.camera,
    )


if __name__ == "__main__":
    main()
