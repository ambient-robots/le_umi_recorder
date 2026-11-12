"""
Stage 2: Inverse Kinematics Post-Processing

Reads raw trajectory from Stage 1, applies IK solver to generate
smooth joint trajectories for the robot arm.

Input:
    - Stage 1 outputs: raw_trajectory.parquet, video_timestamps.parquet

Output:
    - joints.parquet: Joint angles per video frame
    - metadata.json: Processing metadata
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import rerun as rr
from scipy.spatial.transform import Rotation

from .schemas import (
    Stage2Config,
    StageMetadata,
    JointState,
    TrajectoryFrame,
    VideoTimestamp,
    joint_states_to_table,
    PipelineConfig,
)
from .interpolation import (
    interpolate_trajectory_to_video,
)


class IKPostProcessor:
    """
    Applies inverse kinematics to raw trajectory data.

    Converts controller poses to robot joint angles using the placo library.
    Interpolates 50Hz trajectory to 60Hz video timestamps.
    """

    def __init__(self, config: Stage2Config):
        """
        Initialize IK post-processor.

        Args:
            config: Stage 2 configuration
        """
        self.config = config
        self.urdf_path = config.urdf_path

        # IK components (initialized in setup)
        self.ik_robot = None
        self.ik_solver = None
        self.ik_task = None

        # Joint names
        self.arm_joint_names = config.joint_names

        # Coordinate frame transforms (from lerobot_recorder.py)
        self._setup_transforms()

        # IK state
        self.origin_position: Optional[np.ndarray] = None
        self.origin_quaternion: Optional[np.ndarray] = None
        self.current_robot_pose: Optional[np.ndarray] = None

        # Scaling factors
        self.vr_to_robot_pos_scale = 1.0
        self.vr_to_robot_ori_scale = 1.0

        # Rerun visualization
        self._rerun_initialized = False
        self.robot_viz = None

    def _setup_rerun(self):
        """Initialize Rerun visualization."""
        rr.init("stage2_ik_solver", spawn=True)
        rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

        # Log world origin
        rr.log("world_origin", rr.Transform3D(axis_length=0.1), static=True)

        # Log robot base transform (180° around Z)
        robot_rot = Rotation.from_euler('z', 180, degrees=True).as_quat()
        rr.log(
            "robot_base",
            rr.Transform3D(
                translation=[0.0, 0.0, 0.0],
                rotation=rr.Quaternion(xyzw=robot_rot.tolist()),
            ),
            static=True,
        )

        # Try to load URDF visualizer
        try:
            from umi.utils.rerun_urdf import RerunURDFVisualizer
            self.robot_viz = RerunURDFVisualizer(self.urdf_path)
            self.robot_viz._load_urdf_to_rerun()
            print("URDF visualization loaded")
        except Exception as e:
            print(f"Warning: Could not load URDF visualizer: {e}")
            self.robot_viz = None

        self._rerun_initialized = True
        print("Rerun visualization started")

    def _visualize_joint_state(self, joint_state: JointState, frame_idx: int, total_frames: int):
        """Visualize current joint state in Rerun."""
        if not self._rerun_initialized:
            return

        # Log joint angles
        for i, name in enumerate(self.arm_joint_names):
            value = getattr(joint_state, name, 0.0)
            rr.log(f"joints/{name}", rr.Scalar(value))

        # Log IK error
        rr.log("ik/error", rr.Scalar(joint_state.ik_error))
        rr.log("ik/success", rr.Scalar(1.0 if joint_state.ik_success else 0.0))

        # Log progress
        progress = (frame_idx + 1) / total_frames
        rr.log("progress", rr.Scalar(progress))

        # Update robot visualization
        if self.robot_viz:
            positions_dict = {name: getattr(joint_state, name, 0.0) for name in self.arm_joint_names}
            self.robot_viz.set_joint_positions(positions_dict)

    def _setup_transforms(self):
        """Set up coordinate frame transforms."""
        # Tracker to normalized: converts gripSpace (Z-wrist, Y-head) to (Z-up, Y-front)
        self.T_tracker_to_normalized = np.eye(4)
        self.T_tracker_to_normalized[:3, :3] = Rotation.from_euler(
            'x', -125, degrees=True
        ).as_matrix()

        # Tool frame offset relative to normalized frame
        self.T_normalized_to_tool = np.eye(4)
        self.T_normalized_to_tool[:3, 3] = [0, 0.15, -0.03]

        # Robot base rotation (180° around Z so robot faces user)
        self.robot_base_rotation = Rotation.from_euler('z', 180, degrees=True).as_matrix()

        # Tool-to-gripper rotation: swaps X↔Y axes
        self.tool_to_gripper_rotation = np.array([
            [0, 1, 0],
            [-1, 0, 0],
            [0, 0, 1]
        ], dtype=np.float64)

    def setup(self):
        """Initialize placo IK solver."""
        import placo

        if not self.urdf_path:
            raise ValueError("URDF path is required for IK solving")

        print(f"Initializing IK solver with URDF: {self.urdf_path}")

        # Load robot
        self.ik_robot = placo.RobotWrapper(self.urdf_path, placo.Flags.ignore_collisions)
        self.ik_solver = placo.KinematicsSolver(self.ik_robot)
        self.ik_solver.mask_fbase(True)
        self.ik_solver.dt = self.config.solver_dt

        # Add frame task for gripper
        self.ik_task = self.ik_solver.add_frame_task(
            self.config.frame_name, np.eye(4)
        )
        self.ik_task.configure(self.config.frame_name, "soft", 1.0, 0.2)
        self.ik_solver.add_regularization_task(self.config.regularization)

        # Set initial joint positions to zero
        for name in self.arm_joint_names:
            self.ik_robot.set_joint(name, 0.0)
        self.ik_robot.update_kinematics()

        # Initialize current pose to home position
        self.current_robot_pose = self.ik_robot.get_T_world_frame(
            self.config.frame_name
        ).copy()

        print(f"  Joint names: {self.arm_joint_names}")
        print("  IK solver ready")

    def reset(self):
        """Reset IK state for new episode."""
        self.origin_position = None
        self.origin_quaternion = None

        if self.ik_robot:
            # Reset to home position
            for name in self.arm_joint_names:
                self.ik_robot.set_joint(name, 0.0)
            self.ik_robot.update_kinematics()
            self.current_robot_pose = self.ik_robot.get_T_world_frame(
                self.config.frame_name
            ).copy()

    def _compute_relative_rotvec(
        self, origin_quat: np.ndarray, current_quat: np.ndarray
    ) -> np.ndarray:
        """Compute relative rotation as rotation vector (axis-angle)."""
        origin_quat = origin_quat / np.linalg.norm(origin_quat)
        current_quat = current_quat / np.linalg.norm(current_quat)

        # Handle quaternion double cover
        if np.dot(origin_quat, current_quat) < 0:
            current_quat = -current_quat

        origin_rot = Rotation.from_quat(origin_quat)
        current_rot = Rotation.from_quat(current_quat)
        relative_rot = origin_rot.inv() * current_rot

        return relative_rot.as_rotvec()

    def solve_ik(self, frame: TrajectoryFrame) -> JointState:
        """
        Solve IK for a single trajectory frame.

        Args:
            frame: Trajectory frame with controller pose

        Returns:
            Joint state with solved angles
        """
        if self.ik_solver is None:
            raise RuntimeError("IK solver not initialized. Call setup() first.")

        # Build pose from frame
        raw_pos = np.array([frame.position_x, frame.position_y, frame.position_z])
        raw_quat = np.array([
            frame.orientation_x,
            frame.orientation_y,
            frame.orientation_z,
            frame.orientation_w,
        ])

        # Transform to tool frame
        T_world_tracker = np.eye(4)
        T_world_tracker[:3, :3] = Rotation.from_quat(raw_quat).as_matrix()
        T_world_tracker[:3, 3] = raw_pos

        T_world_tool = T_world_tracker @ self.T_tracker_to_normalized @ self.T_normalized_to_tool
        current_pos = T_world_tool[:3, 3]
        current_quat = Rotation.from_matrix(T_world_tool[:3, :3]).as_quat()

        # Initialize origin on first call
        if self.origin_position is None:
            self.origin_position = current_pos.copy()
            self.origin_quaternion = current_quat.copy()
            if self.current_robot_pose is None:
                self.current_robot_pose = self.ik_robot.get_T_world_frame(
                    self.config.frame_name
                ).copy()

        # Compute incremental deltas
        delta_pos = (current_pos - self.origin_position) * self.vr_to_robot_pos_scale
        delta_rotvec = self._compute_relative_rotvec(
            self.origin_quaternion, current_quat
        ) * self.vr_to_robot_ori_scale
        delta_rot = Rotation.from_rotvec(delta_rotvec).as_matrix()

        # Update origin for next frame
        self.origin_position = current_pos.copy()
        self.origin_quaternion = current_quat.copy()

        # Transform deltas to robot frame
        delta_pos_robot = self.robot_base_rotation @ delta_pos
        delta_rot_robot = (
            self.tool_to_gripper_rotation @ delta_rot @ self.tool_to_gripper_rotation.T
        )

        # Apply to current robot pose
        self.current_robot_pose[:3, 3] += delta_pos_robot
        self.current_robot_pose[:3, :3] = self.current_robot_pose[:3, :3] @ delta_rot_robot

        # Solve IK
        self.ik_task.T_world_frame = self.current_robot_pose
        self.ik_solver.solve(True)
        self.ik_robot.update_kinematics()

        # Get joint positions
        joint_positions = [
            self.ik_robot.get_joint(name) for name in self.arm_joint_names
        ]

        # Calculate IK error
        actual_pose = self.ik_robot.get_T_world_frame(self.config.frame_name)
        pos_error = np.linalg.norm(
            actual_pose[:3, 3] - self.current_robot_pose[:3, 3]
        )

        return JointState(
            timestamp_ns=frame.timestamp_ns,
            frame_index=frame.frame_index,
            shoulder_pan=joint_positions[0],
            shoulder_lift=joint_positions[1],
            elbow_flex=joint_positions[2],
            elbow_roll=joint_positions[3],
            wrist_flex=joint_positions[4],
            wrist_roll=joint_positions[5],
            ik_success=pos_error < 0.05,  # 5cm threshold
            ik_error=float(pos_error),
        )

    def process_episode(self, stage1_dir: Path, output_dir: Path) -> Path:
        """
        Process a single episode from Stage 1.

        Args:
            stage1_dir: Path to Stage 1 output directory
            output_dir: Output directory for Stage 2 results

        Returns:
            Path to output directory
        """
        stage1_dir = Path(stage1_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing episode: {stage1_dir}")

        # Load Stage 1 data
        trajectory_path = stage1_dir / "raw_trajectory.parquet"
        timestamps_path = stage1_dir / "video_timestamps.parquet"

        if not trajectory_path.exists():
            raise FileNotFoundError(f"Trajectory file not found: {trajectory_path}")
        if not timestamps_path.exists():
            raise FileNotFoundError(f"Timestamps file not found: {timestamps_path}")

        trajectory_table = pq.read_table(trajectory_path)
        timestamps_table = pq.read_table(timestamps_path)

        print(f"  Loaded {len(trajectory_table)} trajectory frames")
        print(f"  Loaded {len(timestamps_table)} video timestamps")

        # Convert timestamps to list of VideoTimestamp
        video_timestamps = [
            VideoTimestamp(
                frame_index=int(timestamps_table["frame_index"][i].as_py()),
                timestamp_ns=int(timestamps_table["timestamp_ns"][i].as_py()),
                closest_trajectory_idx=int(
                    timestamps_table["closest_trajectory_idx"][i].as_py()
                ),
            )
            for i in range(len(timestamps_table))
        ]

        # Interpolate trajectory to video timestamps
        print("  Interpolating trajectory to video timestamps...")
        interpolated_frames = interpolate_trajectory_to_video(
            trajectory_table, video_timestamps
        )
        print(f"  Interpolated to {len(interpolated_frames)} frames")

        # Reset IK state
        self.reset()

        # Initialize Rerun visualization
        self._setup_rerun()

        # Solve IK for each frame
        print("  Solving IK for each frame...")
        joint_states: List[JointState] = []
        success_count = 0
        total_frames = len(interpolated_frames)

        for i, frame in enumerate(interpolated_frames):
            # Set Rerun time
            rr.set_time_sequence("frame", i)

            joint_state = self.solve_ik(frame)
            joint_states.append(joint_state)

            if joint_state.ik_success:
                success_count += 1

            # Visualize every frame (for animation playback)
            self._visualize_joint_state(joint_state, i, total_frames)

            if (i + 1) % 100 == 0:
                print(f"    Processed {i + 1}/{total_frames} frames")

        # Save results
        joints_table = joint_states_to_table(joint_states)
        pq.write_table(joints_table, output_dir / "joints.parquet")

        # Save metadata
        metadata = StageMetadata(
            stage=2,
            version="1.0.0",
            extra={
                "urdf_path": self.urdf_path,
                "solver_config": {
                    "dt": self.config.solver_dt,
                    "regularization": self.config.regularization,
                    "frame_name": self.config.frame_name,
                },
                "joint_names": self.arm_joint_names,
                "total_frames": len(joint_states),
                "successful_frames": success_count,
                "success_rate": success_count / len(joint_states) if joint_states else 0,
            },
        )
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        print(f"  Saved {len(joint_states)} joint states")
        print(f"  IK success rate: {success_count}/{len(joint_states)} "
              f"({100*success_count/len(joint_states):.1f}%)")

        return output_dir


def main():
    """CLI entry point for Stage 2 IK processing."""
    parser = argparse.ArgumentParser(
        description="Stage 2: IK Post-Processing - Convert poses to joint angles"
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
        help="Output directory (default: input/../stage2)",
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        required=True,
        help="Path to robot URDF file",
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
        stage2_config = config.stage2
        stage2_config.urdf_path = str(args.urdf)
    else:
        stage2_config = Stage2Config(urdf_path=str(args.urdf))

    # Determine output directory
    input_dir = Path(args.input)
    if args.output:
        output_dir = Path(args.output)
    else:
        # Default: sibling stage2 directory
        output_dir = input_dir.parent / "stage2"

    # Process
    processor = IKPostProcessor(stage2_config)
    processor.setup()
    processor.process_episode(input_dir, output_dir)

    print(f"\nStage 2 complete. Output: {output_dir}")


if __name__ == "__main__":
    main()
