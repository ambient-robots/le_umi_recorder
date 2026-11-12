"""
LeRobot Dataset Recorder for UMI
Records controller poses and camera observations in LeRobot Dataset format
with real-time Rerun visualization
"""

import asyncio
import argparse
import sys
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import rerun as rr
from rerun.components import FillMode
from scipy.spatial.transform import Rotation

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'backend'))
# Add rerun_urdf to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'ambient' / 'rerun_urdf'))
from api import ControllerTrackingClient, PoseData, ControllerState

try:
    from .camera_manager import CameraManager, CameraConfig
    from .lerobot_writer import LeRobotFormatWriter, FeatureDefinition
    from .gripper_detector import GripperDetector
except ImportError:
    from camera_manager import CameraManager, CameraConfig
    from lerobot_writer import LeRobotFormatWriter, FeatureDefinition
    from gripper_detector import GripperDetector


def load_glb_mesh(filepath: str, scale: float = 0.01) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a GLB/glTF file and return vertices, triangle indices, and vertex normals.
    Uses trimesh for parsing.

    Args:
        filepath: Path to GLB file
        scale: Scale factor to apply (default 0.01 to convert cm to meters)
    """
    import trimesh

    # Load the GLB file
    scene = trimesh.load(filepath)

    # If it's a scene with multiple meshes, combine them
    if isinstance(scene, trimesh.Scene):
        # Combine all meshes in the scene
        meshes = [geom for geom in scene.geometry.values() if isinstance(geom, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"No meshes found in {filepath}")
        mesh = trimesh.util.concatenate(meshes)
    else:
        mesh = scene

    # Apply scale (GLB is often in cm, we need meters)
    vertices = np.array(mesh.vertices, dtype=np.float32) * scale
    indices = np.array(mesh.faces, dtype=np.uint32)
    normals = np.array(mesh.vertex_normals, dtype=np.float32)

    return vertices, indices, normals


class LeRobotDatasetRecorder:
    """
    Records controller poses and camera observations to LeRobot Dataset format.

    Requires at least one camera. Always visualizes in Rerun.

    Controls:
        1-9 - Select task from list
        s   - Start recording episode
        e   - End episode (save)
        a   - Abort episode (discard)
        r   - Reset origin and robot to home pose
        q   - Quit
    """

    def __init__(
        self,
        repo_id: str,
        cameras: Dict[str, CameraConfig],  # Maps 'right' or 'left'/'right' to camera config
        bimanual: bool = False,
        server_url: str = 'https://localhost:8000',
        fps: int = 30,
        tasks: Optional[List[str]] = None,
        output_dir: str = './datasets',
        urdf_path: Optional[str] = None,
        gripper_calibration_path: Optional[str] = None,
        gripper_camera: Optional[str] = None,
    ):
        self.repo_id = repo_id
        self.server_url = server_url
        self.fps = fps
        self.cameras = cameras  # Dict mapping hand -> CameraConfig
        self.bimanual = bimanual
        self.tasks = tasks or ["UMI teleoperation demonstration"]
        self.selected_task_idx = 0
        self.output_dir = Path(output_dir)
        self.urdf_path = urdf_path

        # Gripper state detection
        self.gripper_detector: Optional[GripperDetector] = None
        self.gripper_camera = gripper_camera
        self.latest_gripper_state: float = 0.5  # Default mid-state

        if gripper_calibration_path:
            try:
                self.gripper_detector = GripperDetector(gripper_calibration_path)
                print(f"Gripper detection enabled: {gripper_calibration_path}")
            except Exception as e:
                print(f"Warning: Could not load gripper calibration: {e}")

        # Robot IK state (initialized in _setup_robot_ik if urdf_path provided)
        self.robot_viz = None  # RerunURDFVisualizer
        self.ik_solver = None  # placo.KinematicsSolver
        self.ik_robot = None   # placo.RobotWrapper
        self.ik_task = None    # Frame task
        # 6 revolute arm joints (no gripper revolute in this URDF)
        self.arm_joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex',
                                'elbow_roll', 'wrist_flex', 'wrist_roll']
        # For dataset: arm joints + gripper value
        self.joint_names = self.arm_joint_names + ['gripper']
        self.latest_joint_positions: Optional[np.ndarray] = None

        # Build camera list for CameraManager
        camera_list = list(cameras.values())

        # Components
        self.client = ControllerTrackingClient(server_url)
        self.camera_manager = CameraManager(camera_list)
        self.writer: Optional[LeRobotFormatWriter] = None

        # State
        self.latest_pose: Optional[PoseData] = None
        self.is_recording = False
        self.episode_count = 0
        self.episode_frame_count = 0
        self.frame_count = 0
        self.start_time = time.time()
        self.frame_interval = 1.0 / fps

        # Keyboard control
        self._pending_action: Optional[str] = None
        self._pending_task_select: Optional[int] = None
        self._quit_requested = False
        self._lock = threading.Lock()

        # Relative control state (XLeVR-style)
        self.origin_position: Optional[np.ndarray] = None  # Origin position when recording started
        self.origin_quaternion: Optional[np.ndarray] = None  # Origin quaternion when recording started
        self.current_robot_pose: Optional[np.ndarray] = None  # Current robot end-effector pose (4x4)

        # Scaling factors (VR motion to robot motion)
        self.vr_to_robot_pos_scale = 1.0  # Position scaling
        self.vr_to_robot_ori_scale = 1.0  # Orientation scaling

        # Load controller meshes (separate for left and right)
        self.controller_meshes: Dict[str, Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
            'left': None,
            'right': None
        }
        mesh_dir = Path(__file__).parent / 'controller_mesh'
        for hand in ['left', 'right']:
            mesh_path = mesh_dir / f'quest-touch-plus-{hand}.glb'
            if mesh_path.exists():
                try:
                    vertices, indices, normals = load_glb_mesh(str(mesh_path))
                    self.controller_meshes[hand] = (vertices, indices, normals)
                    print(f"Loaded {hand} controller mesh: {len(indices)} triangles")
                except Exception as e:
                    print(f"Warning: Could not load {hand} controller mesh: {e}")


        # === Coordinate Frame Transforms ===
        # Frame hierarchy: tracker -> normalized -> {mesh, camera, tool}

        # Tracker to normalized: converts gripSpace (Z-wrist, Y-head) to (Z-up, Y-front)
        self.T_tracker_to_normalized = np.eye(4)
        self.T_tracker_to_normalized[:3, :3] = Rotation.from_euler('x', -125, degrees=True).as_matrix()

        # Mesh alignment within normalized frame
        self.T_normalized_to_mesh = np.eye(4)
        self.T_normalized_to_mesh[:3, :3] = Rotation.from_euler('xz', (180, 180), degrees=True).as_matrix()

        # Camera offset relative to normalized frame
        self.T_normalized_to_camera = np.eye(4)
        self.T_normalized_to_camera[:3, 3] = [0, 0.07, 0.03]
        self.T_normalized_to_camera[:3, :3] = Rotation.from_euler('x', -120, degrees=True).as_matrix()

        # Tool frame offset relative to normalized frame
        self.T_normalized_to_tool = np.eye(4)
        self.T_normalized_to_tool[:3, 3] = [0, 0.15, -0.03]

        # === Robot Frame Transforms ===

        # Robot base rotation (180° around Z so robot faces user)
        self.robot_base_rotation = Rotation.from_euler('z', -90, degrees=True).as_matrix()

        # Tool-to-gripper rotation: swaps X↔Y axes to align controller with robot gripper
        self.tool_to_gripper_rotation = np.array([
            [0, 1, 0],
            [-1, 0, 0],
            [0, 0, 1]
        ], dtype=np.float64)

        # Visualization offset to bring controller closer to robot level
        self.controller_viz_offset = np.array([0.0, 0.0, -0.8])

    @property
    def selected_task(self) -> str:
        """Get currently selected task description."""
        return self.tasks[self.selected_task_idx]

    async def setup(self):
        """Initialize cameras, dataset writer, and Rerun."""
        print(f"Setting up LeRobot Dataset Recorder...")
        print(f"  Repo ID: {self.repo_id}")
        print(f"  FPS: {self.fps}")
        print(f"  Mode: {'Bimanual' if self.bimanual else 'Single UMI'}")
        print(f"  Cameras: {len(self.cameras)}")

        # Connect cameras
        print(f"\nConnecting to cameras...")
        self.camera_manager.connect_all()
        if not self.camera_manager.is_connected:
            raise RuntimeError("No cameras connected! At least one camera is required.")

        # Create dataset writer
        self._create_dataset()

        # Initialize Rerun
        rr.init(f"umi_recorder", spawn=True)
        rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

        # Log world origin frame for reference
        rr.log(
            "world_origin",
            rr.Transform3D(axis_length=0.15),
            static=True
        )

        print("Rerun visualization started")

        # Initialize robot IK if URDF provided
        if self.urdf_path:
            self._setup_robot_ik()

        # Start keyboard listener
        self._start_keyboard_listener()

        # Check server connection
        try:
            status = await self.client.get_status()
            print(f"\nServer status: {status}")
            if not status.get('has_pose_data'):
                print("Warning: No pose data available yet. Make sure the Quest is connected.")
        except Exception as e:
            print(f"Warning: Could not connect to server: {e}")

        # Print controls
        self._print_controls()

    def _print_controls(self):
        """Print available controls and tasks."""
        print("\n" + "=" * 50)
        print("CONTROLS:")
        print("  [1-9] Select task")
        print("  [s]   Start recording episode")
        print("  [e]   End episode (save)")
        print("  [a]   Abort episode (discard)")
        print("  [r]   Reset origin (return robot to home pose)")
        print("  [q]   Quit")
        print("\nTASKS:")
        for i, task in enumerate(self.tasks[:9], 1):
            marker = ">>>" if i - 1 == self.selected_task_idx else "   "
            print(f"  {marker} [{i}] {task}")
        print("=" * 50 + "\n")

    def _create_dataset(self):
        """Create LeRobot dataset writer."""
        features = {}

        if self.urdf_path:
            # Robot mode: 7 joint positions
            features["action"] = FeatureDefinition(
                dtype="float32",
                shape=(7,),
                names={"axes": self.joint_names},
            )
            # Observation state: current joint positions (7D)
            features["observation.state"] = FeatureDefinition(
                dtype="float32",
                shape=(7,),
                names={"axes": self.joint_names},
            )
        else:
            # Controller mode: 14D controller poses
            features["action"] = FeatureDefinition(
                dtype="float32",
                shape=(14,),
                names={"axes": ["left_x", "left_y", "left_z", "left_qx", "left_qy", "left_qz", "left_qw",
                              "right_x", "right_y", "right_z", "right_qx", "right_qy", "right_qz", "right_qw"]},
            )
            # Observation state: gripper values (2D)
            features["observation.state"] = FeatureDefinition(
                dtype="float32",
                shape=(2,),
                names={"axes": ["gripper_left", "gripper_right"]},
            )

        # Camera observations
        for hand, config in self.cameras.items():
            features[f"observation.images.{config.name}"] = FeatureDefinition(
                dtype="video",
                shape=(config.height, config.width, 3),
            )

        dataset_name = self.repo_id.split('/')[-1]
        timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        dataset_path = self.output_dir / timestamp / dataset_name
        print(f"\nCreating dataset at: {dataset_path}")

        self.writer = LeRobotFormatWriter(
            output_dir=str(dataset_path),
            fps=self.fps,
            features=features,
            robot_type="umi",
            repo_id=self.repo_id,
        )

    def _start_keyboard_listener(self):
        """Start keyboard listener for episode control."""
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
                                self._pending_action = 'stop'
                            elif char == 'a':
                                self._pending_action = 'abort'
                            elif char == 'r':
                                self._pending_action = 'reset'
                            elif char == 'q':
                                self._quit_requested = True
                            elif char in '123456789':
                                self._pending_task_select = int(char) - 1
                except Exception:
                    pass

            listener = keyboard.Listener(on_press=on_press)
            listener.start()
        except ImportError:
            print("Warning: pynput not installed. Install with: pip install pynput")

    def _process_keyboard(self):
        """Process pending keyboard actions."""
        with self._lock:
            action = self._pending_action
            self._pending_action = None
            task_select = self._pending_task_select
            self._pending_task_select = None

        # Handle task selection
        if task_select is not None and task_select < len(self.tasks):
            self.selected_task_idx = task_select
            print(f">>> Selected task [{task_select + 1}]: {self.selected_task}")

        # Handle recording actions
        if action == 'start' and not self.is_recording:
            self.is_recording = True
            self.episode_count += 1
            self.episode_frame_count = 0
            self.writer.start_episode()
            print(f"\n>>> Recording episode {self.episode_count} with task: {self.selected_task}")

        elif action == 'stop' and self.is_recording:
            self.is_recording = False
            if self.episode_frame_count > 0:
                self.writer.end_episode(task=self.selected_task)
                print(f">>> Episode {self.episode_count} saved ({self.episode_frame_count} frames)")
            else:
                self.writer.clear_episode_buffer()
                print(">>> No frames recorded, episode discarded")

        elif action == 'abort' and self.is_recording:
            self.is_recording = False
            self.writer.clear_episode_buffer()
            self.episode_count -= 1  # Don't count aborted episode
            print(f">>> Episode aborted ({self.episode_frame_count} frames discarded)")

        elif action == 'reset':
            # Reset origin to current controller position
            self.origin_position = None
            self.origin_quaternion = None
            # Reset robot to home pose (all joints to zero)
            if self.ik_robot:
                # Set all joints to zero
                for name in self.arm_joint_names:
                    self.ik_robot.set_joint(name, 0.0)
                self.ik_robot.update_kinematics()
                # Get home gripper_frame pose for relative control
                self.current_robot_pose = self.ik_robot.get_T_world_frame("gripper").copy()
                # Update visualization
                if self.robot_viz:
                    positions_dict = {name: 0.0 for name in self.arm_joint_names}
                    self.robot_viz.set_joint_positions(positions_dict)
            print(">>> Origin reset - robot returned to home pose")

    def _setup_robot_ik(self):
        """Initialize URDF visualizer and IK solver."""
        import placo
        from umi.utils.rerun_urdf import RerunURDFVisualizer

        print(f"\nInitializing robot IK with URDF: {self.urdf_path}")

        # Load URDF visualizer (uses existing Rerun instance)
        self.robot_viz = RerunURDFVisualizer(self.urdf_path)
        self.robot_viz._load_urdf_to_rerun()

        # Debug output
        print(f"  URDF directory: {self.robot_viz.urdf_dir}")
        print(f"  Available joints: {self.robot_viz.joint_names}")

        # Log robot base transform (rotated 180° around Z so robot faces user)
        robot_rot = Rotation.from_euler('z', 180, degrees=True).as_quat()
        rr.log(
            "base_so101_v2",
            rr.Transform3D(
                translation=[0.0, 0.0, 0.0],
                rotation=rr.Quaternion(xyzw=robot_rot.tolist()),
            ),
        )

        # Initialize placo IK solver
        self.ik_robot = placo.RobotWrapper(self.urdf_path, placo.Flags.ignore_collisions)
        self.ik_solver = placo.KinematicsSolver(self.ik_robot)
        self.ik_solver.mask_fbase(True)
        self.ik_solver.dt = 1.0 / self.fps

        # Add frame task for gripper
        self.ik_task = self.ik_solver.add_frame_task("gripper", np.eye(4))
        self.ik_task.configure("gripper", "soft", 1.0, 0.2)
        self.ik_solver.add_regularization_task(1e-4)

        # Set initial joint positions to zero (only arm joints)
        for name in self.arm_joint_names:
            self.ik_robot.set_joint(name, 0.0)
        self.ik_robot.update_kinematics()

        # Initialize current robot pose to home position (for relative control)
        self.current_robot_pose = self.ik_robot.get_T_world_frame("gripper").copy()

        print(f"  Robot arm joints: {self.arm_joint_names}")
        print("  Robot IK ready")

    def _compute_relative_rotvec(self, origin_quat: np.ndarray, current_quat: np.ndarray) -> np.ndarray:
        """Compute relative rotation as rotation vector (axis-angle)."""
        # Normalize quaternions
        origin_quat = origin_quat / np.linalg.norm(origin_quat)
        current_quat = current_quat / np.linalg.norm(current_quat)

        # Handle quaternion double cover (q and -q are same rotation)
        if np.dot(origin_quat, current_quat) < 0:
            current_quat = -current_quat

        # Relative quaternion: q_rel = conj(q_origin) * q_current
        origin_rot = Rotation.from_quat(origin_quat)
        current_rot = Rotation.from_quat(current_quat)
        relative_rot = origin_rot.inv() * current_rot

        return relative_rot.as_rotvec()

    def _solve_ik(self, controller: ControllerState, gripper_value: float = 0.0) -> Optional[np.ndarray]:
        """Solve IK using incremental relative control.

        First controller data becomes origin. Subsequent deltas are accumulated
        and transformed to robot frame coordinates.
        """
        if self.ik_solver is None:
            return None

        # Compute tool_frame world pose from controller
        raw_pos = np.array(controller.position)
        raw_quat = np.array(controller.orientation)

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
                self.current_robot_pose = self.ik_robot.get_T_world_frame("gripper").copy()
            print(f">>> Origin captured at pos={self.origin_position}")

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
        delta_rot_robot = self.tool_to_gripper_rotation @ delta_rot @ self.tool_to_gripper_rotation.T

        # Apply to current robot pose
        self.current_robot_pose[:3, 3] += delta_pos_robot
        self.current_robot_pose[:3, :3] = self.current_robot_pose[:3, :3] @ delta_rot_robot

        # Solve IK
        self.ik_task.T_world_frame = self.current_robot_pose
        self.ik_solver.solve(True)
        self.ik_robot.update_kinematics()

        # Return joint positions with gripper value appended
        joint_positions = np.array([
            self.ik_robot.get_joint(name) for name in self.arm_joint_names
        ], dtype=np.float32)
        return np.append(joint_positions, gripper_value)

    def on_pose_update(self, pose: PoseData):
        """Callback for pose updates from the server."""
        self.latest_pose = pose

    def _visualize_controller(self, hand: str, controller: ControllerState):
        """Visualize controller with frame hierarchy: normalized -> {mesh, camera, tool}."""
        # Apply visualization offset to bring controller closer to robot level
        pos = np.array(controller.position) + self.controller_viz_offset
        quat = np.array(controller.orientation)

        # Build tracker transform and compute normalized frame
        T_world_tracker = np.eye(4)
        T_world_tracker[:3, :3] = Rotation.from_quat(quat).as_matrix()
        T_world_tracker[:3, 3] = pos
        T_world_normalized = T_world_tracker @ self.T_tracker_to_normalized
        norm_pos = T_world_normalized[:3, 3]
        norm_quat = Rotation.from_matrix(T_world_normalized[:3, :3]).as_quat()

        rr.log(
            f'controller_normalized/{hand}',
            rr.Transform3D(
                translation=norm_pos,
                rotation=rr.Quaternion(xyzw=norm_quat),
                axis_length=0.06
            ),
        )

        # 3. Log mesh as child of normalized frame
        mesh_data = self.controller_meshes.get(hand)
        if mesh_data is not None:
            vertices, indices, normals = mesh_data

            color = [200, 80, 80] if hand == "left" else [80, 80, 200]
            pressed = any(v.get("pressed", False) for v in controller.buttons.values())
            if pressed:
                color = [255, 150, 150] if hand == "left" else [150, 150, 255]

            # Mesh transform relative to normalized
            mesh_quat = Rotation.from_matrix(self.T_normalized_to_mesh[:3, :3]).as_quat()
            mesh_offset = self.T_normalized_to_mesh[:3, 3]

            rr.log(
                f'controller_normalized/{hand}/mesh',
                rr.Transform3D(
                    translation=mesh_offset,
                    rotation=rr.Quaternion(xyzw=mesh_quat),
                ),
            )
            rr.log(
                f'controller_normalized/{hand}/mesh',
                rr.Mesh3D(
                    vertex_positions=vertices,
                    triangle_indices=indices,
                    vertex_normals=normals,
                    vertex_colors=np.tile(color, (len(vertices), 1)),
                ),
            )
        else:
            # Fallback: sphere if no mesh
            color = [255, 100, 100] if hand == "left" else [100, 100, 255]
            pressed = any(v.get("pressed", False) for v in controller.buttons.values())
            if pressed:
                color = [255, 200, 200] if hand == "left" else [200, 200, 255]

            btn_value = list(controller.buttons.values())[0].get("value", 0) if controller.buttons else 0
            rr.log(
                f'controller_normalized/{hand}/sphere',
                rr.Ellipsoids3D(
                    radii=[0.05 * btn_value + 0.01],
                    fill_mode=FillMode.Solid,
                    colors=color,
                ),
            )

        # 4. Log camera frame as child of normalized
        cam_quat = Rotation.from_matrix(self.T_normalized_to_camera[:3, :3]).as_quat()
        cam_offset = self.T_normalized_to_camera[:3, 3]

        rr.log(
            f'controller_normalized/{hand}/camera_frame',
            rr.Transform3D(
                translation=cam_offset,
                rotation=rr.Quaternion(xyzw=cam_quat),
                axis_length=0.04
            ),
        )

        # 5. Log tool frame as child of normalized
        tool_quat = Rotation.from_matrix(self.T_normalized_to_tool[:3, :3]).as_quat()
        tool_offset = self.T_normalized_to_tool[:3, 3]

        rr.log(
            f'controller_normalized/{hand}/tool_frame',
            rr.Transform3D(
                translation=tool_offset,
                rotation=rr.Quaternion(xyzw=tool_quat),
                axis_length=0.04
            ),
        )

    def _visualize_camera_on_controller(self, hand: str, camera_name: str, image: np.ndarray):
        """Attach camera feed to the camera_frame coordinate system."""
        # Camera is attached to the camera_frame under controller_normalized
        entity_path = f'controller_normalized/{hand}/camera_frame'

        # Camera intrinsics (simple pinhole)
        height, width = image.shape[:2]
        focal_length = width  # Approximate focal length
        rr.log(
            f'{entity_path}/image',
            rr.Pinhole(
                focal_length=focal_length,
                width=width,
                height=height,
                image_plane_distance=0.3
            ),
        )

        # Camera image
        rr.log(f'{entity_path}/image', rr.Image(image))

    def _log_controls_overlay(self):
        """Log controls and status overlay to Rerun."""
        status = "RECORDING" if self.is_recording else "IDLE"
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0

        # Build task list with selection marker
        task_lines = []
        for i, task in enumerate(self.tasks[:9], 1):
            marker = ">>" if i - 1 == self.selected_task_idx else "  "
            task_lines.append(f"{marker} [{i}] {task}")

        controls_text = f"""## Controls
- **[1-9]** Select task
- **[s]** Start recording
- **[e]** End episode (save)
- **[a]** Abort episode (discard)
- **[r]** Reset (home pose)
- **[q]** Quit

## Tasks
{chr(10).join(task_lines)}

## Status
- **Mode**: {'Bimanual' if self.bimanual else 'Single UMI'}
- **State**: {status}
- **Episode**: {self.episode_count}
- **Frames**: {self.episode_frame_count}
- **FPS**: {fps:.1f}
"""
        rr.log('ui/controls', rr.TextDocument(controls_text, media_type=rr.MediaType.MARKDOWN))

    def _log_episode_info(self):
        """Log episode information including task label."""
        if self.is_recording:
            # Log task as prominent label for this episode
            rr.log('episode/task', rr.TextDocument(
                f"# Episode {self.episode_count}\n\n**Task:** {self.selected_task}",
                media_type=rr.MediaType.MARKDOWN
            ))

    def _visualize_frame(self, pose: Optional[PoseData], camera_frames: Dict[str, np.ndarray]):
        """Visualize controllers and camera frames in Rerun."""
        # Always use global frame count for continuous visualization (Rerun 0.23+ API)
        rr.set_time("global_frame", sequence=self.frame_count)

        # Episode-specific timeline only during recording
        if self.is_recording:
            rr.set_time("episode", sequence=self.episode_count)
            rr.set_time("episode_frame", sequence=self.episode_frame_count)

        # Robot mode: always visualize robot, solve IK if controller data available
        if self.urdf_path and not self.bimanual and self.robot_viz:
            if pose:
                # Get controller for the selected hand
                controller = pose.right if 'right' in self.cameras else pose.left
                if controller:
                    # Get gripper value from trigger
                    gripper_value = 0.0
                    if controller.buttons:
                        btn = controller.buttons.get('0', controller.buttons.get(0, {}))
                        gripper_value = float(btn.get('value', 0.0)) if isinstance(btn, dict) else 0.0

                    # Solve IK
                    joint_positions = self._solve_ik(controller, gripper_value)
                    if joint_positions is not None:
                        self.latest_joint_positions = joint_positions
                        # Update robot visualization (only arm joints, exclude gripper)
                        positions_dict = dict(zip(self.arm_joint_names, joint_positions[:-1]))
                        self.robot_viz.set_joint_positions(positions_dict)
            else:
                # No controller data - show robot at initial position
                positions_dict = {name: 0.0 for name in self.arm_joint_names}
                self.robot_viz.set_joint_positions(positions_dict)

        # Visualize controllers (still show them even in robot mode)
        if pose:
            if 'left' in self.cameras and pose.left:
                self._visualize_controller('left', pose.left)
            if 'right' in self.cameras and pose.right:
                self._visualize_controller('right', pose.right)

        # Camera images attached to controllers
        for hand, config in self.cameras.items():
            if config.name in camera_frames:
                image = camera_frames[config.name]
                self._visualize_camera_on_controller(hand, config.name, image)

        # Also log cameras in a separate 2D panel for easy viewing
        for camera_name, image in camera_frames.items():
            rr.log(f'cameras/{camera_name}', rr.Image(image))

        # Log gripper state as scalar time series
        if self.gripper_detector is not None:
            rr.log("gripper/state", rr.Scalars(self.latest_gripper_state))

        # Controls overlay and episode info
        self._log_controls_overlay()
        self._log_episode_info()

    def _record_frame(self, pose: Optional[PoseData], camera_frames: Dict[str, np.ndarray]):
        """Record a frame to the dataset."""
        frame_data = {}

        if self.urdf_path and self.latest_joint_positions is not None:
            # Robot mode: log joint positions
            frame_data["action"] = self.latest_joint_positions.copy()
            frame_data["observation.state"] = self.latest_joint_positions.copy()
        else:
            # Controller mode: log controller poses
            def controller_to_action(ctrl):
                if ctrl is None:
                    return np.zeros(7, dtype=np.float32)
                return np.array([*ctrl.position, *ctrl.orientation], dtype=np.float32)

            left = controller_to_action(pose.left if pose else None)
            right = controller_to_action(pose.right if pose else None)
            frame_data["action"] = np.concatenate([left, right])

            # Gripper values
            def get_gripper(ctrl):
                if ctrl is None or not ctrl.buttons:
                    return 0.0
                btn = ctrl.buttons.get('0', ctrl.buttons.get(0, {}))
                return float(btn.get('value', 0.0)) if isinstance(btn, dict) else 0.0

            frame_data["observation.state"] = np.array([
                get_gripper(pose.left if pose else None),
                get_gripper(pose.right if pose else None)
            ], dtype=np.float32)

        # Camera images
        for camera_name, image in camera_frames.items():
            frame_data[f"observation.images.{camera_name}"] = image

        # Vision-based gripper state
        if self.gripper_detector is not None:
            frame_data["observation.gripper_state"] = np.array(
                [self.latest_gripper_state], dtype=np.float32
            )

        self.writer.add_frame(frame_data)
        self.episode_frame_count += 1

    async def run(self):
        """Main recording loop."""
        print("Starting recording loop...")

        # Start pose streaming
        pose_task = asyncio.create_task(
            self.client.connect_stream(self.on_pose_update)
        )

        try:
            while not self._quit_requested:
                loop_start = time.time()

                # Process keyboard
                self._process_keyboard()

                # Capture cameras
                camera_frames = self.camera_manager.capture_all()

                # Detect gripper state from camera frame
                if self.gripper_detector is not None and camera_frames:
                    # Use specified gripper camera or first available
                    gripper_frame = None
                    if self.gripper_camera and self.gripper_camera in camera_frames:
                        gripper_frame = camera_frames[self.gripper_camera]
                    else:
                        gripper_frame = next(iter(camera_frames.values()), None)

                    if gripper_frame is not None:
                        result = self.gripper_detector.get_gripper_state(gripper_frame)
                        self.latest_gripper_state = result.state

                # Visualize
                self._visualize_frame(self.latest_pose, camera_frames)

                # Record if recording
                if self.is_recording:
                    self._record_frame(self.latest_pose, camera_frames)

                self.frame_count += 1

                # Progress
                if self.frame_count % 100 == 0:
                    elapsed = time.time() - self.start_time
                    fps = self.frame_count / elapsed if elapsed > 0 else 0
                    status = "RECORDING" if self.is_recording else "IDLE"
                    print(f"[{status}] Frame {self.frame_count}, FPS: {fps:.1f}")

                # Maintain FPS
                elapsed = time.time() - loop_start
                if elapsed < self.frame_interval:
                    await asyncio.sleep(self.frame_interval - elapsed)

        except KeyboardInterrupt:
            print("\nInterrupted")
        finally:
            pose_task.cancel()
            try:
                await pose_task
            except asyncio.CancelledError:
                pass
            await self.client.close()
            self.camera_manager.disconnect_all()

    def finalize(self):
        """Finalize dataset."""
        if self.writer:
            self.writer.finalize()

        elapsed = time.time() - self.start_time
        print(f"\nRecording complete:")
        print(f"  Episodes: {self.episode_count}")
        print(f"  Total frames: {self.frame_count}")
        print(f"  Duration: {elapsed:.1f}s")


def load_config(filepath: str) -> dict:
    """Load configuration from a YAML file."""
    import yaml
    with open(filepath) as f:
        config = yaml.safe_load(f) or {}
    return config


def load_tasks_from_yaml(filepath: str) -> List[str]:
    """Load task descriptions from a YAML file."""
    import yaml
    with open(filepath) as f:
        data = yaml.safe_load(f)

    tasks = data.get('tasks', [])
    if not tasks:
        raise ValueError(f"No tasks found in {filepath}. Expected 'tasks' key with list of task descriptions.")

    return tasks[:9]  # Max 9 tasks (keys 1-9)


def setup_camera_interactive(config_path: str) -> bool:
    """
    Interactive camera selection and configuration.

    Scans for available cameras, lets user select one,
    and saves the selection to the config file.

    Returns:
        True if setup was successful, False otherwise
    """
    import yaml

    print("\n=== Camera Setup ===\n")
    print("Scanning for available cameras...")

    available = CameraManager.list_available_cameras()

    if not available:
        print("No cameras found! Please connect a USB camera and try again.")
        return False

    # Get camera info (resolution) for each
    camera_info = []
    for idx in available:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            camera_info.append((idx, width, height))
            print(f"  [{len(camera_info)}] Camera {idx}: {width}x{height}")

    if not camera_info:
        print("Could not read from any camera!")
        return False

    # Let user select camera
    print()
    while True:
        try:
            choice = input(f"Select camera [1-{len(camera_info)}]: ").strip()
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(camera_info):
                break
            print(f"Please enter a number between 1 and {len(camera_info)}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nCancelled")
            return False

    selected = camera_info[choice_idx]
    camera_idx, width, height = selected

    # Ask for camera name
    print()
    name = input("Camera name (e.g., 'wrist', 'overhead') [wrist]: ").strip()
    if not name:
        name = "wrist"

    # Ask for resolution (use detected as default)
    print(f"\nResolution detected: {width}x{height}")
    res_input = input(f"Use this resolution? [Y/n]: ").strip().lower()

    if res_input == 'n':
        print("Common resolutions: 640x480, 1280x720, 1920x1080")
        res = input("Enter resolution (WxH): ").strip()
        try:
            w, h = res.lower().split('x')
            width, height = int(w), int(h)
        except ValueError:
            print(f"Invalid format, using {width}x{height}")

    # Build camera string
    camera_string = f"{camera_idx}:{name}:{width}x{height}"
    print(f"\nCamera config: {camera_string}")

    # Load existing config or create new one
    config = {}
    config_file = Path(config_path)

    if config_file.exists():
        try:
            with open(config_file) as f:
                config = yaml.safe_load(f) or {}
            print(f"\nUpdating existing config: {config_path}")
        except Exception as e:
            print(f"Warning: Could not read existing config: {e}")
    else:
        print(f"\nCreating new config: {config_path}")

    # Update camera setting
    config['camera'] = camera_string

    # Ensure other required fields have defaults
    if 'repo_id' not in config:
        config['repo_id'] = 'user/my_dataset'
    if 'fps' not in config:
        config['fps'] = 30
    if 'output_dir' not in config:
        config['output_dir'] = './datasets'
    if 'server' not in config:
        config['server'] = 'https://localhost:8000'
    if 'hand' not in config:
        config['hand'] = 'right'

    # Write config
    try:
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"\n Config saved to: {config_path}")
        print(f"\nYou can now run:")
        print(f"  python -m umi.lerobot_recorder --config {config_path}")
        return True
    except Exception as e:
        print(f"Error saving config: {e}")
        return False


async def main():
    parser = argparse.ArgumentParser(
        description='Record UMI teleoperation to LeRobot dataset format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using a config file (recommended)
  python -m umi.lerobot_recorder --config config.yaml

  # Single UMI mode (camera on right controller by default)
  python -m umi.lerobot_recorder --repo-id user/dataset --camera 0:wrist:640x480

  # CLI args override config file values
  python -m umi.lerobot_recorder --config config.yaml --fps 60

  # List available cameras
  python -m umi.lerobot_recorder --list-cameras
        """
    )

    parser.add_argument('--config', help='Path to YAML config file')
    parser.add_argument('--repo-id', help='Dataset name (required for recording)')
    parser.add_argument('--server', help='Server URL')
    parser.add_argument('--fps', type=int, help='Recording FPS')
    parser.add_argument('--tasks-file', help='YAML file with task descriptions (deprecated, use config)')
    parser.add_argument('--output-dir', help='Output directory')
    parser.add_argument('--list-cameras', action='store_true', help='List cameras and exit')
    parser.add_argument('--setup-camera', action='store_true',
                        help='Interactive camera selection and save to config')
    parser.add_argument('--urdf', help='Path to robot URDF for IK visualization (non-bimanual only)')

    # Gripper detection arguments
    parser.add_argument('--gripper-calibration', help='Path to gripper calibration YAML file')
    parser.add_argument('--gripper-camera', help='Camera name for gripper detection (default: first camera)')

    # Camera arguments
    parser.add_argument('--camera', help='Camera for single UMI mode: INDEX:NAME:WxH')
    parser.add_argument('--hand', choices=['left', 'right'],
                        help='Controller for camera in single mode (default: right)')
    parser.add_argument('--bimanual', action='store_true', help='Enable bimanual mode')
    parser.add_argument('--camera-left', help='Left camera for bimanual mode: INDEX:NAME:WxH')
    parser.add_argument('--camera-right', help='Right camera for bimanual mode: INDEX:NAME:WxH')

    args = parser.parse_args()

    # Load config file if provided
    config = {}
    if args.config:
        try:
            config = load_config(args.config)
            print(f"Loaded config from: {args.config}")
        except Exception as e:
            parser.error(f"Failed to load config file: {e}")

    # Helper to get value: CLI arg > config > default
    def get_val(cli_val, config_key, default=None):
        if cli_val is not None:
            return cli_val
        return config.get(config_key, default)

    # Merge CLI args with config (CLI overrides config)
    repo_id = get_val(args.repo_id, 'repo_id')
    server = get_val(args.server, 'server', 'https://localhost:8000')
    fps = get_val(args.fps, 'fps', 30)
    output_dir = get_val(args.output_dir, 'output_dir', './datasets')
    urdf = get_val(args.urdf, 'urdf')
    gripper_calibration = get_val(args.gripper_calibration, 'gripper_calibration')
    gripper_camera = get_val(args.gripper_camera, 'gripper_camera')
    camera = get_val(args.camera, 'camera')
    hand = get_val(args.hand, 'hand', 'right')
    bimanual = args.bimanual or config.get('bimanual', False)
    camera_left = get_val(args.camera_left, 'camera_left')
    camera_right = get_val(args.camera_right, 'camera_right')

    # Tasks: from config, or from separate tasks-file (deprecated)
    tasks = config.get('tasks', None)
    if args.tasks_file:
        try:
            tasks = load_tasks_from_yaml(args.tasks_file)
            print(f"Loaded {len(tasks)} tasks from {args.tasks_file}")
        except Exception as e:
            parser.error(f"Failed to load tasks file: {e}")
    elif tasks:
        tasks = tasks[:9]  # Max 9 tasks
        print(f"Loaded {len(tasks)} tasks from config")

    if args.list_cameras:
        print("Discovering cameras...")
        available = CameraManager.list_available_cameras()
        print(f"Found: {available}" if available else "No cameras found!")
        print("\nUsage:")
        print("  Single mode:   --camera INDEX:NAME:WxH")
        print("  Bimanual mode: --bimanual --camera-left INDEX:NAME:WxH --camera-right INDEX:NAME:WxH")
        return

    if args.setup_camera:
        config_path = args.config or "config.yaml"
        setup_camera_interactive(config_path)
        return

    if not repo_id:
        parser.error("--repo-id is required (or set repo_id in config)")

    # Validate camera arguments
    cameras: Dict[str, CameraConfig] = {}

    if bimanual:
        # Bimanual mode: require camera_left and camera_right
        if camera:
            parser.error("In bimanual mode, use camera_left and camera_right instead of camera")
        if not camera_left or not camera_right:
            parser.error("Bimanual mode requires both camera_left and camera_right")

        try:
            cameras['left'] = CameraConfig.from_string(camera_left)
            cameras['right'] = CameraConfig.from_string(camera_right)
        except ValueError as e:
            parser.error(f"Invalid camera config: {e}")
    else:
        # Single UMI mode: require camera
        if camera_left or camera_right:
            parser.error("camera_left and camera_right require bimanual mode")
        if not camera:
            parser.error("Single mode requires camera (or use bimanual for bimanual mode)")

        try:
            # Use selected hand (default: right)
            cameras[hand] = CameraConfig.from_string(camera)
        except ValueError as e:
            parser.error(f"Invalid camera config: {e}")

    # Validate URDF argument
    if urdf and bimanual:
        parser.error("urdf is only supported in non-bimanual mode")

    recorder = LeRobotDatasetRecorder(
        repo_id=repo_id,
        cameras=cameras,
        bimanual=bimanual,
        server_url=server,
        fps=fps,
        tasks=tasks,
        output_dir=output_dir,
        urdf_path=urdf,
        gripper_calibration_path=gripper_calibration,
        gripper_camera=gripper_camera,
    )

    try:
        await recorder.setup()
        await recorder.run()
    finally:
        recorder.finalize()


def cli():
    """Entry point for the CLI (wraps async main)"""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print('\nExiting...')


if __name__ == '__main__':
    cli()
