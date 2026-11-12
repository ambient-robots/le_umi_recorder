"""
Data schemas for the UMI recording pipeline.

Defines Pydantic models for configuration and PyArrow schemas for data storage.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pyarrow as pa


# =============================================================================
# PyArrow Schemas for Parquet Files
# =============================================================================

# Stage 1: Raw trajectory data (50Hz)
TRAJECTORY_SCHEMA = pa.schema([
    pa.field("timestamp_ns", pa.int64()),           # Nanoseconds since epoch
    pa.field("frame_index", pa.int64()),            # Sequential frame counter
    pa.field("position_x", pa.float64()),           # Controller position (meters)
    pa.field("position_y", pa.float64()),
    pa.field("position_z", pa.float64()),
    pa.field("orientation_x", pa.float64()),        # Quaternion XYZW
    pa.field("orientation_y", pa.float64()),
    pa.field("orientation_z", pa.float64()),
    pa.field("orientation_w", pa.float64()),
    pa.field("button_trigger", pa.float32()),       # Trigger value 0-1
    pa.field("button_grip", pa.float32()),          # Grip button value 0-1
    pa.field("controller_hand", pa.string()),       # "left" or "right"
])

# Stage 1: Video frame timestamps for sync
VIDEO_TIMESTAMPS_SCHEMA = pa.schema([
    pa.field("frame_index", pa.int64()),            # Video frame number
    pa.field("timestamp_ns", pa.int64()),           # Capture timestamp (ns)
    pa.field("closest_trajectory_idx", pa.int64()), # Nearest trajectory frame
])

# Stage 2: Joint positions from IK solver
JOINTS_SCHEMA = pa.schema([
    pa.field("timestamp_ns", pa.int64()),           # Aligned to video timestamps
    pa.field("frame_index", pa.int64()),            # Video frame index
    pa.field("shoulder_pan", pa.float32()),         # Joint angles (radians)
    pa.field("shoulder_lift", pa.float32()),
    pa.field("elbow_flex", pa.float32()),
    pa.field("elbow_roll", pa.float32()),
    pa.field("wrist_flex", pa.float32()),
    pa.field("wrist_roll", pa.float32()),
    pa.field("ik_success", pa.bool_()),             # IK solver converged
    pa.field("ik_error", pa.float32()),             # Position error (meters)
])

# Stage 4: Gripper states from Gemini
GRIPPER_STATES_SCHEMA = pa.schema([
    pa.field("frame_index", pa.int64()),            # Video frame index
    pa.field("timestamp_ns", pa.int64()),
    pa.field("measured_state", pa.float32()),       # Actual gripper opening 0-100
    pa.field("commanded_state", pa.float32()),      # Intended gripper opening 0-100
    pa.field("confidence", pa.float32()),           # Detection confidence 0-1
    pa.field("annotation", pa.string()),            # Optional: "grasping", "releasing", etc.
])


# =============================================================================
# Dataclasses for Runtime Data
# =============================================================================

@dataclass
class TrajectoryFrame:
    """Single frame of trajectory data."""
    timestamp_ns: int
    frame_index: int
    position_x: float
    position_y: float
    position_z: float
    orientation_x: float
    orientation_y: float
    orientation_z: float
    orientation_w: float
    button_trigger: float
    button_grip: float
    controller_hand: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Arrow table."""
        return {
            "timestamp_ns": self.timestamp_ns,
            "frame_index": self.frame_index,
            "position_x": self.position_x,
            "position_y": self.position_y,
            "position_z": self.position_z,
            "orientation_x": self.orientation_x,
            "orientation_y": self.orientation_y,
            "orientation_z": self.orientation_z,
            "orientation_w": self.orientation_w,
            "button_trigger": self.button_trigger,
            "button_grip": self.button_grip,
            "controller_hand": self.controller_hand,
        }


@dataclass
class VideoTimestamp:
    """Video frame timestamp for synchronization."""
    frame_index: int
    timestamp_ns: int
    closest_trajectory_idx: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp_ns": self.timestamp_ns,
            "closest_trajectory_idx": self.closest_trajectory_idx,
        }


@dataclass
class JointState:
    """Joint positions from IK solver."""
    timestamp_ns: int
    frame_index: int
    shoulder_pan: float
    shoulder_lift: float
    elbow_flex: float
    elbow_roll: float
    wrist_flex: float
    wrist_roll: float
    ik_success: bool
    ik_error: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_ns": self.timestamp_ns,
            "frame_index": self.frame_index,
            "shoulder_pan": self.shoulder_pan,
            "shoulder_lift": self.shoulder_lift,
            "elbow_flex": self.elbow_flex,
            "elbow_roll": self.elbow_roll,
            "wrist_flex": self.wrist_flex,
            "wrist_roll": self.wrist_roll,
            "ik_success": self.ik_success,
            "ik_error": self.ik_error,
        }

    def to_array(self) -> List[float]:
        """Return joint angles as array (6 elements, no gripper)."""
        return [
            self.shoulder_pan,
            self.shoulder_lift,
            self.elbow_flex,
            self.elbow_roll,
            self.wrist_flex,
            self.wrist_roll,
        ]


@dataclass
class GripperState:
    """Gripper state from Gemini detection."""
    frame_index: int
    timestamp_ns: int
    measured_state: float      # 0-100: actual physical opening
    commanded_state: float     # 0-100: intended/target opening
    confidence: float          # 0-1: detection confidence
    annotation: str = ""       # Optional annotation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp_ns": self.timestamp_ns,
            "measured_state": self.measured_state,
            "commanded_state": self.commanded_state,
            "confidence": self.confidence,
            "annotation": self.annotation,
        }


@dataclass
class TaskPrompt:
    """Generated task prompt from Gemini."""
    task_description: str
    confidence: float
    model: str
    generation_timestamp: str
    video_summary: str = ""
    objects: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_description": self.task_description,
            "confidence": self.confidence,
            "model": self.model,
            "generation_timestamp": self.generation_timestamp,
            "video_summary": self.video_summary,
            "objects": self.objects,
            "actions": self.actions,
        }


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class StageMetadata:
    """Metadata for a pipeline stage output."""
    stage: int
    version: str
    created_at: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "version": self.version,
            "created_at": self.created_at,
            **self.extra,
        }


@dataclass
class CameraConfig:
    """Camera configuration for recording."""
    name: str
    index: int
    width: int = 640
    height: int = 480
    fps: int = 60


@dataclass
class Stage1Config:
    """Configuration for Stage 1: Raw Recording."""
    trajectory_fps: int = 50
    video_fps: int = 60
    camera: CameraConfig = field(default_factory=lambda: CameraConfig("wrist", 0))
    controller_hand: str = "right"
    server_url: str = "https://localhost:8000"


@dataclass
class Stage2Config:
    """Configuration for Stage 2: IK Solver."""
    urdf_path: str = ""
    solver_dt: float = 0.0167  # 60Hz
    regularization: float = 1e-4
    frame_name: str = "gripper_frame"
    joint_names: List[str] = field(default_factory=lambda: [
        "shoulder_pan", "shoulder_lift", "elbow_flex",
        "elbow_roll", "wrist_flex", "wrist_roll"
    ])


@dataclass
class Stage3Config:
    """Configuration for Stage 3: Task Prompt Generation."""
    model: str = "gemini-2.0-flash"
    prompt_template: str = "task_v1"
    api_key_env: str = "GEMINI_API_KEY"


@dataclass
class Stage4Config:
    """Configuration for Stage 4: Gripper Detection."""
    model: str = "gemini-2.0-flash"
    batch_size: int = 60  # Frames per batch
    sample_rate: int = 1  # Every Nth frame
    api_key_env: str = "GEMINI_API_KEY"


@dataclass
class AssemblyConfig:
    """Configuration for final dataset assembly."""
    output_format: str = "lerobot_v2.1"
    include_raw_poses: bool = False
    repo_id: str = ""
    fps: int = 60


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""
    stage1: Stage1Config = field(default_factory=Stage1Config)
    stage2: Stage2Config = field(default_factory=Stage2Config)
    stage3: Stage3Config = field(default_factory=Stage3Config)
    stage4: Stage4Config = field(default_factory=Stage4Config)
    assembly: AssemblyConfig = field(default_factory=AssemblyConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "PipelineConfig":
        """Load configuration from YAML file."""
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)

        config = cls()

        if "stage1" in data:
            s1 = data["stage1"]
            if "camera" in s1:
                cam = s1["camera"]
                config.stage1.camera = CameraConfig(
                    name=cam.get("name", "wrist"),
                    index=cam.get("index", 0),
                    width=cam.get("width", 640),
                    height=cam.get("height", 480),
                    fps=cam.get("fps", 60),
                )
            config.stage1.trajectory_fps = s1.get("trajectory_fps", 50)
            config.stage1.video_fps = s1.get("video_fps", 60)
            config.stage1.controller_hand = s1.get("controller_hand", "right")
            config.stage1.server_url = s1.get("server", "https://localhost:8000")

        if "stage2" in data:
            s2 = data["stage2"]
            config.stage2.urdf_path = s2.get("urdf", "")
            config.stage2.solver_dt = s2.get("solver_dt", 0.0167)
            config.stage2.regularization = s2.get("regularization", 1e-4)
            config.stage2.frame_name = s2.get("frame_name", "gripper_frame")

        if "stage3" in data:
            s3 = data["stage3"]
            config.stage3.model = s3.get("model", "gemini-2.0-flash")
            config.stage3.prompt_template = s3.get("prompt_template", "task_v1")

        if "stage4" in data:
            s4 = data["stage4"]
            config.stage4.model = s4.get("model", "gemini-2.0-flash")
            config.stage4.batch_size = s4.get("batch_size", 60)
            config.stage4.sample_rate = s4.get("sample_rate", 1)

        if "assembly" in data:
            asm = data["assembly"]
            config.assembly.output_format = asm.get("output_format", "lerobot_v2.1")
            config.assembly.include_raw_poses = asm.get("include_raw_poses", False)
            config.assembly.repo_id = asm.get("repo_id", "")
            config.assembly.fps = asm.get("fps", 60)

        return config

    def to_yaml(self, path: Path):
        """Save configuration to YAML file."""
        import yaml
        data = {
            "stage1": {
                "trajectory_fps": self.stage1.trajectory_fps,
                "video_fps": self.stage1.video_fps,
                "camera": {
                    "name": self.stage1.camera.name,
                    "index": self.stage1.camera.index,
                    "width": self.stage1.camera.width,
                    "height": self.stage1.camera.height,
                    "fps": self.stage1.camera.fps,
                },
                "controller_hand": self.stage1.controller_hand,
                "server": self.stage1.server_url,
            },
            "stage2": {
                "urdf": self.stage2.urdf_path,
                "solver_dt": self.stage2.solver_dt,
                "regularization": self.stage2.regularization,
                "frame_name": self.stage2.frame_name,
            },
            "stage3": {
                "model": self.stage3.model,
                "prompt_template": self.stage3.prompt_template,
            },
            "stage4": {
                "model": self.stage4.model,
                "batch_size": self.stage4.batch_size,
                "sample_rate": self.stage4.sample_rate,
            },
            "assembly": {
                "output_format": self.assembly.output_format,
                "include_raw_poses": self.assembly.include_raw_poses,
                "repo_id": self.assembly.repo_id,
                "fps": self.assembly.fps,
            },
        }
        with open(path, "w") as f:
            yaml.safe_dump(data, f, default_flow_style=False)


# =============================================================================
# Helper Functions
# =============================================================================

def trajectory_frames_to_table(frames: List[TrajectoryFrame]) -> pa.Table:
    """Convert list of TrajectoryFrame to Arrow table."""
    data = {
        "timestamp_ns": [f.timestamp_ns for f in frames],
        "frame_index": [f.frame_index for f in frames],
        "position_x": [f.position_x for f in frames],
        "position_y": [f.position_y for f in frames],
        "position_z": [f.position_z for f in frames],
        "orientation_x": [f.orientation_x for f in frames],
        "orientation_y": [f.orientation_y for f in frames],
        "orientation_z": [f.orientation_z for f in frames],
        "orientation_w": [f.orientation_w for f in frames],
        "button_trigger": [f.button_trigger for f in frames],
        "button_grip": [f.button_grip for f in frames],
        "controller_hand": [f.controller_hand for f in frames],
    }
    return pa.Table.from_pydict(data, schema=TRAJECTORY_SCHEMA)


def video_timestamps_to_table(timestamps: List[VideoTimestamp]) -> pa.Table:
    """Convert list of VideoTimestamp to Arrow table."""
    data = {
        "frame_index": [t.frame_index for t in timestamps],
        "timestamp_ns": [t.timestamp_ns for t in timestamps],
        "closest_trajectory_idx": [t.closest_trajectory_idx for t in timestamps],
    }
    return pa.Table.from_pydict(data, schema=VIDEO_TIMESTAMPS_SCHEMA)


def joint_states_to_table(states: List[JointState]) -> pa.Table:
    """Convert list of JointState to Arrow table."""
    data = {
        "timestamp_ns": [s.timestamp_ns for s in states],
        "frame_index": [s.frame_index for s in states],
        "shoulder_pan": [s.shoulder_pan for s in states],
        "shoulder_lift": [s.shoulder_lift for s in states],
        "elbow_flex": [s.elbow_flex for s in states],
        "elbow_roll": [s.elbow_roll for s in states],
        "wrist_flex": [s.wrist_flex for s in states],
        "wrist_roll": [s.wrist_roll for s in states],
        "ik_success": [s.ik_success for s in states],
        "ik_error": [s.ik_error for s in states],
    }
    return pa.Table.from_pydict(data, schema=JOINTS_SCHEMA)


def gripper_states_to_table(states: List[GripperState]) -> pa.Table:
    """Convert list of GripperState to Arrow table."""
    data = {
        "frame_index": [s.frame_index for s in states],
        "timestamp_ns": [s.timestamp_ns for s in states],
        "measured_state": [s.measured_state for s in states],
        "commanded_state": [s.commanded_state for s in states],
        "confidence": [s.confidence for s in states],
        "annotation": [s.annotation for s in states],
    }
    return pa.Table.from_pydict(data, schema=GRIPPER_STATES_SCHEMA)
