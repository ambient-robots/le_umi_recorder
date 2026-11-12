"""
Shared Gemini API client for the UMI recording pipeline.

Handles video upload and analysis with Gemini for:
- Stage 3: Task prompt generation
- Stage 4: Gripper state detection
"""

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    genai = None


T = TypeVar("T")


class GeminiAPIError(Exception):
    """Exception raised for Gemini API errors."""
    pass


class GeminiRateLimitError(GeminiAPIError):
    """Exception raised when rate limited."""
    pass


class GeminiClient:
    """
    Client for Gemini API video analysis.

    Features:
    - Video file upload via Files API
    - Automatic retry with exponential backoff
    - Rate limiting compliance
    - Response parsing and validation
    """

    # Default safety settings (permissive for robot video analysis)
    SAFETY_SETTINGS = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    } if GENAI_AVAILABLE else {}

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
        api_key_env: str = "GEMINI_API_KEY",
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
    ):
        """
        Initialize Gemini client.

        Args:
            model: Model name to use
            api_key: API key (if not provided, reads from environment)
            api_key_env: Environment variable name for API key
            max_retries: Maximum number of retries on failure
            base_delay: Base delay for exponential backoff
            max_delay: Maximum delay between retries
        """
        if not GENAI_AVAILABLE:
            raise ImportError(
                "google-generativeai package not installed. "
                "Install with: pip install google-generativeai"
            )

        self.model_name = model
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

        # Configure API key
        api_key = api_key or os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(
                f"Gemini API key not found. Set {api_key_env} environment variable "
                "or pass api_key parameter."
            )
        genai.configure(api_key=api_key)

        # Initialize model
        self.model = genai.GenerativeModel(
            model_name=model,
            safety_settings=self.SAFETY_SETTINGS,
        )

        # Track uploaded files for cleanup
        self._uploaded_files: List[Any] = []

    async def upload_video(
        self,
        video_path: Path,
        display_name: Optional[str] = None,
    ) -> Any:
        """
        Upload video to Gemini Files API.

        Args:
            video_path: Path to video file
            display_name: Optional display name for the file

        Returns:
            Uploaded file object
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        display_name = display_name or video_path.name

        # Upload file
        print(f"Uploading video: {video_path.name}...")
        file = genai.upload_file(
            path=str(video_path),
            display_name=display_name,
            mime_type="video/mp4",
        )

        # Wait for processing
        print(f"Processing video (this may take a moment)...")
        while file.state.name == "PROCESSING":
            await asyncio.sleep(2)
            file = genai.get_file(file.name)

        if file.state.name == "FAILED":
            raise GeminiAPIError(f"Video processing failed: {file.state.name}")

        print(f"Video ready: {file.uri}")
        self._uploaded_files.append(file)
        return file

    async def analyze_video(
        self,
        video_path: Path,
        prompt: str,
        json_response: bool = False,
    ) -> str:
        """
        Upload video and get analysis response.

        Args:
            video_path: Path to video file
            prompt: Analysis prompt
            json_response: If True, request JSON formatted response

        Returns:
            Model response text
        """
        file = await self.upload_video(video_path)

        # Build prompt
        if json_response:
            prompt = prompt + "\n\nRespond with valid JSON only, no markdown formatting."

        return await self._generate_with_retry(file, prompt)

    async def analyze_video_file(
        self,
        file: Any,
        prompt: str,
        json_response: bool = False,
    ) -> str:
        """
        Analyze an already-uploaded video file.

        Args:
            file: Uploaded file object from upload_video()
            prompt: Analysis prompt
            json_response: If True, request JSON formatted response

        Returns:
            Model response text
        """
        if json_response:
            prompt = prompt + "\n\nRespond with valid JSON only, no markdown formatting."

        return await self._generate_with_retry(file, prompt)

    async def _generate_with_retry(self, file: Any, prompt: str) -> str:
        """Generate content with retry logic."""
        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = await asyncio.to_thread(
                    self.model.generate_content,
                    [file, prompt],
                )

                if response.text:
                    return response.text

                # Check for blocked response
                if response.prompt_feedback:
                    raise GeminiAPIError(
                        f"Response blocked: {response.prompt_feedback}"
                    )

                raise GeminiAPIError("Empty response from model")

            except Exception as e:
                last_error = e
                error_str = str(e).lower()

                # Check for rate limiting
                if "rate" in error_str or "quota" in error_str or "429" in error_str:
                    delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                    print(f"Rate limited, waiting {delay:.1f}s before retry...")
                    await asyncio.sleep(delay)
                    continue

                # Check for transient errors
                if "500" in error_str or "503" in error_str or "timeout" in error_str:
                    delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                    print(f"Transient error, retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
                    continue

                # Non-retryable error
                raise

        raise GeminiAPIError(f"Max retries exceeded. Last error: {last_error}")

    def parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Parse JSON from model response.

        Handles common issues like markdown code blocks.
        """
        # Strip markdown code blocks if present
        text = response.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise GeminiAPIError(f"Failed to parse JSON response: {e}\nResponse: {text[:500]}")

    def parse_json_array_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse JSON array from model response."""
        parsed = self.parse_json_response(response)
        if isinstance(parsed, list):
            return parsed
        raise GeminiAPIError(f"Expected JSON array, got: {type(parsed)}")

    async def cleanup(self):
        """Delete uploaded files to free storage."""
        for file in self._uploaded_files:
            try:
                genai.delete_file(file.name)
                print(f"Deleted uploaded file: {file.display_name}")
            except Exception as e:
                print(f"Warning: Failed to delete file {file.display_name}: {e}")
        self._uploaded_files.clear()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup uploaded files."""
        await self.cleanup()


# =============================================================================
# Prompt Templates
# =============================================================================

TASK_PROMPT_V1 = """
You are analyzing a robot manipulation video. The video shows a robot arm
performing a task. Describe the task in a single concise sentence that
could be used as instruction for training a robot policy.

Format your response as: "Pick up [object] and [action] [location]"
Or similar action-oriented instruction format.

Be specific about:
- Objects being manipulated (color, type, size)
- Actions performed (pick, place, push, pour, etc.)
- Start and end positions/locations

Respond with JSON:
{
  "task_description": "Your task description here",
  "confidence": 0.0-1.0,
  "objects": ["object1", "object2"],
  "actions": ["grasp", "lift", "place"],
  "video_summary": "Brief summary of what happens in the video"
}
"""

GRIPPER_DETECTION_PROMPT_V1 = """
You are analyzing a robot manipulation video to detect gripper states.

For this video, analyze the gripper (the end effector that opens and closes
to grasp objects) throughout the video and provide frame-by-frame estimates.

For each frame (or time segment if there are many frames), estimate:

1. measured_state (0-100): How open is the gripper physically?
   - 0 = fully closed
   - 100 = fully open
   - This is what you OBSERVE in the video

2. commanded_state (0-100): What does the robot intend/want?
   - During grasping attempt: 0 (wants to close)
   - During releasing: 100 (wants to open)
   - During transit: maintain current
   - This is what the robot is TRYING to do

Key insight: When gripping an object, commanded might be 0 (close fully)
but measured might be 30 (object prevents full closure).

Sample the video at approximately 1 second intervals and provide:

{
  "frames": [
    {
      "time_seconds": 0.0,
      "measured": 100,
      "commanded": 100,
      "annotation": "gripper_open"
    },
    {
      "time_seconds": 1.0,
      "measured": 50,
      "commanded": 0,
      "annotation": "closing_on_object"
    },
    ...
  ],
  "total_duration_seconds": 10.0,
  "summary": "Brief description of gripper activity"
}

Annotations can be: "gripper_open", "gripper_closed", "closing_on_object",
"grasping_object", "releasing_object", "holding_object", "transit"
"""


def get_task_prompt(version: str = "v1") -> str:
    """Get task prompt template by version."""
    if version == "v1" or version == "task_v1":
        return TASK_PROMPT_V1
    raise ValueError(f"Unknown task prompt version: {version}")


def get_gripper_prompt(version: str = "v1") -> str:
    """Get gripper detection prompt template by version."""
    if version == "v1" or version == "gripper_v1":
        return GRIPPER_DETECTION_PROMPT_V1
    raise ValueError(f"Unknown gripper prompt version: {version}")
