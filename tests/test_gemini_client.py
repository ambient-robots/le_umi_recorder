"""
Tests for umi.pipeline.gemini_client module.

Uses mocking to test API client behavior without actual API calls.
"""

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
import tempfile

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock genai before importing our module
with patch.dict('sys.modules', {'google.generativeai': MagicMock(), 'google.generativeai.types': MagicMock()}):
    from umi.pipeline.gemini_client import (
        GeminiAPIError,
        GeminiRateLimitError,
        get_task_prompt,
        get_gripper_prompt,
    )


class TestGeminiClientParseJson:
    """Tests for JSON response parsing (no API calls needed)."""

    def test_parse_json_response_plain(self):
        """Test parsing plain JSON response."""
        from umi.pipeline.gemini_client import GeminiClient

        # Create client with mocked genai
        with patch('umi.pipeline.gemini_client.genai') as mock_genai:
            mock_genai.configure = MagicMock()
            mock_genai.GenerativeModel = MagicMock()

            client = GeminiClient.__new__(GeminiClient)
            client.model_name = "test"
            client.max_retries = 3
            client.base_delay = 1.0
            client.max_delay = 60.0
            client._uploaded_files = []

            response = '{"key": "value", "number": 42}'
            result = client.parse_json_response(response)

            assert result == {"key": "value", "number": 42}

    def test_parse_json_response_with_markdown_json_block(self):
        """Test parsing JSON wrapped in ```json block."""
        from umi.pipeline.gemini_client import GeminiClient

        with patch('umi.pipeline.gemini_client.genai'):
            client = GeminiClient.__new__(GeminiClient)
            client._uploaded_files = []

            response = '''```json
{"task": "pick up block"}
```'''
            result = client.parse_json_response(response)

            assert result == {"task": "pick up block"}

    def test_parse_json_response_with_plain_markdown_block(self):
        """Test parsing JSON wrapped in plain ``` block."""
        from umi.pipeline.gemini_client import GeminiClient

        with patch('umi.pipeline.gemini_client.genai'):
            client = GeminiClient.__new__(GeminiClient)
            client._uploaded_files = []

            response = '''```
{"data": [1, 2, 3]}
```'''
            result = client.parse_json_response(response)

            assert result == {"data": [1, 2, 3]}

    def test_parse_json_response_with_whitespace(self):
        """Test parsing JSON with leading/trailing whitespace."""
        from umi.pipeline.gemini_client import GeminiClient

        with patch('umi.pipeline.gemini_client.genai'):
            client = GeminiClient.__new__(GeminiClient)
            client._uploaded_files = []

            response = '''

            {"test": true}

            '''
            result = client.parse_json_response(response)

            assert result == {"test": True}

    def test_parse_json_response_invalid_json(self):
        """Test that invalid JSON raises GeminiAPIError."""
        from umi.pipeline.gemini_client import GeminiClient

        with patch('umi.pipeline.gemini_client.genai'):
            client = GeminiClient.__new__(GeminiClient)
            client._uploaded_files = []

            response = '{"invalid: json'

            with pytest.raises(GeminiAPIError) as exc_info:
                client.parse_json_response(response)

            assert "Failed to parse JSON" in str(exc_info.value)

    def test_parse_json_array_response(self):
        """Test parsing JSON array response."""
        from umi.pipeline.gemini_client import GeminiClient

        with patch('umi.pipeline.gemini_client.genai'):
            client = GeminiClient.__new__(GeminiClient)
            client._uploaded_files = []

            response = '[{"frame": 1}, {"frame": 2}]'
            result = client.parse_json_array_response(response)

            assert result == [{"frame": 1}, {"frame": 2}]

    def test_parse_json_array_response_not_array(self):
        """Test that non-array JSON raises error."""
        from umi.pipeline.gemini_client import GeminiClient

        with patch('umi.pipeline.gemini_client.genai'):
            client = GeminiClient.__new__(GeminiClient)
            client._uploaded_files = []

            response = '{"not": "array"}'

            with pytest.raises(GeminiAPIError) as exc_info:
                client.parse_json_array_response(response)

            assert "Expected JSON array" in str(exc_info.value)


class TestPromptTemplates:
    """Tests for prompt template functions."""

    def test_get_task_prompt_v1(self):
        """Test getting v1 task prompt."""
        prompt = get_task_prompt("v1")

        assert "task" in prompt.lower()
        assert "robot" in prompt.lower()
        assert "JSON" in prompt

    def test_get_task_prompt_task_v1(self):
        """Test getting task_v1 task prompt (alias)."""
        prompt = get_task_prompt("task_v1")

        assert prompt == get_task_prompt("v1")

    def test_get_task_prompt_unknown(self):
        """Test that unknown version raises error."""
        with pytest.raises(ValueError) as exc_info:
            get_task_prompt("v99")

        assert "Unknown task prompt version" in str(exc_info.value)

    def test_get_gripper_prompt_v1(self):
        """Test getting v1 gripper prompt."""
        prompt = get_gripper_prompt("v1")

        assert "gripper" in prompt.lower()
        assert "measured" in prompt.lower()
        assert "commanded" in prompt.lower()
        assert "0-100" in prompt

    def test_get_gripper_prompt_gripper_v1(self):
        """Test getting gripper_v1 prompt (alias)."""
        prompt = get_gripper_prompt("gripper_v1")

        assert prompt == get_gripper_prompt("v1")

    def test_get_gripper_prompt_unknown(self):
        """Test that unknown version raises error."""
        with pytest.raises(ValueError) as exc_info:
            get_gripper_prompt("v99")

        assert "Unknown gripper prompt version" in str(exc_info.value)


class TestGeminiClientInit:
    """Tests for GeminiClient initialization."""

    def test_init_with_api_key(self):
        """Test initialization with explicit API key."""
        from umi.pipeline.gemini_client import GeminiClient

        with patch('umi.pipeline.gemini_client.genai') as mock_genai:
            with patch('umi.pipeline.gemini_client.GENAI_AVAILABLE', True):
                mock_genai.configure = MagicMock()
                mock_genai.GenerativeModel = MagicMock()

                client = GeminiClient(api_key="test-api-key")

                mock_genai.configure.assert_called_once_with(api_key="test-api-key")

    def test_init_with_env_var(self):
        """Test initialization with API key from environment."""
        import os
        from umi.pipeline.gemini_client import GeminiClient

        with patch('umi.pipeline.gemini_client.genai') as mock_genai:
            with patch('umi.pipeline.gemini_client.GENAI_AVAILABLE', True):
                with patch.dict(os.environ, {"GEMINI_API_KEY": "env-api-key"}):
                    mock_genai.configure = MagicMock()
                    mock_genai.GenerativeModel = MagicMock()

                    client = GeminiClient()

                    mock_genai.configure.assert_called_once_with(api_key="env-api-key")

    def test_init_without_api_key_raises(self):
        """Test that missing API key raises ValueError."""
        import os
        from umi.pipeline.gemini_client import GeminiClient

        with patch('umi.pipeline.gemini_client.genai'):
            with patch('umi.pipeline.gemini_client.GENAI_AVAILABLE', True):
                with patch.dict(os.environ, {}, clear=True):
                    # Remove GEMINI_API_KEY if present
                    os.environ.pop("GEMINI_API_KEY", None)

                    with pytest.raises(ValueError) as exc_info:
                        GeminiClient()

                    assert "API key not found" in str(exc_info.value)

    def test_init_custom_model(self):
        """Test initialization with custom model name."""
        from umi.pipeline.gemini_client import GeminiClient

        with patch('umi.pipeline.gemini_client.genai') as mock_genai:
            with patch('umi.pipeline.gemini_client.GENAI_AVAILABLE', True):
                mock_genai.configure = MagicMock()
                mock_genai.GenerativeModel = MagicMock()

                client = GeminiClient(model="gemini-pro", api_key="test")

                assert client.model_name == "gemini-pro"
                mock_genai.GenerativeModel.assert_called_once()


@pytest.mark.asyncio
class TestGeminiClientAsync:
    """Async tests for GeminiClient methods."""

    async def test_upload_video_file_not_found(self):
        """Test that missing video file raises FileNotFoundError."""
        from umi.pipeline.gemini_client import GeminiClient

        with patch('umi.pipeline.gemini_client.genai') as mock_genai:
            with patch('umi.pipeline.gemini_client.GENAI_AVAILABLE', True):
                mock_genai.configure = MagicMock()
                mock_genai.GenerativeModel = MagicMock()

                client = GeminiClient(api_key="test")

                with pytest.raises(FileNotFoundError):
                    await client.upload_video(Path("/nonexistent/video.mp4"))

    async def test_upload_video_success(self):
        """Test successful video upload."""
        from umi.pipeline.gemini_client import GeminiClient

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake video content")
            video_path = Path(f.name)

        try:
            with patch('umi.pipeline.gemini_client.genai') as mock_genai:
                with patch('umi.pipeline.gemini_client.GENAI_AVAILABLE', True):
                    mock_genai.configure = MagicMock()
                    mock_genai.GenerativeModel = MagicMock()

                    # Mock file upload
                    mock_file = MagicMock()
                    mock_file.state.name = "ACTIVE"
                    mock_file.uri = "gs://test/video.mp4"
                    mock_genai.upload_file = MagicMock(return_value=mock_file)

                    client = GeminiClient(api_key="test")
                    result = await client.upload_video(video_path)

                    assert result == mock_file
                    assert mock_file in client._uploaded_files
        finally:
            video_path.unlink()

    async def test_generate_with_retry_success(self):
        """Test successful generation without retry."""
        from umi.pipeline.gemini_client import GeminiClient

        with patch('umi.pipeline.gemini_client.genai') as mock_genai:
            with patch('umi.pipeline.gemini_client.GENAI_AVAILABLE', True):
                mock_genai.configure = MagicMock()

                # Mock model response
                mock_response = MagicMock()
                mock_response.text = "Generated response"
                mock_model = MagicMock()
                mock_model.generate_content = MagicMock(return_value=mock_response)
                mock_genai.GenerativeModel = MagicMock(return_value=mock_model)

                client = GeminiClient(api_key="test")
                mock_file = MagicMock()

                result = await client._generate_with_retry(mock_file, "test prompt")

                assert result == "Generated response"

    async def test_generate_with_retry_rate_limit(self):
        """Test retry on rate limit error."""
        from umi.pipeline.gemini_client import GeminiClient

        with patch('umi.pipeline.gemini_client.genai') as mock_genai:
            with patch('umi.pipeline.gemini_client.GENAI_AVAILABLE', True):
                mock_genai.configure = MagicMock()

                # First call fails with rate limit, second succeeds
                mock_response = MagicMock()
                mock_response.text = "Success after retry"
                mock_model = MagicMock()
                mock_model.generate_content = MagicMock(
                    side_effect=[
                        Exception("Rate limit exceeded (429)"),
                        mock_response,
                    ]
                )
                mock_genai.GenerativeModel = MagicMock(return_value=mock_model)

                client = GeminiClient(api_key="test", base_delay=0.01)  # Fast retry for test
                mock_file = MagicMock()

                result = await client._generate_with_retry(mock_file, "test prompt")

                assert result == "Success after retry"
                assert mock_model.generate_content.call_count == 2

    async def test_generate_with_retry_max_retries_exceeded(self):
        """Test that max retries raises error."""
        from umi.pipeline.gemini_client import GeminiClient

        with patch('umi.pipeline.gemini_client.genai') as mock_genai:
            with patch('umi.pipeline.gemini_client.GENAI_AVAILABLE', True):
                mock_genai.configure = MagicMock()

                # All calls fail with rate limit
                mock_model = MagicMock()
                mock_model.generate_content = MagicMock(
                    side_effect=Exception("Rate limit (429)")
                )
                mock_genai.GenerativeModel = MagicMock(return_value=mock_model)

                client = GeminiClient(
                    api_key="test",
                    max_retries=2,
                    base_delay=0.01,
                )
                mock_file = MagicMock()

                with pytest.raises(GeminiAPIError) as exc_info:
                    await client._generate_with_retry(mock_file, "test prompt")

                assert "Max retries exceeded" in str(exc_info.value)
                assert mock_model.generate_content.call_count == 2

    async def test_generate_with_retry_non_retryable_error(self):
        """Test that non-retryable errors are raised immediately."""
        from umi.pipeline.gemini_client import GeminiClient

        with patch('umi.pipeline.gemini_client.genai') as mock_genai:
            with patch('umi.pipeline.gemini_client.GENAI_AVAILABLE', True):
                mock_genai.configure = MagicMock()

                # Fail with non-retryable error
                mock_model = MagicMock()
                mock_model.generate_content = MagicMock(
                    side_effect=ValueError("Invalid input")
                )
                mock_genai.GenerativeModel = MagicMock(return_value=mock_model)

                client = GeminiClient(api_key="test", max_retries=5)
                mock_file = MagicMock()

                with pytest.raises(ValueError) as exc_info:
                    await client._generate_with_retry(mock_file, "test prompt")

                assert "Invalid input" in str(exc_info.value)
                # Should not retry for non-retryable errors
                assert mock_model.generate_content.call_count == 1

    async def test_cleanup_deletes_files(self):
        """Test that cleanup deletes uploaded files."""
        from umi.pipeline.gemini_client import GeminiClient

        with patch('umi.pipeline.gemini_client.genai') as mock_genai:
            with patch('umi.pipeline.gemini_client.GENAI_AVAILABLE', True):
                mock_genai.configure = MagicMock()
                mock_genai.GenerativeModel = MagicMock()
                mock_genai.delete_file = MagicMock()

                client = GeminiClient(api_key="test")

                # Add mock uploaded files
                file1 = MagicMock()
                file1.name = "file1"
                file1.display_name = "video1.mp4"
                file2 = MagicMock()
                file2.name = "file2"
                file2.display_name = "video2.mp4"
                client._uploaded_files = [file1, file2]

                await client.cleanup()

                assert mock_genai.delete_file.call_count == 2
                assert client._uploaded_files == []

    async def test_context_manager(self):
        """Test async context manager behavior."""
        from umi.pipeline.gemini_client import GeminiClient

        with patch('umi.pipeline.gemini_client.genai') as mock_genai:
            with patch('umi.pipeline.gemini_client.GENAI_AVAILABLE', True):
                mock_genai.configure = MagicMock()
                mock_genai.GenerativeModel = MagicMock()
                mock_genai.delete_file = MagicMock()

                async with GeminiClient(api_key="test") as client:
                    file = MagicMock()
                    file.name = "test"
                    file.display_name = "test.mp4"
                    client._uploaded_files = [file]

                # Cleanup should have been called on exit
                mock_genai.delete_file.assert_called_once()


class TestGeminiResponseParsing:
    """Tests for parsing Gemini responses in real-world formats."""

    def test_parse_task_response(self, mock_gemini_task_response):
        """Test parsing a task prompt response."""
        from umi.pipeline.gemini_client import GeminiClient

        with patch('umi.pipeline.gemini_client.genai'):
            client = GeminiClient.__new__(GeminiClient)
            client._uploaded_files = []

            response = json.dumps(mock_gemini_task_response)
            result = client.parse_json_response(response)

            assert result["task_description"] == mock_gemini_task_response["task_description"]
            assert result["confidence"] == mock_gemini_task_response["confidence"]
            assert result["objects"] == mock_gemini_task_response["objects"]

    def test_parse_gripper_response(self, mock_gemini_gripper_response):
        """Test parsing a gripper detection response."""
        from umi.pipeline.gemini_client import GeminiClient

        with patch('umi.pipeline.gemini_client.genai'):
            client = GeminiClient.__new__(GeminiClient)
            client._uploaded_files = []

            response = json.dumps(mock_gemini_gripper_response)
            result = client.parse_json_response(response)

            assert "frames" in result
            assert len(result["frames"]) == len(mock_gemini_gripper_response["frames"])
            assert result["total_duration_seconds"] == 2.0

    def test_parse_gripper_response_with_markdown(self, mock_gemini_gripper_response):
        """Test parsing gripper response wrapped in markdown."""
        from umi.pipeline.gemini_client import GeminiClient

        with patch('umi.pipeline.gemini_client.genai'):
            client = GeminiClient.__new__(GeminiClient)
            client._uploaded_files = []

            response = f"```json\n{json.dumps(mock_gemini_gripper_response)}\n```"
            result = client.parse_json_response(response)

            assert "frames" in result
            assert result["summary"] == mock_gemini_gripper_response["summary"]
