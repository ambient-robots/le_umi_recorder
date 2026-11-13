"""Tracking-server regressions, independent of the UMI recording dependencies.

Run with: python -m unittest backend.test_server -v
"""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import aiohttp

from backend.server import ControllerTrackingServer, _Subscriber, load_server_config


def pose_message(timestamp=123.0):
    # The existing Quest frontend sends no tracking flags or per-hand timestamp.
    return {
        "type": "pose",
        "timestamp": timestamp,
        "coordinate_system": "local",
        "controllers": {
            "left": {
                "position": [1.0, 2.0, 3.0],
                "orientation": [0.0, 0.0, 0.0, 1.0],
                "buttons": {"0": {"pressed": True, "value": 0.75}},
            }
        },
    }


class TrackingServerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.server = ControllerTrackingServer(
            host="127.0.0.1", http_port=0, heartbeat=None, send_timeout=0.1
        )
        await self.server.start_http_server(use_ssl=False)
        self.addAsyncCleanup(self.server.stop)
        self.session = aiohttp.ClientSession()
        self.addAsyncCleanup(self.session.close)
        self.url = f"http://127.0.0.1:{self.server.bound_port}"
        self.quest = await self.connect("quest3")

    async def connect(self, kind):
        ws = await self.session.ws_connect(f"{self.url}/ws")
        self.addAsyncCleanup(ws.close)
        await ws.send_json({"type": "handshake", "client": kind})
        self.assertEqual((await ws.receive_json(timeout=1))["type"], "ack")
        return ws

    async def get(self, path):
        async with self.session.get(self.url + path) as response:
            return response.status, await response.json()

    async def until(self, predicate):
        async def wait():
            while not predicate():
                await asyncio.sleep(0.001)
        await asyncio.wait_for(wait(), timeout=1)

    async def publish(self, timestamp=123.0):
        count = self.server.total_frames_received
        await self.quest.send_json(pose_message(timestamp))
        await self.until(lambda: self.server.total_frames_received > count)

    async def test_rest_contract_for_recording_clients(self):
        self.assertEqual((await self.get("/api/pose/latest"))[0], 404)
        await self.publish()
        status, pose = await self.get("/api/pose/latest")
        self.assertEqual(status, 200)
        self.assertEqual(pose["timestamp"], 123.0)
        self.assertEqual(pose["coordinate_system"], "local")
        self.assertIsInstance(pose["received_at"], float)
        left = pose["controllers"]["left"]
        self.assertEqual(left["position"], [1.0, 2.0, 3.0])
        self.assertEqual(left["orientation"], [0.0, 0.0, 0.0, 1.0])
        self.assertEqual(left["timestamp"], pose["timestamp"])
        self.assertEqual(left["buttons"]["0"]["value"], 0.75)
        self.assertEqual(await self.get("/api/pose/left"), (200, left))
        self.assertEqual((await self.get("/api/pose/right"))[0], 404)
        _, status = await self.get("/api/status")
        self.assertTrue(status["has_pose_data"])
        self.assertEqual(status["quest_clients"], 1)
        self.assertGreater(status["current_frame_rate"], 0)

    async def test_expired_pose_is_neither_served_nor_replayed(self):
        await self.publish()
        self.server.latest_pose.received_monotonic -= self.server.pose_stale_after + 1
        for path in ("/api/pose/latest", "/api/pose/left", "/api/pose/right"):
            status, body = await self.get(path)
            self.assertEqual(status, 404)
            self.assertIn("stale", body["error"])
        self.assertFalse((await self.get("/api/status"))[1]["has_pose_data"])
        viewer = await self.connect("visualizer")
        await self.publish(124.0)
        # A stale replay would be queued before this fresh frame.
        self.assertEqual((await viewer.receive_json(timeout=1))["timestamp"], 124.0)

    async def test_disconnect_clears_pose_and_reconnection_recovers(self):
        await self.publish()
        await self.quest.close()
        await self.until(lambda: not self.server.quest_clients)
        self.assertEqual((await self.get("/api/pose/latest"))[0], 404)
        self.quest = await self.connect("quest3")
        await self.publish(125.0)
        self.assertEqual((await self.get("/api/pose/latest"))[1]["timestamp"], 125.0)

    async def test_xr_visibility_and_session_end_clear_pose(self):
        for state in ({"visibility": "hidden"}, {"xr_session": "ended"}):
            with self.subTest(state=state):
                await self.publish()
                await self.quest.send_json({"type": "status", **state})
                await self.until(lambda: self.server.latest_pose is None)
                self.assertEqual((await self.get("/api/pose/latest"))[0], 404)

    async def test_invalid_frame_cannot_replace_last_valid_pose(self):
        await self.publish()
        malformed = [
            {"type": "pose", "controllers": {}},
            {**pose_message(), "timestamp": float("nan")},
        ]
        bad_quaternion = pose_message()
        bad_quaternion["controllers"]["left"]["orientation"] = [0, 0, 0, 0]
        malformed.append(bad_quaternion)
        for count, message in enumerate(malformed, start=1):
            await self.quest.send_json(message)
            await self.until(lambda: self.server.rejected_frames == count)
        self.assertEqual(self.server.total_frames_received, 1)
        self.assertEqual((await self.get("/api/pose/latest"))[1]["timestamp"], 123.0)

    async def test_slow_subscriber_does_not_block_recording_or_other_viewers(self):
        viewer = await self.connect("visualizer")

        class BlockedSocket:
            async def send_str(self, payload):
                await asyncio.Event().wait()

        socket = BlockedSocket()
        transport = Mock()
        slow = _Subscriber(socket, SimpleNamespace(transport=transport, remote="slow"), 2)
        slow.task = asyncio.create_task(self.server._run_subscriber(slow))
        self.server.visualizer_clients[socket] = slow
        try:
            for timestamp in range(10):
                self.assertTrue(self.server.handle_pose_data(pose_message(timestamp)))
            self.assertEqual(slow.queue.qsize(), 2)
            self.assertEqual(slow.dropped, 8)
            self.assertEqual((await self.get("/api/pose/latest"))[1]["timestamp"], 9)
            frames = [await viewer.receive_json(timeout=1) for _ in range(10)]
            self.assertEqual(frames[-1]["timestamp"], 9)
            await asyncio.wait_for(slow.task, timeout=1)
            transport.abort.assert_called_once()
            self.assertNotIn(socket, self.server.visualizer_clients)
        finally:
            self.server._remove_subscriber(slow)
            await asyncio.gather(slow.task, return_exceptions=True)

    async def test_frontend_assets_revalidate_and_shutdown_closes_clients(self):
        async with self.session.get(self.url + "/app.js") as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(response.headers["Cache-Control"], "no-cache")
        viewer = await self.connect("visualizer")
        await self.server.stop()
        for ws in (self.quest, viewer):
            self.assertEqual((await ws.receive(timeout=1)).type, aiohttp.WSMsgType.CLOSE)
        self.assertIsNone(self.server.bound_port)
        await self.server.stop()


class ServerConfigTests(unittest.TestCase):
    def test_config_load_and_invalid_port(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({
                "server": {"host": "127.0.0.1", "port": 8123, "log_level": "debug"},
                "tracking": {},
            }))
            settings, ignored = load_server_config(path)
            self.assertEqual(settings, {"host": "127.0.0.1", "port": 8123, "log_level": "DEBUG"})
            self.assertEqual(ignored, ["tracking"])
            path.write_text('{"server": {"port": -1}}')
            with self.assertRaises(ValueError):
                load_server_config(path)


if __name__ == "__main__":
    unittest.main()
