"""
WebXR Controller Tracking Backend Server
High-performance WebSocket server for receiving and distributing controller poses

Liveness rules (a pose that is not live must never look live):
  * a cached pose older than ``pose_stale_after`` seconds (default 0.5 s) is never replayed to a new
    subscriber and is not served by the REST endpoints;
  * the cached pose is dropped when the last Quest client disconnects, when the headset reports that the
    XR session is hidden or has ended, and (lazily) when no pose arrived for ``pose_stale_after`` seconds;
  * ``/api/status`` reports ``pose_age_seconds`` and ``has_pose_data`` means "a fresh pose is available".

Distribution rules:
  * every subscriber has its own bounded queue (drop-oldest) and sender task, so a subscriber that stops
    reading can never stall pose ingestion or the other subscribers;
  * a subscriber whose socket does not accept data for ``send_timeout`` seconds is disconnected (it has to
    reconnect), and a WebSocket heartbeat drops dead sockets on both the Quest and the subscriber side.
"""

import argparse
import asyncio
import json
import logging
import math
import ssl
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Set, Tuple
from pathlib import Path

from aiohttp import web

logger = logging.getLogger(__name__)

# A pose older than this is not "live": not replayed on handshake, not served over REST.
POSE_STALE_AFTER_S = 0.5
# aiohttp sends a ping every WS_HEARTBEAT_S and closes the socket when no pong arrives within half of it.
# Deliberately generous: a subscriber whose event loop is busy for a few seconds must not be dropped.
WS_HEARTBEAT_S = 30.0
# Frames queued per subscriber before the oldest one is dropped (about 0.5 s at 60 Hz).
SUBSCRIBER_QUEUE_SIZE = 32
# A subscriber whose socket accepts no data for this long is disconnected.
SUBSCRIBER_SEND_TIMEOUT_S = 5.0
# Window over which /api/status computes the frame rate (decays to 0 after this much silence).
FRAME_RATE_WINDOW_S = 1.0

DEFAULT_HOST = '0.0.0.0'
DEFAULT_PORT = 8000
DEFAULT_LOG_LEVEL = 'INFO'
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / 'config.json'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
VALID_LOG_LEVELS = ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')

# Per-hand tracking states the frontend may report in the additive ``tracking`` field of a pose message.
TRACKING_STATES = ('tracked', 'emulated', 'lost', 'absent', 'disabled')


@dataclass
class ControllerPose:
    """Represents a 6DOF controller pose"""
    position: List[float]  # [x, y, z]
    orientation: List[float]  # [x, y, z, w] quaternion
    buttons: dict
    timestamp: float
    # False when the headset reports an emulated (inferred, not optically tracked) position.
    # None when the sender did not report it (older frontend).
    tracked: Optional[bool] = None


@dataclass
class PoseFrame:
    """Complete pose data for one or more controllers"""
    timestamp: float
    coordinate_system: str
    controllers: Dict[str, ControllerPose]
    received_at: float  # wall clock (time.time()) when the server received the frame
    # Per-hand tracking state as reported by the frontend ({} when not reported), see TRACKING_STATES.
    tracking: Dict[str, str] = field(default_factory=dict)
    # Monotonic receive time, used for all age computations (not serialised).
    received_monotonic: float = field(default_factory=time.monotonic)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        controllers = {}
        for hand, pose in self.controllers.items():
            entry = {
                'position': pose.position,
                'orientation': pose.orientation,
                'buttons': pose.buttons,
                'timestamp': pose.timestamp
            }
            if pose.tracked is not None:
                entry['tracked'] = pose.tracked
            controllers[hand] = entry

        result = {
            'timestamp': self.timestamp,
            'coordinate_system': self.coordinate_system,
            'controllers': controllers,
            'received_at': self.received_at
        }
        if self.tracking:
            result['tracking'] = self.tracking
        return result


def _finite_number(value: Any, what: str) -> float:
    """Return ``value`` as float, raising ValueError unless it is a finite real number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{what} must be a number, got {value!r}')
    if not math.isfinite(value):
        raise ValueError(f'{what} must be finite, got {value!r}')
    return float(value)


def _finite_vector(value: Any, length: int, what: str) -> List[float]:
    """Return ``value`` as a list of ``length`` finite floats, raising ValueError otherwise."""
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f'{what} must be a list of {length} numbers, got {value!r}')
    return [_finite_number(v, f'{what}[{i}]') for i, v in enumerate(value)]


def parse_pose_message(data: dict) -> PoseFrame:
    """
    Validate a ``type: 'pose'`` message from the headset and convert it to a PoseFrame.

    Raises:
        ValueError: if a field is missing or malformed (the message is then rejected as a whole,
            a half-valid frame is never stored or forwarded).
    """
    if 'timestamp' not in data:
        raise ValueError("pose message has no 'timestamp'")
    timestamp = _finite_number(data['timestamp'], 'timestamp')

    raw_controllers = data.get('controllers')
    if not isinstance(raw_controllers, dict):
        raise ValueError(f"pose message 'controllers' must be an object, got {raw_controllers!r}")

    controllers: Dict[str, ControllerPose] = {}
    for hand, pose_data in raw_controllers.items():
        if not isinstance(pose_data, dict):
            raise ValueError(f"controllers[{hand!r}] must be an object, got {pose_data!r}")
        position = _finite_vector(pose_data.get('position'), 3, f'controllers[{hand!r}].position')
        orientation = _finite_vector(pose_data.get('orientation'), 4, f'controllers[{hand!r}].orientation')
        if math.sqrt(sum(q * q for q in orientation)) < 1e-6:
            raise ValueError(f'controllers[{hand!r}].orientation is a zero quaternion')
        buttons = pose_data.get('buttons', {})
        if not isinstance(buttons, dict):
            raise ValueError(f"controllers[{hand!r}].buttons must be an object, got {buttons!r}")
        tracked = pose_data.get('tracked')
        if tracked is not None and not isinstance(tracked, bool):
            raise ValueError(f"controllers[{hand!r}].tracked must be a boolean, got {tracked!r}")
        controllers[hand] = ControllerPose(
            position=position,
            orientation=orientation,
            buttons=buttons,
            timestamp=timestamp,
            tracked=tracked
        )

    tracking = data.get('tracking', {})
    if not isinstance(tracking, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in tracking.items()):
        raise ValueError(f"pose message 'tracking' must map hand names to state strings, got {tracking!r}")

    coordinate_system = data.get('coordinate_system', 'local')
    if not isinstance(coordinate_system, str):
        raise ValueError(f"'coordinate_system' must be a string, got {coordinate_system!r}")

    return PoseFrame(
        timestamp=timestamp,
        coordinate_system=coordinate_system,
        controllers=controllers,
        received_at=time.time(),
        tracking=dict(tracking),
        received_monotonic=time.monotonic()
    )


class _Subscriber:
    """One stream subscriber: a bounded drop-oldest queue drained by its own sender task."""

    def __init__(self, ws: web.WebSocketResponse, request: web.Request, maxsize: int):
        self.ws = ws
        self.transport = request.transport
        self.peer = request.remote
        self.queue: 'asyncio.Queue[str]' = asyncio.Queue(maxsize=maxsize)
        self.dropped = 0
        self.closed = False
        self.task: Optional['asyncio.Task[None]'] = None

    def offer(self, payload: str) -> None:
        """Queue a serialised frame without ever blocking; the oldest queued frame is dropped when full."""
        if self.queue.full():
            self.queue.get_nowait()
            self.dropped += 1
            # Log the first drop and then once per 100, so a stuck subscriber does not flood the log.
            if self.dropped == 1 or self.dropped % 100 == 0:
                logger.warning(
                    f'Subscriber {self.peer} is not keeping up: dropped {self.dropped} pose frame(s) so far')
        self.queue.put_nowait(payload)


class ControllerTrackingServer:
    """Main server handling WebSocket connections and data distribution"""

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        http_port: int = DEFAULT_PORT,
        pose_stale_after: float = POSE_STALE_AFTER_S,
        heartbeat: Optional[float] = WS_HEARTBEAT_S,
        subscriber_queue_size: int = SUBSCRIBER_QUEUE_SIZE,
        send_timeout: float = SUBSCRIBER_SEND_TIMEOUT_S,
    ):
        """
        Args:
            host: interface to bind
            http_port: port serving the frontend, the REST API and the WebSocket (0 = ephemeral, see bound_port)
            pose_stale_after: seconds after which the cached pose is no longer served or replayed
            heartbeat: WebSocket ping interval in seconds (None disables the heartbeat)
            subscriber_queue_size: frames queued per subscriber before the oldest is dropped
            send_timeout: seconds a subscriber socket may refuse data before it is disconnected
        """
        if pose_stale_after <= 0:
            raise ValueError(f'pose_stale_after must be > 0, got {pose_stale_after}')
        if heartbeat is not None and heartbeat <= 0:
            raise ValueError(f'heartbeat must be > 0 or None, got {heartbeat}')
        if subscriber_queue_size < 1:
            raise ValueError(f'subscriber_queue_size must be >= 1, got {subscriber_queue_size}')
        if send_timeout <= 0:
            raise ValueError(f'send_timeout must be > 0, got {send_timeout}')

        self.host = host
        self.http_port = http_port
        self.pose_stale_after = pose_stale_after
        self.heartbeat = heartbeat
        self.subscriber_queue_size = subscriber_queue_size
        self.send_timeout = send_timeout

        # Connected clients
        self.quest_clients: Set[web.WebSocketResponse] = set()
        self.visualizer_clients: Dict[web.WebSocketResponse, _Subscriber] = {}
        # TCP transport of every open WebSocket, to be able to drop a peer that does not answer
        self._transports: Dict[web.WebSocketResponse, Optional[asyncio.BaseTransport]] = {}

        # Latest pose data (None when there is none or it was invalidated). Always read it through
        # get_fresh_pose() so that a stale pose is never handed out.
        self.latest_pose: Optional[PoseFrame] = None

        # Last XR state reported by the headset ({} when unknown): keys 'visibility', 'xr_session'
        self.quest_xr_state: Dict[str, str] = {}

        # Statistics
        self.total_frames_received = 0
        self.rejected_frames = 0
        self.start_time = time.time()
        self.last_frame_time = 0.0  # wall clock of the last accepted frame (0 = never)
        self._frame_times: Deque[float] = deque()  # monotonic arrival times within FRAME_RATE_WINDOW_S

        # HTTP app
        self.app = web.Application()
        self.setup_routes()
        self.app.on_shutdown.append(self._on_shutdown)
        self.runner: Optional[web.AppRunner] = None
        self.bound_port: Optional[int] = None  # actual port once started (differs from http_port when 0)

    def setup_routes(self):
        """Setup HTTP REST API routes"""
        # API routes
        self.app.router.add_get('/api/status', self.handle_status)
        self.app.router.add_get('/api/pose/latest', self.handle_latest_pose)
        self.app.router.add_get('/api/pose/left', self.handle_left_pose)
        self.app.router.add_get('/api/pose/right', self.handle_right_pose)

        # WebSocket route
        self.app.router.add_get('/ws', self.handle_websocket)

        # Frontend routes - serve static files
        self.app.router.add_get('/', self.handle_index)
        self.app.router.add_get('/index.html', self.handle_index)
        self.app.router.add_get('/app.js', self.handle_app_js)
        self.app.router.add_get('/styles.css', self.handle_styles_css)

    # ------------------------------------------------------------------
    # Pose cache
    # ------------------------------------------------------------------

    def pose_age(self) -> Optional[float]:
        """Seconds since the cached pose was received, None when there is no cached pose."""
        if self.latest_pose is None:
            return None
        return time.monotonic() - self.latest_pose.received_monotonic

    def get_fresh_pose(self) -> Optional[PoseFrame]:
        """The cached pose if it is younger than ``pose_stale_after``, else None."""
        age = self.pose_age()
        if age is None or age > self.pose_stale_after:
            return None
        return self.latest_pose

    def clear_pose(self, reason: str) -> None:
        """Invalidate the cached pose (tracking is known to be lost)."""
        if self.latest_pose is not None:
            logger.info(f'Cached pose cleared: {reason}')
        self.latest_pose = None

    def current_frame_rate(self) -> float:
        """Frames accepted during the last FRAME_RATE_WINDOW_S seconds, in Hz (0 when the stream is silent)."""
        cutoff = time.monotonic() - FRAME_RATE_WINDOW_S
        while self._frame_times and self._frame_times[0] < cutoff:
            self._frame_times.popleft()
        return round(len(self._frame_times) / FRAME_RATE_WINDOW_S, 1)

    def _no_pose_response(self, what: str) -> web.Response:
        """404 response explaining whether there is no pose at all or only a stale one."""
        age = self.pose_age()
        if age is not None and age > self.pose_stale_after:
            message = (f'{what}: last pose is stale ({age:.2f} s old, '
                       f'limit {self.pose_stale_after} s)')
        else:
            message = what
        return web.json_response({'error': message, 'pose_age_seconds': age}, status=404)

    # ------------------------------------------------------------------
    # REST API
    # ------------------------------------------------------------------

    async def handle_status(self, request: web.Request) -> web.Response:
        """GET /api/status - Server status"""
        uptime = time.time() - self.start_time

        status = {
            'status': 'running',
            'uptime_seconds': uptime,
            'quest_clients': len(self.quest_clients),
            'visualizer_clients': len(self.visualizer_clients),
            'total_frames_received': self.total_frames_received,
            'rejected_frames': self.rejected_frames,
            'current_frame_rate': self.current_frame_rate(),
            # True only while a fresh pose (younger than pose_stale_after_seconds) is available
            'has_pose_data': self.get_fresh_pose() is not None,
            # Age of the cached pose, null when there is none
            'pose_age_seconds': self.pose_age(),
            'pose_stale_after_seconds': self.pose_stale_after,
            'last_update': self.last_frame_time if self.last_frame_time else None,
            # Last XR state reported by the headset page, null when unknown
            'quest_visibility': self.quest_xr_state.get('visibility'),
            'quest_xr_session': self.quest_xr_state.get('xr_session')
        }

        return web.json_response(status)

    async def handle_latest_pose(self, request: web.Request) -> web.Response:
        """GET /api/pose/latest - Latest pose data (404 when there is no fresh pose)"""
        pose = self.get_fresh_pose()
        if pose is None:
            return self._no_pose_response('No pose data available')

        return web.json_response(pose.to_dict())

    async def _handle_hand_pose(self, hand: str) -> web.Response:
        """Shared implementation of /api/pose/left and /api/pose/right"""
        pose = self.get_fresh_pose()
        if pose is None or hand not in pose.controllers:
            return self._no_pose_response(f'No {hand} controller data')

        controller = pose.controllers[hand]
        body = {
            'position': controller.position,
            'orientation': controller.orientation,
            'buttons': controller.buttons,
            'timestamp': controller.timestamp
        }
        if controller.tracked is not None:
            body['tracked'] = controller.tracked
        return web.json_response(body)

    async def handle_left_pose(self, request: web.Request) -> web.Response:
        """GET /api/pose/left - Left controller pose"""
        return await self._handle_hand_pose('left')

    async def handle_right_pose(self, request: web.Request) -> web.Response:
        """GET /api/pose/right - Right controller pose"""
        return await self._handle_hand_pose('right')

    # ------------------------------------------------------------------
    # Frontend
    # ------------------------------------------------------------------

    # 'no-cache' makes the headset browser revalidate on every load, so it never runs an outdated app.js
    # against a newer server.
    _STATIC_HEADERS = {'Cache-Control': 'no-cache'}

    async def handle_index(self, request: web.Request) -> web.Response:
        """Serve index.html"""
        frontend_dir = Path(__file__).parent.parent / 'frontend'
        index_file = frontend_dir / 'index.html'
        return web.FileResponse(index_file, headers=dict(self._STATIC_HEADERS))

    async def handle_app_js(self, request: web.Request) -> web.Response:
        """Serve app.js"""
        frontend_dir = Path(__file__).parent.parent / 'frontend'
        js_file = frontend_dir / 'app.js'
        return web.FileResponse(
            js_file, headers={'Content-Type': 'application/javascript', **self._STATIC_HEADERS})

    async def handle_styles_css(self, request: web.Request) -> web.Response:
        """Serve styles.css"""
        frontend_dir = Path(__file__).parent.parent / 'frontend'
        css_file = frontend_dir / 'styles.css'
        return web.FileResponse(css_file, headers={'Content-Type': 'text/css', **self._STATIC_HEADERS})

    # ------------------------------------------------------------------
    # WebSocket
    # ------------------------------------------------------------------

    async def handle_websocket(self, request: web.Request) -> web.WebSocketResponse:
        """WebSocket endpoint - serves both Quest and visualizer clients"""
        # The heartbeat pings the peer and closes the socket when no pong comes back, so a sleeping
        # headset or a vanished subscriber does not stay registered forever.
        ws = web.WebSocketResponse(heartbeat=self.heartbeat)
        await ws.prepare(request)
        self._transports[ws] = request.transport

        client_type = None
        subscriber: Optional[_Subscriber] = None

        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                    except json.JSONDecodeError as e:
                        logger.error(f'Invalid JSON received from {request.remote}: {e}')
                        continue
                    if not isinstance(data, dict):
                        logger.error(f'Ignoring non-object message from {request.remote}: {data!r}')
                        continue

                    # Determine client type from first message
                    if client_type is None:
                        client_type, subscriber = await self._handle_handshake(ws, request, data)

                    # Handle messages from Quest
                    elif client_type == 'quest':
                        self.handle_quest_message(data)

                elif msg.type == web.WSMsgType.ERROR:
                    logger.error(f'WebSocket error: {ws.exception()}')

        finally:
            # Cleanup on disconnect
            self._transports.pop(ws, None)
            if client_type == 'quest':
                self.quest_clients.discard(ws)
                logger.info('Quest client disconnected')
                if not self.quest_clients:
                    # Nothing can update the pose any more: never leave a frozen pose behind.
                    self.clear_pose('last Quest client disconnected')
                    self.quest_xr_state = {}
            elif subscriber is not None:
                self._remove_subscriber(subscriber)
                logger.info('Visualizer client disconnected')

        return ws

    async def _handle_handshake(
        self, ws: web.WebSocketResponse, request: web.Request, data: dict
    ) -> Tuple[Optional[str], Optional[_Subscriber]]:
        """Register the client announced by the first message. Returns (client_type, subscriber)."""
        client = data.get('client') if data.get('type') == 'handshake' else None

        if client == 'quest3':
            self.quest_clients.add(ws)
            logger.info('Quest client connected')
            await ws.send_json({'type': 'ack', 'message': 'Quest client registered'})
            return 'quest', None

        if client == 'visualizer':
            await ws.send_json({'type': 'ack', 'message': 'Visualizer client registered'})
            subscriber = _Subscriber(ws, request, self.subscriber_queue_size)
            subscriber.task = asyncio.ensure_future(self._run_subscriber(subscriber))
            subscriber.task.add_done_callback(
                lambda task, subscriber=subscriber: self._subscriber_task_done(subscriber, task))
            self.visualizer_clients[ws] = subscriber
            logger.info('Visualizer client connected')

            # Replay the latest pose only while it is still live. A pose of unknown age must never be
            # handed to a new subscriber as if it were current.
            pose = self.get_fresh_pose()
            if pose is not None:
                subscriber.offer(json.dumps(pose.to_dict()))
            return 'visualizer', subscriber

        logger.warning(f'Ignoring message from {request.remote} before a valid handshake: {data!r}')
        await ws.send_json({
            'type': 'error',
            'message': "Expected {'type': 'handshake', 'client': 'quest3' | 'visualizer'} as first message"
        })
        return None, None

    def handle_quest_message(self, data: dict) -> None:
        """Dispatch one message from the headset page. Never awaits: ingestion cannot be stalled."""
        message_type = data.get('type')
        if message_type == 'pose':
            self.handle_pose_data(data)
        elif message_type == 'status':
            self.handle_quest_status(data)
        else:
            logger.warning(f'Ignoring Quest message of unknown type {message_type!r}')

    def handle_quest_status(self, data: dict) -> None:
        """
        Process a ``type: 'status'`` message: the headset page reports XR session visibility changes and
        the end of the session, i.e. the moments from which no pose frames will arrive.
        """
        state: Dict[str, str] = {}
        for key in ('visibility', 'xr_session'):
            value = data.get(key)
            if value is None:
                continue
            if not isinstance(value, str):
                logger.error(f'Rejected Quest status message: {key!r} must be a string, got {value!r}')
                return
            state[key] = value
        self.quest_xr_state = state
        logger.info(f'Quest XR state: {state}')

        if state.get('xr_session') == 'ended':
            self.clear_pose('XR session ended')
        elif state.get('visibility') not in (None, 'visible'):
            self.clear_pose(f"XR session visibility is '{state['visibility']}'")

    def handle_pose_data(self, data: dict) -> bool:
        """
        Process incoming pose data from Quest. Returns True when the frame was accepted.

        Synchronous on purpose: the frame is stored and handed to the per-subscriber queues without
        awaiting any subscriber, so the Quest receive loop is never blocked by a slow consumer.
        """
        try:
            pose_frame = parse_pose_message(data)
        except ValueError as e:
            self.rejected_frames += 1
            # Log the first rejection and then once per 100, the headset sends up to 90 frames/s.
            if self.rejected_frames == 1 or self.rejected_frames % 100 == 0:
                logger.error(f'Rejected malformed pose message ({self.rejected_frames} so far): {e}')
            return False

        # Update latest pose
        self.latest_pose = pose_frame

        # Update statistics
        self.total_frames_received += 1
        self.last_frame_time = pose_frame.received_at
        self._frame_times.append(pose_frame.received_monotonic)

        # Broadcast to visualizer clients
        self.broadcast_to_visualizers(pose_frame.to_dict())

        # Log periodically
        if self.total_frames_received % 100 == 0:
            logger.info(
                f'Received {self.total_frames_received} frames, current rate: {self.current_frame_rate()} Hz')

        return True

    def broadcast_to_visualizers(self, data: dict) -> None:
        """Queue pose data for all connected visualizers (never blocks, see _Subscriber.offer)"""
        if not self.visualizer_clients:
            return

        payload = json.dumps(data)
        for subscriber in self.visualizer_clients.values():
            subscriber.offer(payload)

    async def _run_subscriber(self, subscriber: _Subscriber) -> None:
        """Sender task of one subscriber: drains its queue into its socket."""
        while not subscriber.closed:
            payload = await subscriber.queue.get()
            try:
                await asyncio.wait_for(subscriber.ws.send_str(payload), timeout=self.send_timeout)
            except asyncio.TimeoutError:
                logger.warning(
                    f'Subscriber {subscriber.peer} accepted no data for {self.send_timeout} s; '
                    f'closing its connection (it has to reconnect)')
                self._abort_subscriber(subscriber)
                return
            except ConnectionError as e:
                # The peer is gone; the receive loop of its handler notices the closed socket and cleans up.
                logger.info(f'Subscriber {subscriber.peer} went away: {e}')
                self._abort_subscriber(subscriber)
                return

    def _abort_subscriber(self, subscriber: _Subscriber) -> None:
        """Stop sending to a subscriber and drop its TCP connection without waiting for it."""
        subscriber.closed = True
        self.visualizer_clients.pop(subscriber.ws, None)
        if subscriber.transport is not None:
            # abort() rather than ws.close(): the close handshake would have to wait for the very peer
            # that is not reading.
            subscriber.transport.abort()

    def _subscriber_task_done(self, subscriber: _Subscriber, task: 'asyncio.Task[None]') -> None:
        """Done callback of a sender task: an unexpected failure must not leave a mute subscriber behind."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error(
                f'Sender task of subscriber {subscriber.peer} failed; closing its connection',
                exc_info=(type(exc), exc, exc.__traceback__))
            self._abort_subscriber(subscriber)

    def _remove_subscriber(self, subscriber: _Subscriber) -> None:
        """Unregister a subscriber and stop its sender task (called when its handler ends)."""
        subscriber.closed = True
        self.visualizer_clients.pop(subscriber.ws, None)
        if subscriber.task is not None and not subscriber.task.done():
            subscriber.task.cancel()

    async def close_all_clients(self, code: int = 1001, message: bytes = b'Server shutdown') -> None:
        """
        Close every WebSocket from the server side (used at shutdown).

        Peers get ``send_timeout`` seconds to complete the close handshake; a peer that does not (for
        example one that stopped reading) has its TCP connection dropped instead.
        """
        for subscriber in list(self.visualizer_clients.values()):
            # Stop the sender first so that it cannot be mid-frame when the close frame is written.
            self._remove_subscriber(subscriber)
        clients = list(self._transports)
        if not clients:
            return
        try:
            await asyncio.wait_for(
                asyncio.gather(*(ws.close(code=code, message=message) for ws in clients)),
                timeout=self.send_timeout)
        except asyncio.TimeoutError:
            logger.warning(
                f'Not every client completed the WebSocket close handshake within {self.send_timeout} s; '
                f'dropping the remaining connections')
            for ws in clients:
                transport = self._transports.get(ws)
                if transport is not None:
                    transport.abort()

    async def _on_shutdown(self, app: web.Application) -> None:
        """aiohttp shutdown hook: close the WebSockets so that their handlers finish promptly."""
        await self.close_all_clients()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def create_ssl_context(self):
        """Create SSL context if certificates exist"""
        cert_dir = Path(__file__).parent / 'certs'
        cert_file = cert_dir / 'cert.pem'
        key_file = cert_dir / 'key.pem'

        if cert_file.exists() and key_file.exists():
            try:
                ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
                ssl_context.load_cert_chain(str(cert_file), str(key_file))
                logger.info(f'✓ SSL certificates found, enabling HTTPS')
                return ssl_context
            except Exception as e:
                logger.warning(f'Failed to load SSL certificates: {e}')
                logger.warning(f'Falling back to HTTP')
                return None
        else:
            logger.info(f'No SSL certificates found at {cert_dir}')
            logger.info(f'Run "python generate_cert.py" to enable HTTPS')
            return None

    async def start_http_server(self, use_ssl: bool = True) -> web.AppRunner:
        """
        Start the HTTP/WebSocket server

        Args:
            use_ssl: serve HTTPS when certificates exist in backend/certs (the default). False forces
                plain HTTP, which is what tests and local tools without a headset use.

        Returns:
            The aiohttp runner (also kept in ``self.runner``); ``self.bound_port`` holds the real port.
        """
        if self.runner is not None:
            raise RuntimeError('Server is already started')

        runner = web.AppRunner(self.app)
        await runner.setup()

        # Try to create SSL context
        ssl_context = self.create_ssl_context() if use_ssl else None

        # Start server with or without SSL
        try:
            site = web.TCPSite(runner, self.host, self.http_port, ssl_context=ssl_context)
            await site.start()
        except BaseException:
            await runner.cleanup()
            raise
        self.runner = runner
        self.bound_port = runner.addresses[0][1]

        # Log appropriate URL
        protocol = 'https' if ssl_context else 'http'
        logger.info(f'{protocol.upper()} server started on {protocol}://{self.host}:{self.bound_port}')

        if ssl_context:
            # Get local IP address for Quest access
            import socket
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                local_ip = s.getsockname()[0]
                s.close()
            except Exception:
                local_ip = "<your-ip>"
            logger.info(f'✓ WebXR will work! Access from Quest: {protocol}://{local_ip}:{self.bound_port}')
        elif use_ssl:
            logger.warning(f'⚠ WebXR requires HTTPS! Generate certificates with: python generate_cert.py')

        return runner

    async def stop(self) -> None:
        """Close all client connections and stop the HTTP server (idempotent)."""
        runner, self.runner = self.runner, None
        if runner is None:
            return
        await self.close_all_clients()
        await runner.cleanup()
        self.bound_port = None

    async def run(self):
        """Run the server"""
        logger.info('Starting Controller Tracking Server...')
        await self.start_http_server()

        # Keep running until cancelled (Ctrl-C cancels the main task)
        try:
            await asyncio.Event().wait()
        finally:
            logger.info('Shutting down server...')
            await self.stop()


def load_server_config(path: Optional[Path] = None) -> Tuple[Dict[str, Any], List[str]]:
    """
    Read the ``server`` section of config.json.

    Args:
        path: explicit config file (must exist), or None for the repository's config.json (optional)

    Returns:
        (settings, ignored): ``settings`` holds the keys that were present out of ``host``, ``port`` and
        ``log_level``; ``ignored`` lists the other top-level sections found in the file, which this
        server does not implement.

    Raises:
        FileNotFoundError: an explicitly given file does not exist
        ValueError: the file is not valid JSON or a ``server`` setting is malformed or unknown
    """
    if path is None:
        if not DEFAULT_CONFIG_PATH.exists():
            return {}, []
        path = DEFAULT_CONFIG_PATH
    elif not path.exists():
        raise FileNotFoundError(f'Config file not found: {path}')

    try:
        config = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f'{path} is not valid JSON: {e}') from e
    if not isinstance(config, dict):
        raise ValueError(f'{path}: top level must be a JSON object')

    server = config.get('server', {})
    if not isinstance(server, dict):
        raise ValueError(f"{path}: 'server' must be a JSON object")

    unknown = sorted(set(server) - {'host', 'port', 'log_level'})
    if unknown:
        raise ValueError(f"{path}: unknown key(s) in 'server': {unknown} (supported: host, port, log_level)")

    settings: Dict[str, Any] = {}
    if 'host' in server:
        if not isinstance(server['host'], str) or not server['host']:
            raise ValueError(f"{path}: server.host must be a non-empty string, got {server['host']!r}")
        settings['host'] = server['host']
    if 'port' in server:
        port = server['port']
        if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
            raise ValueError(f'{path}: server.port must be an integer in 0..65535, got {port!r}')
        settings['port'] = port
    if 'log_level' in server:
        level = server['log_level']
        if not isinstance(level, str) or level.upper() not in VALID_LOG_LEVELS:
            raise ValueError(f'{path}: server.log_level must be one of {VALID_LOG_LEVELS}, got {level!r}')
        settings['log_level'] = level.upper()

    ignored = sorted(key for key in config if key != 'server')
    return settings, ignored


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse the command line (every option overrides the corresponding config.json setting)."""
    parser = argparse.ArgumentParser(description='WebXR controller tracking server')
    parser.add_argument('--config', type=Path, default=None,
                        help=f'config file (default: {DEFAULT_CONFIG_PATH} if it exists)')
    parser.add_argument('--host', default=None, help=f'interface to bind (default: {DEFAULT_HOST})')
    parser.add_argument('--port', type=int, default=None, help=f'port (default: {DEFAULT_PORT})')
    parser.add_argument('--log-level', default=None, type=str.upper, choices=VALID_LOG_LEVELS,
                        help=f'log level (default: {DEFAULT_LOG_LEVEL})')
    return parser.parse_args(argv)


async def main(argv: Optional[List[str]] = None):
    """Main entry point: command line > config.json > built-in defaults"""
    args = parse_args(argv)
    settings, ignored = load_server_config(args.config)

    host = args.host if args.host is not None else settings.get('host', DEFAULT_HOST)
    port = args.port if args.port is not None else settings.get('port', DEFAULT_PORT)
    log_level = args.log_level if args.log_level is not None else settings.get('log_level', DEFAULT_LOG_LEVEL)

    logging.basicConfig(level=getattr(logging, log_level), format=LOG_FORMAT)
    if ignored:
        logger.warning(
            f'config file section(s) {ignored} are not implemented by this server and have no effect '
            f"(only 'server.host', 'server.port' and 'server.log_level' are read)")

    server = ControllerTrackingServer(host=host, http_port=port)
    await server.run()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
