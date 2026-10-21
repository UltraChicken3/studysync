"""Local API and WebSocket checks with no webcam or Gemini key."""

import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from study_buddy_inference import InferenceResult
from study_session import SessionController
from study_web import create_app


class Camera:
    def __init__(self):
        self.released = False

    def isOpened(self):
        return True

    def read(self):
        return True, np.zeros((4, 4, 3), dtype=np.uint8)

    def release(self):
        self.released = True


class Classifier:
    def infer_frame(self, frame):
        return InferenceResult((0.01, 0.95, 0.01, 0.01, 0.01, 0.01),
                               "Confused", "ok", (0, 0, 4, 4))


def wait_for(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("Timed out waiting for session update")


class WebApiTests(unittest.TestCase):
    def setUp(self):
        self.cameras = []

        def make_camera(index):
            camera = Camera()
            self.cameras.append(camera)
            return camera

        self.controller = SessionController(
            camera_factory=make_camera,
            classifier_factory=Classifier,
            sample_interval=0.01,
        )
        self.temp = TemporaryDirectory()
        self.app = create_app(
            controller=self.controller,
            static_dir=Path(self.temp.name) / "missing",
            allowed_hosts=("testserver",),
            disconnect_grace_seconds=0.12,
        )
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.token = self.client.get("/bootstrap").json()["token"]
        self.auth = {"X-StudySync-Token": self.token}

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.temp.cleanup()

    def start(self):
        response = self.client.post(
            "/sessions",
            headers=self.auth,
            json={"subject": "Math", "goal": "Practice derivatives", "duration_minutes": 20},
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["session_id"]

    def test_control_routes_reject_invalid_origin_host_token_and_size(self):
        self.assertEqual(self.client.get("/health").json()["session_status"], "idle")
        self.assertEqual(self.client.get("/sessions/current").status_code, 403)
        body = {"subject": "Math", "goal": "Review", "duration_minutes": 10}
        self.assertEqual(self.client.post("/sessions", json=body).status_code, 403)
        self.assertEqual(self.client.post(
            "/sessions", headers={**self.auth, "Origin": "https://example.com"},
            json=body,
        ).status_code, 403)
        self.assertEqual(self.client.post(
            "/sessions", headers={**self.auth, "Host": "example.com"},
            json=body,
        ).status_code, 403)
        self.assertEqual(self.client.post(
            "/sessions", headers={**self.auth, "Sec-Fetch-Site": "cross-site"},
            json=body,
        ).status_code, 403)
        self.assertEqual(self.client.post(
            "/sessions", headers=self.auth,
            content=b"x" * 4097,
        ).status_code, 413)
        self.assertIsNone(self.client.get("/sessions/current", headers=self.auth).json())

    def test_reconnect_does_not_start_second_camera_and_disconnect_pauses(self):
        session_id = self.start()
        wait_for(lambda: len(self.cameras) == 1)
        self.assertEqual(self.client.get(
            f"/sessions/{session_id}", headers=self.auth
        ).json()["session_id"], session_id)
        self.assertEqual(self.client.post(
            "/sessions", headers=self.auth,
            json={"subject": "Other", "goal": "Review", "duration_minutes": 10},
        ).status_code, 409)

        with self.client.websocket_connect(f"/sessions/{session_id}/events") as ws:
            ws.send_json({"token": self.token})
            event = ws.receive_json()
            self.assertEqual(event["session_id"], session_id)
            self.assertEqual(event["type"], "snapshot")
            self.assertNotIn("scores", event["payload"])
            self.assertNotIn("frame", event["payload"])
        with self.client.websocket_connect(f"/sessions/{session_id}/events") as ws:
            ws.send_json({"token": self.token})
            self.assertEqual(ws.receive_json()["session_id"], session_id)
            time.sleep(0.15)
            self.assertEqual(self.controller.snapshot().status, "running")
            self.assertEqual(len(self.cameras), 1)

        wait_for(lambda: self.controller.snapshot().status == "paused")
        self.assertTrue(self.cameras[0].released)
        resumed = self.client.post(
            f"/sessions/{session_id}/resume", headers=self.auth
        )
        self.assertEqual(resumed.status_code, 200)
        wait_for(lambda: len(self.cameras) == 2)
        stopped = self.client.post(
            f"/sessions/{session_id}/stop", headers=self.auth
        )
        self.assertEqual(stopped.json()["status"], "stopped")
        self.assertTrue(self.cameras[1].released)

    def test_websocket_auth_and_shutdown_release(self):
        session_id = self.start()
        wait_for(lambda: len(self.cameras) == 1)
        with self.client.websocket_connect(f"/sessions/{session_id}/events") as ws:
            ws.send_json({"token": "wrong"})
            with self.assertRaises(WebSocketDisconnect):
                ws.receive_json()
        with self.assertRaises(WebSocketDisconnect):
            with self.client.websocket_connect(
                f"/sessions/{session_id}/events",
                headers={"Origin": "https://example.com"},
            ):
                pass
        self.client.__exit__(None, None, None)
        self.assertTrue(self.cameras[0].released)
        self.client = TestClient(self.app)
        self.client.__enter__()

    def test_old_session_tab_does_not_keep_new_camera_session_running(self):
        first_id = self.start()
        with self.client.websocket_connect(f"/sessions/{first_id}/events") as old_ws:
            old_ws.send_json({"token": self.token})
            old_ws.receive_json()
            self.client.post(f"/sessions/{first_id}/stop", headers=self.auth)
            second_id = self.start()
            self.assertNotEqual(first_id, second_id)
            wait_for(lambda: self.controller.snapshot().status == "paused")
            self.assertTrue(self.cameras[-1].released)


if __name__ == "__main__":
    unittest.main()
