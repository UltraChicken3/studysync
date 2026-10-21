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
from study_offer import OfferPolicy
from study_tutor import TutorService
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
            offer_policy_factory=lambda: OfferPolicy(
                window_seconds=0.1, min_samples=2, hold_seconds=0.02,
                max_gap_seconds=0.1, clear_seconds=0.03, cooldown_seconds=0.2,
            ),
        )
        self.tutor_calls = []

        async def fake_tutor(prompt, key, timeout):
            self.tutor_calls.append(prompt)
            return "Use the chain rule."

        self.tutor = TutorService(
            generate=fake_tutor, key=lambda: "private-test-key", confirmed=lambda: True,
            min_interval_seconds=0,
        )
        self.temp = TemporaryDirectory()
        self.app = create_app(
            controller=self.controller,
            static_dir=Path(self.temp.name) / "missing",
            allowed_hosts=("testserver",),
            disconnect_grace_seconds=0.12,
            tutor=self.tutor,
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
            content=b"x" * 16385,
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

    def test_camera_preview_requires_opt_in_and_reuses_capture(self):
        session_id = self.start()
        preview_url = f"/sessions/{session_id}/preview"
        with self.client.websocket_connect(preview_url) as ws:
            ws.send_json({"token": "wrong"})
            with self.assertRaises(WebSocketDisconnect):
                ws.receive_bytes()
        with self.client.websocket_connect(preview_url) as ws:
            ws.send_json({"token": self.token})
            jpeg = ws.receive_bytes()
            self.assertTrue(jpeg.startswith(b"\xff\xd8"))
            self.assertTrue(jpeg.endswith(b"\xff\xd9"))
            self.assertEqual(len(self.cameras), 1)
            self.assertEqual(self.client.post(
                f"/sessions/{session_id}/pause", headers=self.auth,
            ).json()["status"], "paused")
        self.assertTrue(self.cameras[0].released)
        with self.assertRaises(WebSocketDisconnect):
            with self.client.websocket_connect(preview_url):
                pass

    def test_old_session_tab_does_not_keep_new_camera_session_running(self):
        first_id = self.start()
        with self.client.websocket_connect(f"/sessions/{first_id}/events") as old_ws:
            old_ws.send_json({"token": self.token})
            old_ws.receive_json()
            self.client.post(f"/sessions/{first_id}/stop", headers=self.auth)
            second_id = self.start()
            self.assertNotEqual(first_id, second_id)
            wait_for(lambda: self.controller.snapshot().status == "paused")
            wait_for(lambda: self.cameras[-1].released)

    def test_manual_help_works_with_camera_off_and_requires_preview(self):
        response = self.client.post("/sessions", headers=self.auth, json={
            "subject": "Math", "goal": "Review", "duration_minutes": 20,
            "monitoring": False,
        })
        self.assertEqual(response.status_code, 200)
        session_id = response.json()["session_id"]
        self.assertEqual(response.json()["status"], "study_only")
        self.assertEqual(self.cameras, [])
        help_url = f"/sessions/{session_id}/help"
        body = {"mode": "steps", "topic": "Chain rule", "excerpt": "d/dx f(g(x))"}
        preview = self.client.post(f"{help_url}/preview", headers=self.auth, json=body)
        self.assertEqual(preview.status_code, 200, preview.text)
        self.assertIn("small achievable steps", preview.json()["prompt"])
        self.assertNotIn("private-test-key", preview.text)
        self.assertEqual(self.client.post(help_url, json=body).status_code, 403)
        mismatch = self.client.post(help_url, headers=self.auth, json={
            **body, "request_id": "a" * 32, "approved_digest": "0" * 64,
        })
        self.assertEqual(mismatch.status_code, 409)
        guide = self.client.post(help_url, headers=self.auth, json={
            **body, "request_id": "a" * 32,
            "approved_digest": preview.json()["digest"],
        })
        self.assertEqual(guide.status_code, 200, guide.text)
        self.assertEqual(guide.json()["answer"], "Use the chain rule.")
        self.assertEqual(self.tutor_calls, [preview.json()["prompt"]])
        self.assertEqual(self.cameras, [])
        feedback = self.client.post(
            f"/sessions/{session_id}/feedback", headers=self.auth,
            json={"choice": "useful"},
        )
        self.assertEqual(feedback.json(), {"recorded": True, "count": 1})
        self.assertNotIn("private-test-key", guide.text)
        self.assertEqual(self.client.post(f"{help_url}/preview", headers=self.auth, json={
            "mode": "unknown", "topic": "Chain rule",
        }).status_code, 422)

    def test_model_offer_requires_acceptance_before_guide(self):
        session_id = self.start()
        with self.client.websocket_connect(f"/sessions/{session_id}/events") as ws:
            ws.send_json({"token": self.token})
            ws.receive_json()
            wait_for(lambda: self.controller.snapshot().offer is not None)
            offer = self.controller.snapshot().offer
            self.assertEqual(offer["cue_class"], "Confused")
            self.assertEqual(offer["guide_mode"], "explain")
            self.assertNotIn("scores", offer)
            self.assertEqual(self.client.get(
                f"/sessions/{session_id}", headers=self.auth,
            ).json()["offer"]["offer_id"], offer["offer_id"])
            body = {"mode": "explain", "topic": "Derivatives"}
            path = f"/sessions/{session_id}/help"
            preview = self.client.post(f"{path}/preview", headers=self.auth, json=body).json()
            help_body = {**body, "request_id": "e" * 32,
                         "approved_digest": preview["digest"],
                         "offer_id": offer["offer_id"]}
            blocked = self.client.post(path, headers=self.auth, json=help_body)
            self.assertEqual(blocked.status_code, 409)
            accepted = self.client.post(
                f"/sessions/{session_id}/offers/{offer['offer_id']}/action",
                headers=self.auth, json={"action": "accept"},
            )
            self.assertEqual(accepted.status_code, 200, accepted.text)
            guide = self.client.post(path, headers=self.auth, json=help_body)
            self.assertEqual(guide.status_code, 200, guide.text)
            self.assertEqual(len(self.tutor_calls), 1)
            self.assertEqual(self.controller.snapshot().status, "running")

    def test_stopwatch_and_approved_reply_without_camera(self):
        started = self.client.post("/sessions", headers=self.auth, json={
            "subject": "Math", "goal": "Review", "timer_mode": "stopwatch",
            "duration_minutes": None, "monitoring": False,
        })
        self.assertEqual(started.status_code, 200, started.text)
        snapshot = started.json()
        self.assertEqual(snapshot["status"], "study_only")
        self.assertIsNone(snapshot["duration_seconds"])
        self.assertEqual(snapshot["timer_mode"], "stopwatch")
        self.assertEqual(self.cameras, [])
        session_id = snapshot["session_id"]
        url = f"/sessions/{session_id}/help"
        body = {"mode": "engage", "topic": "Backpropagation"}
        preview = self.client.post(f"{url}/preview", headers=self.auth, json=body).json()
        initial_id = "f" * 32
        guide = self.client.post(url, headers=self.auth, json={
            **body, "request_id": initial_id, "approved_digest": preview["digest"],
        })
        self.assertEqual(guide.status_code, 200, guide.text)
        follow_url = f"/sessions/{session_id}/followup"
        follow_body = {"guide_request_id": initial_id,
                       "guide_context": guide.json()["answer"],
                       "reply": "Decrease the positive contribution."}
        bad_guide = self.client.post(f"{follow_url}/preview", headers=self.auth, json={
            **follow_body, "guide_request_id": "0" * 32,
        })
        self.assertEqual(bad_guide.status_code, 409)
        follow_preview = self.client.post(
            f"{follow_url}/preview", headers=self.auth, json=follow_body,
        )
        self.assertEqual(follow_preview.status_code, 200, follow_preview.text)
        self.assertIn("Decrease the positive contribution.", follow_preview.json()["prompt"])
        mismatch = self.client.post(follow_url, headers=self.auth, json={
            **follow_body, "request_id": "d" * 32, "approved_digest": "0" * 64,
        })
        self.assertEqual(mismatch.status_code, 409)
        result = self.client.post(follow_url, headers=self.auth, json={
            **follow_body, "request_id": "d" * 32,
            "approved_digest": follow_preview.json()["digest"],
        })
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(result.json()["mode"], "followup")
        self.assertEqual(self.tutor_calls[-1], follow_preview.json()["prompt"])
        self.assertEqual(len(self.tutor_calls), 2)
        self.assertEqual(self.cameras, [])


if __name__ == "__main__":
    unittest.main()
