"""Lifecycle checks without a webcam, API key, or student image."""

import time
import unittest
from threading import Event

import numpy as np

from study_buddy_inference import InferenceResult
from study_session import SessionController


class FakeCamera:
    def __init__(self, opened=True, fail_read=False):
        self.opened = opened
        self.fail_read = fail_read
        self.released = False
        self.reads = 0

    def isOpened(self):
        return self.opened

    def read(self):
        self.reads += 1
        if self.fail_read:
            return False, None
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        frame[:, 0] = (10, 20, 30)
        frame[:, 1] = (40, 50, 60)
        return True, frame

    def release(self):
        self.released = True


class FakeClassifier:
    def __init__(self, observed):
        self.observed = observed
        self.frames = []

    def infer_frame(self, frame):
        if frame is None:
            return InferenceResult(None, None, "camera_error")
        self.frames.append(frame.copy())
        self.observed.set()
        return InferenceResult((0.01, 0.95, 0.01, 0.01, 0.01, 0.01),
                               "Confused", "ok", (0, 0, 2, 2))


def wait_for(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("Timed out waiting for camera worker")


class SessionTests(unittest.TestCase):
    def test_pause_resume_stop_releases_each_camera_and_reuses_classifier(self):
        cameras = []
        classifiers = []
        observed = Event()

        def make_camera(index):
            self.assertEqual(index, 0)
            camera = FakeCamera()
            cameras.append(camera)
            return camera

        def make_classifier():
            classifier = FakeClassifier(observed)
            classifiers.append(classifier)
            return classifier

        controller = SessionController(
            camera_factory=make_camera,
            classifier_factory=make_classifier,
            sample_interval=0.01,
        )
        try:
            started = controller.start("Math", "Practice derivatives", 25)
            self.assertTrue(observed.wait(2))
            with self.assertRaisesRegex(RuntimeError, "already owns"):
                controller.start("Physics", "Review forces", 25)
            self.assertEqual(controller.snapshot().tentative_class, "Confused")
            np.testing.assert_array_equal(classifiers[0].frames[0][0, 0], (40, 50, 60))

            paused = controller.pause()
            self.assertEqual(paused.session_id, started.session_id)
            self.assertEqual(paused.status, "paused")
            self.assertTrue(cameras[0].released)
            self.assertIsNone(paused.tentative_class)
            elapsed = paused.elapsed_seconds
            time.sleep(0.03)
            self.assertEqual(controller.snapshot().elapsed_seconds, elapsed)

            observed.clear()
            resumed = controller.resume()
            self.assertEqual(resumed.status, "running")
            self.assertTrue(observed.wait(2))
            stopped = controller.stop()
            self.assertEqual(stopped.status, "stopped")
            self.assertTrue(cameras[1].released)
            self.assertEqual(len(classifiers), 1)
            self.assertEqual(len(cameras), 2)
            self.assertGreaterEqual(stopped.elapsed_seconds, elapsed)
        finally:
            controller.shutdown()

    def test_camera_failure_is_reported_and_releases_camera(self):
        camera = FakeCamera(fail_read=True)
        controller = SessionController(
            camera_factory=lambda index: camera,
            classifier_factory=lambda: FakeClassifier(Event()),
            sample_interval=0.01,
        )
        try:
            controller.start("Math", "Review limits", 10)
            wait_for(lambda: controller.snapshot().status == "error")
            self.assertEqual(controller.snapshot().error, "Camera feed stopped")
            self.assertTrue(camera.released)
            self.assertIsNone(controller.snapshot().tentative_class)
        finally:
            controller.shutdown()

    def test_unavailable_camera_and_invalid_transitions(self):
        camera = FakeCamera(opened=False)
        controller = SessionController(
            camera_factory=lambda index: camera,
            classifier_factory=lambda: FakeClassifier(Event()),
        )
        with self.assertRaisesRegex(RuntimeError, "No study session"):
            controller.stop()
        with self.assertRaises(ValueError):
            controller.start("", "Review", 10)
        with self.assertRaises(ValueError):
            controller.start("Math", "Review", 0)
        try:
            controller.start("Math", "Review", 10)
            wait_for(lambda: controller.snapshot().status == "error")
            self.assertEqual(controller.snapshot().error, "Could not open the camera")
            self.assertTrue(camera.released)
            with self.assertRaisesRegex(RuntimeError, "paused"):
                controller.resume()
        finally:
            controller.shutdown()

    def test_duration_ends_capture_without_a_browser_or_api(self):
        current_time = [100.0]
        observed = Event()
        camera = FakeCamera()
        controller = SessionController(
            camera_factory=lambda index: camera,
            classifier_factory=lambda: FakeClassifier(observed),
            clock=lambda: current_time[0],
            sample_interval=0.01,
        )
        try:
            controller.start("Math", "Review", 1)
            self.assertTrue(observed.wait(2))
            current_time[0] = 161.0
            wait_for(lambda: controller.snapshot().status == "stopped")
            self.assertTrue(camera.released)
            self.assertEqual(controller.snapshot().elapsed_seconds, 60)
        finally:
            controller.shutdown()


if __name__ == "__main__":
    unittest.main()
