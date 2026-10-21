"""Synthetic checks for the saved model's inference contract."""

import unittest

import numpy as np

from study_buddy_inference import (
    DEFAULT_MODEL_PATH,
    StudentStateClassifier,
    preprocess_face,
)
from study_buddy_logic import CLASS_NAMES


class FakeCascade:
    def __init__(self, faces):
        self.faces = faces
        self.calls = 0

    def empty(self):
        return False

    def detectMultiScale(self, gray, **kwargs):
        self.calls += 1
        return self.faces


class FakeModel:
    input_shape = (None, 224, 224, 3)
    output_shape = (None, 6)

    def __init__(self, scores):
        self.scores = np.asarray(scores, dtype=np.float32)
        self.calls = 0
        self.batch = None

    def predict(self, batch, verbose=0):
        self.calls += 1
        self.batch = batch
        return self.scores[None, :]


class InferenceTests(unittest.TestCase):
    def test_crop_color_order_and_resnet_preprocessing(self):
        frame = np.zeros((6, 6, 3), dtype=np.uint8)
        frame[1:3, 2:4] = (10, 20, 30)  # OpenCV BGR

        batch = preprocess_face(frame, (2, 1, 2, 2))

        self.assertEqual(batch.shape, (1, 224, 224, 3))
        self.assertEqual(batch.dtype, np.float32)
        # ResNet50 preprocess_input converts RGB back to BGR, then subtracts
        # ImageNet channel means. The result therefore follows the BGR values.
        np.testing.assert_allclose(
            batch[0, 0, 0],
            [10 - 103.939, 20 - 116.779, 30 - 123.68],
            rtol=0,
            atol=1e-3,
        )
        np.testing.assert_allclose(batch[0, -1, -1], batch[0, 0, 0])

    def test_largest_face_and_class_mapping(self):
        frame = np.zeros((6, 6, 3), dtype=np.uint8)
        frame[1:4, 2:5] = (10, 20, 30)
        cascade = FakeCascade([(0, 0, 1, 1), (2, 1, 3, 3)])
        model = FakeModel([0.01, 0.95, 0.01, 0.01, 0.01, 0.01])
        classifier = StudentStateClassifier(model=model, cascade=cascade)

        result = classifier.infer_frame(frame)

        self.assertEqual(result.capture_status, "ok")
        self.assertEqual(result.face_box, (2, 1, 3, 3))
        self.assertEqual(result.tentative_class, CLASS_NAMES[1])
        self.assertEqual(len(result.scores), 6)
        self.assertEqual(model.calls, 1)
        self.assertEqual(cascade.calls, 1)
        np.testing.assert_allclose(
            model.batch[0, 0, 0],
            [10 - 103.939, 20 - 116.779, 30 - 123.68],
            rtol=0,
            atol=1e-3,
        )

        for index, expected_class in enumerate(CLASS_NAMES):
            scores = np.full(6, 0.01, dtype=np.float32)
            scores[index] = 0.95
            model.scores = scores
            self.assertEqual(classifier.infer_frame(frame).tentative_class, expected_class)

    def test_no_face_and_camera_error_are_distinct(self):
        cascade = FakeCascade([])
        model = FakeModel([0.01, 0.95, 0.01, 0.01, 0.01, 0.01])
        classifier = StudentStateClassifier(model=model, cascade=cascade)

        no_face = classifier.infer_frame(np.zeros((4, 4, 3), dtype=np.uint8))
        camera_error = classifier.infer_frame(None)

        self.assertEqual((no_face.capture_status, no_face.tentative_class, no_face.scores),
                         ("no_face", None, None))
        self.assertEqual((camera_error.capture_status, camera_error.tentative_class,
                          camera_error.scores), ("camera_error", None, None))
        self.assertEqual(model.calls, 0)
        self.assertEqual(cascade.calls, 1)

    def test_uncertain_is_not_a_seventh_model_class(self):
        model = FakeModel([0.30, 0.24, 0.14, 0.14, 0.10, 0.08])
        classifier = StudentStateClassifier(
            model=model, cascade=FakeCascade([(0, 0, 4, 4)])
        )

        result = classifier.infer_frame(np.zeros((4, 4, 3), dtype=np.uint8))

        self.assertEqual(result.capture_status, "ok")
        self.assertEqual(result.tentative_class, "Uncertain")
        self.assertEqual(len(result.scores), len(CLASS_NAMES))

    def test_saved_model_smoke_with_synthetic_frame(self):
        classifier = StudentStateClassifier(
            DEFAULT_MODEL_PATH,
            cascade=FakeCascade([(0, 0, 224, 224)]),
        )

        result = classifier.infer_frame(np.zeros((224, 224, 3), dtype=np.uint8))

        self.assertEqual(classifier.model.input_shape, (None, 224, 224, 3))
        self.assertEqual(classifier.model.output_shape, (None, 6))
        self.assertEqual(result.capture_status, "ok")
        self.assertEqual(len(result.scores), 6)
        self.assertTrue(np.isfinite(result.scores).all())
        self.assertAlmostEqual(sum(result.scores), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
