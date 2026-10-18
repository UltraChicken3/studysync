import unittest

from study_buddy_logic import CueTracker, classify_probabilities, help_mode


class CueLogicTests(unittest.TestCase):
    def test_low_confidence_and_small_margin_do_not_label_student(self):
        self.assertEqual(classify_probabilities([.30, .24, .14, .14, .10, .08]), "Uncertain")
        self.assertEqual(classify_probabilities([.66, .21, .03, .03, .04, .03]), "Bored")
        self.assertEqual(classify_probabilities([.66, .22, .02, .02, .04, .04], min_margin=.5), "Uncertain")

    def test_offer_requires_time_then_cue_clear_and_cooldown(self):
        tracker = CueTracker(hold_seconds=2, reset_seconds=2, cooldown_seconds=60)
        self.assertIsNone(tracker.observe("confusion", 0))
        self.assertIsNone(tracker.observe("confusion", 1.9))
        self.assertEqual(tracker.observe("confusion", 2), "confusion")
        self.assertIsNone(tracker.observe("confusion", 100))
        self.assertIsNone(tracker.observe(None, 101))
        self.assertIsNone(tracker.observe(None, 103))
        self.assertIsNone(tracker.observe("confusion", 104))
        self.assertEqual(tracker.observe("confusion", 106), "confusion")

    def test_frustration_uses_explanation_and_missing_face_clears_cue(self):
        self.assertEqual(help_mode("Frustrated"), "confusion")
        self.assertIsNone(help_mode("Looking Away"))
        tracker = CueTracker(hold_seconds=2)
        tracker.observe("boredom", 0)
        tracker.observe(None, 1)
        self.assertIsNone(tracker.observe("boredom", 2))
        self.assertEqual(tracker.observe("boredom", 4), "boredom")


if __name__ == "__main__":
    unittest.main()
