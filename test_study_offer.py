"""Offer timing checks using synthetic six-class probability timelines."""

import unittest

from study_buddy_logic import CLASS_NAMES
from study_offer import OfferPolicy


def scores(name: str) -> tuple[float, ...]:
    return tuple(0.9 if label == name else 0.02 for label in CLASS_NAMES)


class OfferTests(unittest.TestCase):
    def policy(self, **overrides):
        options = dict(window_seconds=2, min_samples=2, hold_seconds=1,
                       max_gap_seconds=1.5, clear_seconds=1,
                       cooldown_seconds=3)
        options.update(overrides)
        return OfferPolicy(**options)

    def test_sustained_scores_one_offer_per_episode_and_clear(self):
        policy = self.policy()
        self.assertIsNone(policy.observe(scores("Confused"), 0))
        self.assertIsNone(policy.observe(scores("Confused"), 0.5))
        self.assertIsNone(policy.observe(scores("Confused"), 1))
        offer = policy.observe(scores("Confused"), 1.5)
        self.assertEqual(offer.cue_class, "Confused")
        self.assertEqual(offer.guide_mode, "explain")
        self.assertEqual(offer.sample_count, 4)
        self.assertEqual(offer.window_seconds, 1.5)
        self.assertIsNone(policy.observe(scores("Confused"), 2))
        policy.act(offer.offer_id, "dismiss", 2)
        self.assertIsNone(policy.observe(scores("Confused"), 2.5))
        for now in (3, 3.5, 4, 4.5, 5):
            policy.observe(scores("Focused"), now)
        self.assertTrue(policy.armed)
        offers = [policy.observe(scores("Frustrated"), now)
                  for now in (5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9)]
        offers = [item for item in offers if item is not None]
        self.assertEqual(len(offers), 1)
        self.assertEqual(offers[0].guide_mode, "steps")

    def test_uncertain_no_face_gap_disabled_and_snooze(self):
        policy = self.policy()
        low = (0.2, 0.2, 0.1, 0.2, 0.2, 0.1)
        for now in (0, 0.5, 1, 1.5, 2):
            self.assertIsNone(policy.observe(low, now))
        self.assertIsNone(policy.observe(scores("Confused"), 2.5))
        self.assertIsNone(policy.observe(None, 4.1))
        self.assertIsNone(policy.observe(scores("Confused"), 4.5))
        policy.set_enabled({"Bored"})
        for now in (5, 5.5, 6, 6.5, 7):
            self.assertIsNone(policy.observe(scores("Confused"), now))
        policy.reset_evidence()
        for now in (8, 8.5, 9):
            self.assertIsNone(policy.observe(scores("Bored"), now))
        offer = policy.observe(scores("Bored"), 9.5)
        self.assertEqual(offer.guide_mode, "engage")
        policy.act(offer.offer_id, "snooze", 9.5)
        self.assertGreater(policy.snoozed_until, 300)
        for now in (10, 10.5, 11, 11.5, 12):
            policy.observe(scores("Focused"), now)
        for now in (13, 13.5, 14, 14.5, 15):
            self.assertIsNone(policy.observe(scores("Bored"), now))

    def test_drowsy_is_optional_and_invalid_actions_rejected(self):
        policy = self.policy(enabled=frozenset({"Drowsy"}))
        for now in (0, 0.5, 1):
            policy.observe(scores("Drowsy"), now)
        offer = policy.observe(scores("Drowsy"), 1.5)
        self.assertEqual(offer.guide_mode, "break")
        with self.assertRaises(ValueError):
            policy.act("wrong", "accept", 2)
        with self.assertRaises(ValueError):
            policy.act(offer.offer_id, "unknown", 2)
        self.assertEqual(policy.act(offer.offer_id, "wrong_cue", 2), offer)
        self.assertIsNone(policy.active)


if __name__ == "__main__":
    unittest.main()
