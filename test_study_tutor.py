"""Tutor contract checks with no webcam, key, or remote request."""

import asyncio
import unittest

from study_tutor import (
    TutorError, TutorService, make_prompt, make_followup_prompt, prompt_digest,
)


class TutorTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.now = [100.0]
        self.calls = []

        async def generate(prompt, key, timeout):
            self.calls.append((prompt, key, timeout))
            return "A short guide"

        self.service = TutorService(
            generate=generate, key=lambda: "private-test-key",
            confirmed=lambda: True, clock=lambda: self.now[0],
            per_session_limit=2, rolling_day_limit=3, min_interval_seconds=60,
        )

    async def ask(self, request_id="a" * 32, session_id="s", mode="explain", topic="Limits"):
        prompt = make_prompt(mode, topic)
        return await self.service.ask(session_id, request_id, mode, topic, "", prompt_digest(prompt))

    async def test_class_specific_prompts_and_only_approved_text(self):
        explain = make_prompt("explain", "Derivatives", "My textbook excerpt")
        steps = make_prompt("steps", "Derivatives")
        engage = make_prompt("engage", "Derivatives")
        self.assertIn("plain language", explain)
        self.assertIn("concrete example", explain)
        self.assertIn("small achievable steps", steps)
        self.assertIn("next action", steps)
        self.assertIn("analogy", engage)
        self.assertIn("interactive challenge", engage)
        self.assertIn("My textbook excerpt", explain)
        result = await self.ask()
        self.assertEqual(result["answer"], "A short guide")
        self.assertEqual(self.calls[0][0], make_prompt("explain", "Limits"))
        self.assertNotIn("private-test-key", str(result))
        with self.assertRaisesRegex(TutorError, "preview"):
            await self.service.ask("s", "b" * 32, "explain", "Different", "", prompt_digest(explain))
        self.assertEqual(len(self.calls), 1)

    async def test_budgets_cooldown_and_duplicate_ids(self):
        await self.ask()
        with self.assertRaises(TutorError) as duplicate:
            await self.ask()
        self.assertEqual(duplicate.exception.code, "duplicate_request")
        with self.assertRaises(TutorError) as cooldown:
            await self.ask("b" * 32)
        self.assertEqual(cooldown.exception.code, "cooldown")
        self.now[0] += 60
        await self.ask("b" * 32)
        self.now[0] += 60
        with self.assertRaises(TutorError) as session_budget:
            await self.ask("c" * 32)
        self.assertEqual(session_budget.exception.code, "session_budget")
        await self.ask("c" * 32, session_id="other")
        self.now[0] += 60
        with self.assertRaises(TutorError) as daily_budget:
            await self.ask("d" * 32, session_id="third")
        self.assertEqual(daily_budget.exception.code, "daily_budget")
        self.assertEqual(len(self.calls), 3)

    async def test_missing_key_and_confirmation_block_remote_call(self):
        self.service.key = lambda: None
        with self.assertRaises(TutorError) as missing:
            await self.ask()
        self.assertEqual(missing.exception.code, "missing_key")
        self.service.key = lambda: "private-test-key"
        self.service.confirmed = lambda: False
        with self.assertRaises(TutorError) as unconfirmed:
            await self.ask()
        self.assertEqual(unconfirmed.exception.code, "free_tier_unconfirmed")
        self.assertEqual(self.calls, [])

    async def test_followup_requires_completed_guide_and_uses_same_budget(self):
        context = "### Quick challenge\nWhich direction should the weight change?"
        reply = "Decrease it."
        prompt = make_followup_prompt(context, reply)
        self.assertIn(context, prompt)
        self.assertIn(reply, prompt)
        self.assertIn("Do not ask a new question", prompt)
        with self.assertRaises(TutorError) as missing:
            await self.service.ask_followup("s", "a" * 32, "b" * 32,
                                            context, reply, prompt_digest(prompt))
        self.assertEqual(missing.exception.code, "guide_not_found")
        await self.ask()
        self.now[0] += 60
        result = await self.service.ask_followup(
            "s", "a" * 32, "b" * 32, context, reply, prompt_digest(prompt),
        )
        self.assertEqual(result["mode"], "followup")
        self.assertEqual(self.calls[-1][0], prompt)
        self.now[0] += 60
        with self.assertRaises(TutorError) as budget:
            await self.service.ask_followup(
                "s", "a" * 32, "c" * 32, context, reply, prompt_digest(prompt),
            )
        self.assertEqual(budget.exception.code, "session_budget")

    async def test_concurrency_timeout_quota_empty_and_secret_redaction(self):
        waiting = asyncio.Event()
        release = asyncio.Event()

        async def slow(prompt, key, timeout):
            waiting.set()
            await release.wait()
            return "Done"

        self.service.generate = slow
        first = asyncio.create_task(self.ask())
        await waiting.wait()
        with self.assertRaises(TutorError) as busy:
            await self.ask("b" * 32)
        self.assertEqual(busy.exception.code, "busy")
        release.set()
        await first

        self.now[0] += 60
        async def never(prompt, key, timeout):
            await asyncio.sleep(1)
            return "Late"
        self.service.generate = never
        self.service.timeout_seconds = 0.01
        with self.assertRaises(TutorError) as timeout:
            await self.ask("b" * 32)
        self.assertEqual(timeout.exception.code, "timeout")

        self.now[0] += 60
        class Quota(Exception):
            code = 429
        async def quota(prompt, key, timeout):
            raise Quota("private-test-key leaked in provider exception")
        self.service.generate = quota
        with self.assertRaises(TutorError) as limit:
            await self.ask("c" * 32, session_id="other")
        self.assertEqual(limit.exception.code, "provider_quota")
        self.assertNotIn("private-test-key", limit.exception.message)

        self.now[0] += 60
        self.service.rolling_day_limit = 5
        async def empty(prompt, key, timeout):
            return "  "
        self.service.generate = empty
        with self.assertRaises(TutorError) as no_answer:
            await self.ask("d" * 32, session_id="third")
        self.assertEqual(no_answer.exception.code, "empty_answer")

    def test_feedback_stores_counters_only(self):
        result = self.service.feedback("s", "useful")
        self.assertEqual(result["count"], 1)
        self.assertEqual(self.service._feedback["s"]["useful"], 1)
        self.assertNotIn("answer", str(self.service._feedback))


if __name__ == "__main__":
    unittest.main()
