"""Local credential configuration without a provider request or real key."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from study_config import get_setting


class LocalConfigTests(unittest.TestCase):
    def test_ignored_file_supplies_both_values_and_process_environment_wins(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / ".env"
            config.write_text(
                "# Local test values\nGEMINI_API_KEY='file-test-key'\n"
                "STUDYSYNC_GEMINI_FREE_TIER_CONFIRMED=1\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(get_setting("GEMINI_API_KEY", config), "file-test-key")
                self.assertEqual(get_setting(
                    "STUDYSYNC_GEMINI_FREE_TIER_CONFIRMED", config,
                ), "1")
                os.environ["GEMINI_API_KEY"] = "environment-test-key"
                self.assertEqual(get_setting("GEMINI_API_KEY", config), "environment-test-key")

    def test_missing_file_does_not_confirm_free_tier(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / ".env"
            with patch.dict(os.environ, {}, clear=True):
                self.assertIsNone(get_setting("GEMINI_API_KEY", missing))
                self.assertIsNone(get_setting(
                    "STUDYSYNC_GEMINI_FREE_TIER_CONFIRMED", missing,
                ))
            with self.assertRaises(ValueError):
                get_setting("UNKNOWN_SETTING", missing)


if __name__ == "__main__":
    unittest.main()
