"""Optional smoke test for the configured Gemini account.

Set GEMINI_API_KEY in the environment before running this script.
"""

import os

from google import genai


def main():
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit("Set GEMINI_API_KEY before running this test.")
    model = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
    client = genai.Client(api_key=key)
    response = client.models.generate_content(
        model=model, contents="Reply with: Study Buddy API ready."
    )
    print(response.text)


if __name__ == "__main__":
    main()
