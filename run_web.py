"""Run StudySync on loopback; open http://127.0.0.1:8765 afterward."""

import argparse
import os

import uvicorn

from study_web import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local StudySync web app")
    parser.add_argument("--demo", action="store_true", help="Use synthetic data; no camera or model inference")
    args = parser.parse_args()
    port = int(os.environ.get("STUDYSYNC_PORT", "8765"))
    dev_origin = os.environ.get("STUDYSYNC_DEV_ORIGIN")
    controller = None
    if args.demo:
        from study_dev import SyntheticCamera, SyntheticClassifier
        from study_session import SessionController

        controller = SessionController(
            camera_factory=lambda index: SyntheticCamera(),
            classifier_factory=SyntheticClassifier,
        )
    uvicorn.run(
        create_app(controller=controller, dev_origin=dev_origin, demo_mode=args.demo),
        host="127.0.0.1",
        port=port,
        access_log=False,
    )


if __name__ == "__main__":
    main()
