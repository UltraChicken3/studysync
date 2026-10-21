"""Study Buddy: tentative webcam cues and student-controlled study help.

Set GEMINI_API_KEY in the environment before requesting generated answers.
The camera classifier is a cue for offering help, never proof of a mental state.
"""

import os
import queue
import threading
import time
from pathlib import Path

import cv2
import customtkinter as ctk

from study_buddy_inference import StudentStateClassifier
from study_buddy_logic import CueTracker, help_mode


ROOT = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("STUDY_BUDDY_MODEL", ROOT / "study_buddy_resnet_final.keras"))
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
CAMERA_INDEX = int(os.environ.get("STUDY_BUDDY_CAMERA", "0"))


def make_prompt(mode, topic):
    if mode == "confusion":
        return (
            "A student asks for help with the following topic or question. "
            "Explain it in plain language, give one concrete example, then ask one "
            "short check-for-understanding question. If the question is ambiguous, "
            "state your assumption. Keep the reply concise.\n\n"
            f"Student's question: {topic}"
        )
    return (
        "A student wants to make the following study topic more engaging. "
        "Give one surprising connection or analogy and one quick interactive "
        "challenge. Keep facts accurate and the reply concise.\n\n"
        f"Student's topic: {topic}"
    )


def generate_answer(mode, topic):
    # Keep the optional API import out of the webcam startup path.
    from google import genai

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=make_prompt(mode, topic),
    )
    answer = (response.text or "").strip()
    if not answer:
        raise RuntimeError("The model returned no text. Try a more specific question.")
    return answer


class HelpPanel:
    def __init__(self):
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("dark-blue")
        self.app = ctk.CTk()
        self.app.title("Study Buddy")
        self.app.geometry("650x630")
        self.app.minsize(520, 520)
        self.app.grid_columnconfigure(0, weight=1)
        self.app.grid_rowconfigure(5, weight=1)
        self.closed = False
        self.busy = False
        self.offer_active = False
        self.results = queue.Queue()
        self.app.protocol("WM_DELETE_WINDOW", self.close)

        ctk.CTkLabel(self.app, text="Study Buddy", font=("Arial", 27, "bold"),
                     anchor="w").grid(row=0, column=0, padx=24, pady=(22, 4), sticky="ew")
        self.status = ctk.CTkLabel(
            self.app, text="Starting camera...", anchor="w", wraplength=460,
            text_color="#aab6c4"
        )
        self.status.grid(row=1, column=0, padx=24, pady=(0, 14), sticky="ew")

        self.offer = ctk.CTkLabel(
            self.app, text="Ask a question whenever you want.", anchor="w",
            wraplength=460, font=("Arial", 17, "bold")
        )
        self.offer.grid(row=2, column=0, padx=24, pady=(0, 12), sticky="ew")

        form = ctk.CTkFrame(self.app)
        form.grid(row=3, column=0, padx=24, pady=0, sticky="ew")
        form.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(form, text="What are you working on?", anchor="w").grid(
            row=0, column=0, padx=16, pady=(14, 3), sticky="ew"
        )
        self.topic = ctk.CTkEntry(
            form, placeholder_text="Enter a topic or a specific question", height=42
        )
        self.topic.grid(row=1, column=0, padx=16, pady=(0, 12), sticky="ew")
        self.topic.bind("<Return>", lambda event: self.ask())
        self.mode = ctk.CTkSegmentedButton(
            form, values=["Explain simply", "Make it engaging"]
        )
        self.mode.set("Explain simply")
        self.mode.grid(row=2, column=0, padx=16, pady=(0, 14), sticky="ew")

        actions = ctk.CTkFrame(self.app, fg_color="transparent")
        actions.grid(row=4, column=0, padx=24, pady=14, sticky="ew")
        actions.grid_columnconfigure(0, weight=1)
        self.ask_button = ctk.CTkButton(
            actions, text="Get help", height=42, command=self.ask
        )
        self.ask_button.grid(row=0, column=0, padx=(0, 8), sticky="ew")
        ctk.CTkButton(
            actions, text="Not now", height=42, width=105,
            fg_color="#394553", hover_color="#4b5a68", command=self.dismiss
        ).grid(row=0, column=1)

        self.answer = ctk.CTkTextbox(self.app, wrap="word", font=("Arial", 15))
        self.answer.grid(row=5, column=0, padx=24, pady=(0, 24), sticky="nsew")
        self.show_answer("When you press Get help, your typed question is sent to Gemini. Camera frames stay on this computer.")

    def show_answer(self, message):
        self.answer.configure(state="normal")
        self.answer.delete("1.0", "end")
        self.answer.insert("1.0", message)
        self.answer.configure(state="disabled")

    def set_status(self, message):
        if self.status.cget("text") != message:
            self.status.configure(text=message)

    def suggest(self, mode):
        if self.busy or self.offer_active:
            return
        self.offer_active = True
        if mode == "confusion":
            self.mode.set("Explain simply")
            message = "Need a hand? Ask about the part that feels unclear."
        else:
            self.mode.set("Make it engaging")
            message = "Want a fresh angle on what you're studying?"
        self.offer.configure(text=message)
        self.app.lift()

    def dismiss(self):
        self.offer_active = False
        self.offer.configure(text="Ask a question whenever you want.")

    def ask(self):
        if self.busy:
            return
        topic = self.topic.get().strip()
        if not topic:
            self.show_answer("Enter a topic or question first.")
            self.topic.focus_set()
            return
        if not os.environ.get("GEMINI_API_KEY"):
            self.show_answer("Set the GEMINI_API_KEY environment variable to enable generated help.")
            return
        mode = "confusion" if self.mode.get() == "Explain simply" else "boredom"
        self.dismiss()
        self.busy = True
        self.ask_button.configure(state="disabled", text="Working...")
        self.show_answer("Writing an answer...")

        def worker():
            try:
                self.results.put((True, generate_answer(mode, topic)))
            except Exception as exc:
                self.results.put((False, str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def tick(self):
        try:
            while True:
                success, message = self.results.get_nowait()
                self.busy = False
                self.ask_button.configure(state="normal", text="Get help")
                self.show_answer(message if success else f"Could not get an answer: {message}")
        except queue.Empty:
            pass
        self.app.update()

    def close(self):
        self.closed = True
        self.app.destroy()


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    classifier = StudentStateClassifier(MODEL_PATH)
    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera {CAMERA_INDEX}")

    panel = HelpPanel()
    tracker = CueTracker()
    try:
        while not panel.closed:
            ok, frame = camera.read()
            frame = cv2.flip(frame, 1) if ok and frame is not None and frame.size else None
            result = classifier.infer_frame(frame)
            now = time.monotonic()
            if result.capture_status == "camera_error":
                panel.set_status("Camera feed stopped.")
                break
            if result.capture_status == "no_face":
                state = "Face not visible"
                tracker.observe(None, now)
            else:
                state = result.tentative_class
                offer = tracker.observe(help_mode(state), now)
                if offer:
                    panel.suggest(offer)
                x, y, w, h = result.face_box
                cv2.rectangle(frame, (x, y), (x+w, y+h), (180, 200, 220), 2)

            if state in ("Confused", "Frustrated"):
                status = "Need an explanation? Type a question whenever you want."
            elif state == "Bored":
                status = "Want a different angle? Choose Make it engaging."
            elif state == "Drowsy":
                status = "A short break may help if you feel tired."
            elif state == "Face not visible":
                status = "Face not visible. You can still ask for help manually."
            else:
                status = "Study at your pace. Press H or use the panel to ask for help."
            panel.set_status(status)
            cv2.putText(frame, "Press H for help | Q to quit", (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.imshow("Study Buddy camera", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("h"):
                panel.app.lift()
                panel.topic.focus_set()
            panel.tick()
    finally:
        camera.release()
        cv2.destroyAllWindows()
        if not panel.closed:
            panel.close()


if __name__ == "__main__":
    main()



