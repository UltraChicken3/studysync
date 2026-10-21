# StudySync: model-led study assistance

This is a webcam study assistant centered on the original six-class ResNet50 image classifier. Its Bored, Confused, and Frustrated outputs can lead to different help offers; the student chooses whether to request generated help. The current desktop prototype sends **only the student's typed question** to Gemini when the student presses **Get help**. Camera frames stay local. Facial predictions are tentative cues, not proof of a person's internal state.

## What was in the original project

- `nus_proj.py` captured webcam frames, detected faces with a Haar cascade, resized the selected face to 224 × 224, used ResNet50 preprocessing, and ran a six-class saved Keras model.
- The six classes were Bored, Confused, Drowsy, Focused, Frustrated, and Looking Away. The code assumes the six output order matches the alphabetic folder order in `Student Dataset`.
- A 30-frame calibration average was subtracted from later softmax outputs. Five-frame averaging, a 20-frame state history, and a 60-second cooldown attempted to suppress noisy prompts.
- Confused/Frustrated and Bored invoked distinct Gemini prompts. Drowsy/Looking Away caused an alert.
- `model_eval.py` and `roc_curve_nus.py` evaluated the 20% `ImageDataGenerator` validation subset. The project deck compares MobileNet's speed with ResNet's capacity and describes consent-based help.

These are understandable prototype choices: transfer learning is sensible for 2,120 labeled images; temporal filtering tries to reduce spurious single-frame predictions; and two response modes make assistance relevant to the student's immediate need. The saved model has a ResNet50 backbone, global average pooling, a 256-unit ReLU layer, 50% dropout, and a six-way softmax head.

## Findings, in priority order

1. **Credential exposure.** The old app and debug script contained a Gemini API key. It has been removed from the Python files. Revoke/rotate that key in Google AI Studio before using Gemini again, since a key committed or shared earlier remains exposed.
2. **Conflicting labels.** The six-class folder has 2,120 JPEGs. SHA-256 comparison found **35 exact duplicate image groups** labeled both Bored and Drowsy. Clean these conflicts before retraining. Also check near duplicates and images extracted from the same person or video.
3. **Validation does not establish real-world accuracy.** The evaluation scripts used a 20% folder validation subset and called the result “final test accuracy.” That subset is not an independent test set. No training script, split manifest, or saved prediction report is present here, so the model's claimed performance and class order cannot be fully reproduced.
4. **The target is under-specified.** A still face may show expression, but confusion, boredom, focus, and looking away depend on task context and time. The deck itself notes that available datasets poorly represent real study behaviors. The RAF-DB artifacts output seven emotion classes and cannot directly replace this six-class app model. DAiSEE includes clip-level Boredom, Engagement, Confusion, and Frustration labels and separate Train, Validation, and Test folders, but this repo has no pipeline that maps those clips to the app's six classes.
5. **The original decision rule distorted scores.** Subtracting a 30-frame “baseline” from softmax probabilities does not calibrate model confidence. It can make any normal change exceed a 0.01 threshold and assumes the first 30 frames show the student's normal focused state. Missing face was treated as looking away, which also happens during reading, poor lighting, or camera failure. Frame counts made prompt timing depend on hardware speed.
6. **The old interaction interrupted studying.** A blocking input dialog and a second topmost answer window paused the video loop; network errors were only printed to the console. The wording stated “You look confused/bored,” and an unanswered or cancelled dialog could reappear once the trigger recurred. Broad `except: pass` hid inference failures.
7. **Portability and maintenance.** Absolute paths pointed outside this folder. There was no dependency list. The old `google-generativeai` client is a legacy SDK.

## Changes made

- Replaced the blocking dialogs with one persistent CustomTkinter help panel. The student can switch between **Explain simply** and **Make it engaging**, type a question, dismiss a suggestion, or ask manually at any time. The generated answer and errors appear in the same panel.
- Gemini requests run in a worker thread so the camera and panel continue responding. The app uses the supported `google-genai` SDK and a configurable model ID; the default is `gemini-3.5-flash-lite`.
- Suggestions use a decisive model output plus a two-second continuous cue, then require the cue to clear and a cooldown before another offer. The 0.65 probability and 0.15 margin gates are starting values, **not validated operating thresholds**. The UI says “possible” and never submits a request based on a camera prediction alone.
- Removed alarm sounds and the assumption that an absent face means looking away. Possible fatigue is a quiet status message. The largest detected face drives the cue, avoiding mixed predictions from multiple people.
- Made model and data paths relative to the project by default, with environment overrides. The evaluation scripts now name their metric **validation accuracy** and verify folder class order. The key is read from `GEMINI_API_KEY` only.
- Moved the original face-crop and ResNet50 inference path into `study_buddy_inference.py`. See [MODEL_BASELINE.md](MODEL_BASELINE.md) for the saved model's hash, input/output contract, and class-order assumption.

## Run locally

Use native Windows CPython 3.12 (64-bit), not MSYS Python. The tested package versions are pinned in `requirements.txt`. If `.venv` is already installed, run `.\.venv\Scripts\python.exe nus_proj.py` from the project folder. For a fresh checkout, use a native Windows Python executable in PowerShell:

OpenCV is pinned to the 4.x line because the original Haar face detector uses `cv2.CascadeClassifier`; the standard OpenCV 5 wheel no longer provides it. If monitoring reports a missing `CascadeClassifier`, run `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` and restart the web service.

```powershell
& 'C:\Path\To\Python312\python.exe' -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:GEMINI_API_KEY = '<new key>'
.\.venv\Scripts\python.exe nus_proj.py
```

Press **H** in the camera window to focus the help panel, or **Q** to quit. The panel also accepts questions without a camera cue. Without `GEMINI_API_KEY`, the webcam and panel work, but requesting generated text shows a setup message. `STUDY_BUDDY_MODEL` can point to another compatible six-output Keras model; `GEMINI_MODEL` selects the Gemini text model; `STUDY_BUDDY_CAMERA` selects the camera index. A different image backbone also requires its matching preprocessing and class mapping in the code.

## Model-first web roadmap

The [model-first implementation plan](IMPLEMENTATION_PLAN.md) retains the original ResNet50 and makes its predictions the trigger for a class-specific study guide offer. The [Phase 2–6 handoff](PHASES_2_TO_6.md) breaks the remaining work into implementation and verification steps. The local web shell has a session controller, loopback API, WebSocket status stream, React interface, manual text guides, and provisional model-led offers. The desktop app still supports manual Gemini questions.

### Run the local web shell

Build the static interface once and start the loopback service from the project root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
npm --prefix frontend ci
npm --prefix frontend run build
.\.venv\Scripts\python.exe run_web.py
```

Open `http://127.0.0.1:8765`. Enter a subject and goal, then choose **Timer** for a set session length or **Stopwatch** to count up until you stop. Press **Start monitoring** to use the webcam and unchanged saved model. The browser normally receives only session and tentative class status. **Show camera preview** is an optional local view of the same capture already used by the model; hiding it closes the preview stream. Its small JPEG frames travel only over the loopback connection to this browser, are not stored, and are never included in Gemini requests. Uncheck camera cues to study without opening a webcam; the chosen clock still runs and Pause stops it. Pause and Stop release the camera during monitoring. Closing every connected page pauses the session after 15 seconds. Nothing starts monitoring on page load.

### Enable manual web guides

The web guide uses only the pinned text model `gemini-3.5-flash-lite`. Google currently lists its text input and output as Free Tier eligible, but [active rate limits vary by project](https://ai.google.dev/gemini-api/docs/rate-limits). In [Google AI Studio](https://aistudio.google.com/), check that your project and key are on the Free Tier and inspect the active limits for this model. Rotate any old key from the original prototype first. Then set these variables locally **before** starting `run_web.py`:

```powershell
$env:GEMINI_API_KEY = '<new free-project key>'
$env:STUDYSYNC_GEMINI_FREE_TIER_CONFIRMED = '1'
.\.venv\Scripts\python.exe run_web.py
```

The student previews the complete prompt and presses **Send this text and get guide**. The guide renders Markdown headings, lists, and emphasis. If it asks a question, the student types an answer in **Respond to the guide**, previews the follow-up prompt, and presses **Send this reply** for feedback. They can also ask one follow-up question there. The follow-up sends their typed reply and up to 2,400 characters from the end of the guide; it uses another request from the same budget. The first guide request includes only the typed question and optional pasted excerpt; frames, face scores, the key, and the session goal are not sent. The excerpt is limited to 1,600 characters, the question to 500, and the follow-up reply to 500. The local service allows one request at a time, at most two attempts per session, five in a rolling 24 hours, and one per minute, with a 20-second timeout and 1,024 output-token cap. These caps are process-local and reset when the service restarts; they are safeguards, not a substitute for AI Studio's live project quota. Gemini's own Free Tier quota applies to the entire project, including other clients. If a project has billing enabled, verify its tier before setting the confirmation flag. Free Tier submitted content may be used by Google to improve its products, so paste only text you are comfortable sharing. See [Google's pricing](https://ai.google.dev/gemini-api/docs/pricing).

Without the key or confirmation, sessions, camera cues, prompt preview, and feedback remain local and usable. A real request is intentionally not part of automated tests. No automatic Gemini request occurs from a face prediction.
During monitoring, the original six scores are averaged over a rolling three-second window. Sustained Confused, Frustrated, or Bored evidence creates one optional offer for that episode. The offer appears as a fixed notice on the page so it remains visible while scrolling. The student can also enable desktop alerts for offers while the page is hidden; browser notification permission is requested only when they click **Enable desktop alerts**. Drowsy break offers are optional and off by default; Focused, Looking Away, Uncertain, and no face do not create guide offers. You can accept, dismiss, mark the cue wrong, snooze for five minutes, or disable an offer class. Accepting preselects a guide type and topic but still requires prompt preview and a second explicit send action. These thresholds are provisional and need calibration and a consenting pilot before claims about reliability.

For UI work without a camera, use `.\.venv\Scripts\python.exe run_web.py --demo`. The page clearly marks synthetic cues and neither opens a camera nor runs the saved model. The normal command always uses the original checkpoint. To run Vite separately, set `STUDYSYNC_DEV_ORIGIN` to `http://127.0.0.1:5173`, start `run_web.py`, then use `npm --prefix frontend run dev`; Vite is bound to loopback and proxies to port 8765.

## Model improvement plan

1. **Freeze the original baseline.** Preserve the saved ResNet50 weights, six-class order, face crop, and preprocessing while connecting its sustained predictions to study guide offers.
2. **Fix the local data.** Resolve conflicting duplicates and audit near duplicates, subjects, clips, lighting, and labels. Use subject- or video-separated train/validation/test splits. Keep face images out of the public repository.
3. **Measure the complete chain.** Report per-class metrics, calibration assessment, CPU latency, false offers per study hour, student corrections, and guide usefulness. Compare model-led offers with manual help to determine whether the model adds value.
4. **Optimize after the baseline.** Consider temporal modeling, calibration, retraining the same ResNet50, or a smaller model only after the original pipeline has measured results and a proposed change is reviewed.

The logic and inference tests check cue behavior and the original model path with synthetic frames. The saved Keras model loads and produces six finite outputs in the configured environment. A live webcam session and Gemini response still require hardware access and a configured API key.

## Demo and deck checks

The two supplied demo videos show the original camera overlay, the small blocking “Boredom Buster” input dialog, and the separate fixed-size answer window. The recorded source-code screen also shows the old API key. Rotate the key and redact the recording before sharing it. The deck's evaluation slides include a “Wav2Vec 67.62%” label, which does not match the image classifier used by this app; do not cite it as this model's accuracy.


## Git repository contents

The public repository tracks the application, tests, documentation, and the six-class `study_buddy_resnet_final.keras` model through Git LFS. The `Student Dataset` folder remains local for evaluation and is ignored along with DAiSEE, RAF-DB, the duplicate ZIP archive, unused RAF-DB model files, demo videos, and presentation. To run the evaluation scripts, place a permitted copy of `Student Dataset` at the project root with its six original class folders. Do not upload face images, student records, recordings, or credentials without established rights and consent. The demo recordings show an old API key; rotate it before using Gemini.

Install Git LFS before cloning or pushing. If a clone shows only a small text pointer where the Keras model should be, run `git lfs pull` inside the clone. The API key belongs in the local environment, and packages are installed from `requirements.txt`; neither belongs in a commit.

The GitHub destination is `https://github.com/UltraChicken3/studysync`. Its public history is prepared without the local face-image dataset.
