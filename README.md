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

## Run locally

Use native Windows CPython 3.12 (64-bit), not MSYS Python. The tested package versions are pinned in `requirements.txt`. If `.venv` is already installed, run `.\.venv\Scripts\python.exe nus_proj.py` from the project folder. For a fresh checkout, use a native Windows Python executable in PowerShell:

```powershell
& 'C:\Path\To\Python312\python.exe' -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:GEMINI_API_KEY = '<new key>'
.\.venv\Scripts\python.exe nus_proj.py
```

Press **H** in the camera window to focus the help panel, or **Q** to quit. The panel also accepts questions without a camera cue. Without `GEMINI_API_KEY`, the webcam and panel work, but requesting generated text shows a setup message. `STUDY_BUDDY_MODEL` can point to another compatible six-output Keras model; `GEMINI_MODEL` selects the Gemini text model; `STUDY_BUDDY_CAMERA` selects the camera index. A different image backbone also requires its matching preprocessing and class mapping in the code.

## Model-first web roadmap

The [model-first implementation plan](IMPLEMENTATION_PLAN.md) retains the original ResNet50 and makes its predictions the trigger for a class-specific study guide offer. It covers cross-app monitoring, a local web interface, data and model evaluation, free-tier API controls, release phases, and later optimization candidates. The web app has not been built yet.

## Model improvement plan

1. **Freeze the original baseline.** Preserve the saved ResNet50 weights, six-class order, face crop, and preprocessing while connecting its sustained predictions to study guide offers.
2. **Fix the local data.** Resolve conflicting duplicates and audit near duplicates, subjects, clips, lighting, and labels. Use subject- or video-separated train/validation/test splits. Keep face images out of the public repository.
3. **Measure the complete chain.** Report per-class metrics, calibration assessment, CPU latency, false offers per study hour, student corrections, and guide usefulness. Compare model-led offers with manual help to determine whether the model adds value.
4. **Optimize after the baseline.** Consider temporal modeling, calibration, retraining the same ResNet50, or a smaller model only after the original pipeline has measured results and a proposed change is reviewed.

The included `test_study_buddy_logic.py` checks the confidence gate and one-off suggestion behavior. A live webcam, TensorFlow inference, and Gemini response could not be tested in this workspace because those packages and a configured camera/API key are not available.

## Demo and deck checks

The two supplied demo videos show the original camera overlay, the small blocking “Boredom Buster” input dialog, and the separate fixed-size answer window. The recorded source-code screen also shows the old API key. Rotate the key and redact the recording before sharing it. The deck's evaluation slides include a “Wav2Vec 67.62%” label, which does not match the image classifier used by this app; do not cite it as this model's accuracy.


## Git repository contents

The public repository tracks the application, tests, documentation, and the six-class `study_buddy_resnet_final.keras` model through Git LFS. The `Student Dataset` folder remains local for evaluation and is ignored along with DAiSEE, RAF-DB, the duplicate ZIP archive, unused RAF-DB model files, demo videos, and presentation. To run the evaluation scripts, place a permitted copy of `Student Dataset` at the project root with its six original class folders. Do not upload face images, student records, recordings, or credentials without established rights and consent. The demo recordings show an old API key; rotate it before using Gemini.

Install Git LFS before cloning or pushing. If a clone shows only a small text pointer where the Keras model should be, run `git lfs pull` inside the clone. The API key belongs in the local environment, and packages are installed from `requirements.txt`; neither belongs in a commit.

The GitHub destination is `https://github.com/UltraChicken3/studysync`. Its public history is prepared without the local face-image dataset.
