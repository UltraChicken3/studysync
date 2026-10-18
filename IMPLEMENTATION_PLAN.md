# StudySync Model-First Implementation Plan

# Decision and project goal

StudySync is a deep learning project centered on its original six-class ResNet50 student-state classifier. The saved model, class order, 224 × 224 face crop, and ResNet50 preprocessing remain the baseline. A local Python companion runs the model during a student-started study session, including while another browser tab or desktop app has focus. A web interface displays the model-led help offer and the resulting study guide. Monitoring can be paused or stopped at any time.

The intended causal path is **camera frame → face crop → original ResNet50 → time-windowed class evidence → class-specific help offer → student accepts → study guide grounded in the current topic**. The classifier decides when and what kind of guide to offer; Gemini writes the guide only after the student accepts or asks manually. Facial appearance cannot establish a student's internal state with certainty, so offers are phrased as questions and can be dismissed. The model is the central research artifact, and its reliability is a release criterion rather than an assumed fact.

This plan supersedes the earlier direction that treated the ResNet50 as a secondary research-only cue. Optional quiz answers may help the tutor adapt a response, but they are not the primary trigger for study guides. Replacing the model or changing its six labels requires a separate decision after baseline results are available.

## Release scope and student journey

1. The student opens the local web app, enters a subject and current goal, and optionally pastes notes. The app explains that frames stay local and that selected text is sent to Gemini only when help is requested. PDF extraction can follow after the baseline works.
2. The student chooses a session length and starts monitoring. The interface shows camera status, a small optional preview, elapsed time, and Start, Pause, Stop, and Snooze controls.
3. The local companion samples the webcam throughout the active session, including while another tab or desktop app has focus. It runs the original model on detected face crops and keeps a rolling window of predictions rather than recording video.
4. Sustained, sufficiently decisive Bored, Confused, or Frustrated output creates a class-specific help offer. The student can accept, dismiss, snooze, or open help manually. Drowsy may offer a break; Focused, Looking Away, Uncertain, and face-not-visible normally produce no study guide request.
5. When help is accepted, the tutor uses the subject and permitted study material to provide a study guide shaped by the offer: simpler explanation, smaller steps, or a more engaging analogy and challenge. The student can ask a follow-up or rate the answer.
6. At the end, the app shows session duration, breaks, help requests, and the student's own feedback. It does not display an unvalidated percentage labeled “attention.”

Initial scope covers model-led self-study assistance on Windows. Exam enforcement, identity verification, screen recording, remote supervision, and disciplinary decisions are outside the first release. Screen capture would require a separate design and permission flow.

## Architecture

| Component | Responsibility |
| --- | --- |
| React and TypeScript frontend | Session setup, study topic, live model status, class-specific help cards, study guide, settings, and review. Vite builds the assets. |
| Local FastAPI service | Serves the built frontend on 127.0.0.1, owns the session state, exposes a small HTTP API, and sends cue events over WebSocket. |
| Camera and model worker | Opens the webcam, runs the existing Haar face detector and ResNet50 preprocessing and inference locally, and releases the camera on pause, stop, or shutdown. |
| Offer policy | Combines time-windowed model probabilities, uncertainty gates, cooldowns, and snoozes; emits at most one class-specific offer per episode. |
| Tutor service | Keeps the Gemini API key on the Python side, builds a short class-specific guide from the accepted offer and selected study context, handles quota failures and timeouts, and returns the guide. |
| Local storage | Stores settings and optional session summaries in SQLite. Raw frames and continuous video are not persisted. |

The production frontend and API should share the same localhost origin. The service binds only to 127.0.0.1, validates the request Origin and Host, and checks an ephemeral session token for control and WebSocket requests. Development can use Vite's development server with a restricted local proxy. The frontend never receives the API key.

Browser-only camera processing is not the selected architecture because hidden tabs can throttle rendering and timers. The Python worker owns the capture loop, so tab switching does not stop observation. The web page receives low-volume state events through a WebSocket; it does not need a video stream to remain open in the background. If the page disconnects or the service loses its active UI, the service stops monitoring after a short grace period and releases the camera. Desktop notifications are opt-in and used when the page is hidden; the in-page card is always available when the student returns. The service must never start monitoring automatically on machine boot or browser reload.

Proposed API contracts:

| Operation | Purpose |
| --- | --- |
| POST /sessions | Start one session with goal, duration, and monitoring settings. |
| POST /sessions/{id}/pause, /resume, /stop | Control the session and camera lifecycle. |
| GET /sessions/{id} | Retrieve current status after a browser reload. |
| WS /sessions/{id}/events | Deliver status, cue offers, errors, and tutor progress. |
| POST /sessions/{id}/help | Submit an accepted or manual question with selected context. |
| POST /sessions/{id}/feedback | Record whether an offer or answer was useful. |
| GET /health | Report local service, camera, model, and key readiness without exposing secrets. |

The backend should have one session controller and one camera owner. Each event carries a session ID, timestamp, event type, and minimal payload. The browser can reconnect and fetch the current session state without starting a second camera loop.

## Model-led offer policy

Retain the current model output order: Bored, Confused, Drowsy, Focused, Frustrated, Looking Away. `Uncertain` is an inference-policy result, not a seventh model class. Face-not-visible and camera failure are capture conditions, not emotion classes. The web backend should reuse the baseline face-crop and `preprocess_input` path so the frontend migration does not silently change model behavior.

| Sustained model result | Offer and accepted guide |
| --- | --- |
| Confused | “Would a simpler explanation help?” → plain-language explanation and one worked example. |
| Frustrated | “Want to break this into smaller steps?” → first step, hint, and a manageable next action. |
| Bored | “Want a different angle?” → analogy, application, or short challenge grounded in the topic. |
| Drowsy | Optional break offer; no claim that the student is medically fatigued. |
| Focused | No interruption. |
| Looking Away | No study guide trigger in the baseline; looking at notes or another display may be normal. |
| Uncertain or no face | No emotion claim or study guide trigger. |

The baseline policy starts from the existing decisive-score gate and sustained-cue tracker. Before shipping automatic offers, tune the score gate, margin, hold time, clear time, and cooldown against a local validation set and pilot feedback. Aggregate probabilities over a time window, then require a stable class rather than reacting to individual frames. The existing 0.65 score, 0.15 margin, and two-second hold are prototype values, not calibrated probabilities or proven thresholds. Use monotonic time so camera frame rate does not change the policy. At most one offer is shown per episode; dismissal and snooze suppress repeats.

The student may override a wrong offer (“That does not fit”), accept a different guide type, or ask manually. Feedback records the offered class, whether it was accepted, and whether the guide helped, without saving a face image. This gives a measurable link from the classifier to the educational result. A facial prediction remains a tentative cue: the system must not tell a student that it knows their emotion or punish an apparent lapse.

## Tutor behavior and interface

The main page needs a goal and timer area, a clearly visible monitoring and model status, the current topic, a small help entry point, and a study guide drawer. Offers appear as dismissible cards and optional system notifications, without an alarm or a forced topmost dialog. The interface should also permit manual help when the camera is off or unavailable.

After an accepted offer, the tutor receives the study topic, selected guide type, and a bounded excerpt of material approved by the student. The prompt should request the corresponding explanation, steps, or engaging connection without declaring the detected emotion as fact. It should return a concise guide and one optional follow-up question. The student can mark the guide useful, irrelevant, too easy, too hard, or incorrect. The model prediction selects the default guide type, but the student can change it. Errors, no API key, rate limiting, and offline mode need visible recovery actions.

Start with a typed topic and pasted text. The app should show which material will be included in a request before sending it. Add PDF extraction and retrieval over larger documents after the baseline guide is reliable. Default session summaries should store counts and feedback rather than full conversation text; saving transcripts should require a separate setting. No frame or face crop is sent to Gemini.

## Data and model work

1. Preserve the original `study_buddy_resnet_final.keras` weights as the baseline artifact. Record the six-class output order, face detector, crop behavior, RGB conversion, ResNet50 preprocessing, and expected input/output shapes. Add a small structural smoke test that does not require publishing a student's face.
2. Keep the original `Student Dataset` locally. Create a local manifest with path, label, SHA-256, image dimensions, and source or subject identity where known. Resolve the 35 exact duplicate groups carrying both Bored and Drowsy labels and audit near duplicates. Do not publish face images or the full manifest until provenance, consent, and redistribution rights are established.
3. Recover the original training recipe where possible, or document what cannot be recovered. Reproduce a fixed baseline with controlled preprocessing, class mapping, seed, augmentation, and checkpoints. The present folder validation subset is not an independent test set.
4. Use subject- or video-separated train, validation, and test partitions. If subject provenance is missing, obtain it or label the evaluation as limited. DAiSEE can support a later video experiment, but its clip-level labels do not directly replace the current six-class head.
5. Report per-class precision and recall, macro F1, confusion matrix, calibration assessment, CPU inference latency, and performance under different lighting and camera setups. Also report unwanted offers per study hour, missed useful offers, acceptance rate, student corrections, and guide helpfulness. Evaluate the full model → offer → guide chain, not just frame classification accuracy.
6. Decide whether automatic class-specific offers are ready from those results. If a class performs poorly, keep its output visible as a research result but disable its automatic offer until improved. This preserves the model as the centerpiece without claiming unsupported accuracy.

**Baseline freeze:** retain the ResNet50 architecture, saved weights, six labels, and preprocessing for the first integrated web release. Training changes, a new backbone, a new dataset, or new labels are separate follow-on experiments requiring review. Threshold calibration and temporal aggregation around the model are allowed baseline integration work because they do not modify its weights.

**Free API control:** use `gemini-3.5-flash-lite` on a verified Gemini Free Tier project without billing. Send no request on every frame or on offer display; call the API only after an accepted offer or manual question. Limit excerpt and response size, allow one request at a time, keep a local per-session and per-day request budget below the project's live quota, and handle HTTP 429 by showing a retry-later message. Google publishes free-tier pricing, but the active RPM, TPM, and RPD limits must be checked in AI Studio for the user's project. Free-tier content may be used to improve Google's products, so the UI must disclose that selected study text is sent to Gemini.

**Public repository rule:** the public Git history contains source, docs, and the model through Git LFS. `Student Dataset`, demo recordings, slide deck, API keys, local transcripts, and pilot recordings stay out of every published commit. Document where a permitted user places a local dataset for evaluation. Rotate the previously exposed Gemini key before using the API.

## Build sequence and completion gates

### Phase 1: Repository and reproducible baseline

Create separate frontend and backend packages, lock dependency versions, document one-command local startup, and add a model specification and baseline evaluation script. Keep the dataset local and document the limits of reproducibility without redistributing it. Completion gate: a fresh install can load the original model, verify its six outputs and preprocessing path, and run tests without absolute paths or a real face image in Git.

### Phase 2: Local web shell and session lifecycle

Build the React pages and FastAPI session controller. Serve the production build locally, implement WebSocket status events, browser reconnect, and Start/Pause/Stop. Integrate the existing Haar face detector and ResNet50 worker; add a synthetic probability stream for development. Completion gate: the same saved model drives local inference during tab and app switches, while Stop and service shutdown release the camera.

### Phase 3: Tutoring and study context

Implement class-specific accepted-offer guides, manual questions, bounded topic context, and feedback. Keep Gemini calls asynchronous, budgeted, and student-initiated. Completion gate: a Confused, Frustrated, or Bored event can produce the corresponding offer and accepted guide, manual help still works with monitoring disabled, and no API key appears in browser assets or logs.

### Phase 4: Model offer policy and interruption control

Add probability aggregation, class-specific offer mapping, a cue state machine, help cards, snooze, and optional desktop notifications. Use synthetic probability streams and consented local sessions to test timing and repetition. Completion gate: offers depend on sustained configured model evidence; dismissal and snooze hold; camera loss or low confidence produces no emotion claim.

### Phase 5: Baseline model evaluation and pilot

Clean the labels, evaluate the retained ResNet50 on the strongest defensible split, and test the complete offer-to-guide path with consenting students in realistic lighting and study setups. Tune policy thresholds from feedback, not from the test set. A provisional product gate is no more than one unwanted model-led offer per study hour in the pilot; set a guide-usefulness target before collecting pilot results. Report uncertainty and subgroup limitations. Publish aggregate measurements alongside the model, not face images or a single accuracy number.

### Phase 6: Packaging and release

Package the local companion for Windows, include the web build and original model, add first-run checks for camera, Free Tier API configuration, and Git LFS. Completion gate: a new user can start, pause, resume, and end a session without a terminal; capture stops on exit; and camera or API failure leaves manual help available.

## Verification matrix

- Unit tests: class order and mapping, score gating, time-window aggregation, cooldown, snooze, session transitions, free-request budget, and class-specific prompt construction.
- Integration tests: saved ResNet50 loading and preprocessing parity, FastAPI routes, WebSocket reconnect, one-camera ownership, accepted-offer guide creation, Gemini failures and 429 responses, and storage migrations.
- Manual hardware tests: Chrome and Edge on Windows, hidden tab, another desktop app in focus, sleep/resume, camera unplugged, poor lighting, glasses, and multiple faces.
- Privacy checks: inspect network traffic to confirm no webcam frames leave the computer; verify that Stop releases the camera; scan the entire history to be pushed for keys, faces, and recordings; verify that keys are absent from frontend bundles and logs.
- User study: record student feedback for model offers and guides, including wrong-cue reports, and measure unwanted offers per hour and guide usefulness. Separate results by environment and camera quality before declaring a class reliable.

## Design review: strengths, limitations, and decisions

The concept has a clear research story: the original classifier drives a measurable intervention, and the study guide is the downstream action. A successful demo must show the complete causal chain with a real model output, not a hard-coded emotion selector. The guide should have a clearly different structure for confusion, frustration, and boredom; otherwise the classifier adds little value.

The weakest assumption is that a single face crop predicts study-related emotion reliably across students and contexts. The local dataset has conflicting labels, the existing validation split is not independent, and a face can look similar while someone is thinking, reading, tired, or disengaged. The baseline therefore needs a model uncertainty state, sustained evidence, class-level enable/disable switches, and user correction. These controls preserve the model's role while reducing costly false interruptions. A second risk is that emotion does not identify the study topic; the student must provide a topic or approved material for the guide. A third risk is quota or connectivity failure; the model and session controls must continue working without Gemini.

The baseline should answer three questions with separate measures: **Does the model distinguish its six labels on defensible data?** **Do its class-specific offers arrive at useful moments?** **Does accepting an offer produce a useful guide?** High image accuracy alone cannot answer the second or third question. A policy-only ablation using manual help, and an ablation with the same guide shown at random times, can show whether model-triggered timing adds value without replacing the model in the product.

## Optimizations after the baseline

These are candidates, not approved changes to the original model. Select them from measured baseline failures, then compare each with the frozen baseline on the same split and hardware.

| Candidate | Use it when | Evaluation and tradeoff |
| --- | --- | --- |
| Calibrate scores and thresholds per class | Some classes create too many wrong offers despite apparently decisive softmax scores. | Compare reliability and unwanted offers per hour; reserve a separate test set. This changes the decision layer, not weights. |
| Stronger temporal modeling | Predictions flicker or very short expressions dominate decisions. | First compare rolling probability summaries and hysteresis; only then test a sequence model over frame embeddings. Measure latency and missed useful offers. |
| Subject-aware retraining of the same ResNet50 | A clean, provenance-aware dataset is available and current class recall is inadequate. | Preserve the original checkpoint and compare a retrained checkpoint on a held-out subject split. Avoid training on adjacent frames from test subjects. |
| Better face detection and quality gating | Haar detection fails frequently for pose, lighting, or glasses. | Compare detection coverage and classifier accuracy on the same consented clips; check that crop differences do not silently change labels. MediaPipe is one candidate. |
| Class redesign or multi-label video data | Six mutually exclusive still-image labels cannot represent mixed or changing states. | Treat this as a new research version. DAiSEE-style clip labels need a new head and mapping; do not present them as a drop-in replacement. |
| Smaller or quantized model | CPU latency, installation size, or battery use blocks adoption. | Compare MobileNetV3, distillation, or quantization with ResNet50 on class metrics, offer utility, latency, and package size. Replacement requires approval. |
| Personalization from corrections | Aggregate policy is poor for a consenting student's camera setup. | Start with adjustable thresholds and snooze preferences. Train personalized weights only with explicit consent and enough local data; check overfitting. |
| Better guide grounding | Guides are generic or factually weak despite correct offers. | Add local PDF parsing and bounded retrieval, then evaluate answer correctness, useful-guide rate, latency, and free-tier token use. |
| Fully local tutor | Free API quotas or data-use terms are unacceptable for real student material. | Compare a small local text model or curated templates for quality and hardware cost. Keep the emotion classifier unchanged. |

Do not optimize for raw classification accuracy alone. Prioritize fewer unwanted interruptions and more useful accepted guides while keeping the student in control.

## References

- [Browser camera permission and secure contexts](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia)
- [Background tab behavior](https://developer.mozilla.org/en-US/docs/Web/API/Page_Visibility_API)
- [Challenges of inferring emotion from facial movement](https://www.psychologicalscience.org/journals/pspi/1529100619832930/)
- [MediaPipe Face Landmarker for Python live streams](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python)
- [FastAPI WebSockets](https://fastapi.tiangolo.com/advanced/websockets/)
- [Keras MobileNetV3 models and preprocessing](https://keras.io/api/applications/mobilenet/mobilenet_models/)
- [DAiSEE dataset paper](https://arxiv.org/abs/1609.01885)
- [Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing)
- [Gemini API rate limits](https://ai.google.dev/gemini-api/docs/rate-limits)
