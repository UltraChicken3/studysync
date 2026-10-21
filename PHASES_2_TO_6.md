# StudySync implementation handoff: Phases 2–6

This expands the [model-first plan](IMPLEMENTATION_PLAN.md) into buildable work. The
original `study_buddy_resnet_final.keras` checkpoint, six labels, and
`study_buddy_inference.py` preprocessing path remain frozen. Every phase must
work with synthetic frames and a fake tutor before requiring a camera or key.
Do not put frames, face crops, student material, credentials, or session records
in Git. Keep the existing desktop app usable while the web interface is built.

## Current baseline and sequence

- Phase 1 model integration is implemented locally. Its `MODEL_BASELINE.md` and
  synthetic/real-model smoke tests must stay green before each integration.
- Phase 2 step 1 has begun in `study_session.py`: the controller owns one
  camera worker, reuses one classifier, and exposes start/pause/resume/stop.
  Synthetic lifecycle tests cover release and camera failures. API, events,
  reconnect, UI presence, frontend, and hardware checks remain open.
- Build in this order: Phase 2 session core and local web shell; Phase 3
  student-initiated tutor; Phase 4 sustained offer policy; Phase 5 evaluation
  and consented pilot; Phase 6 packaging. Phase 2 can emit tentative classes,
  but automatic offers use the final Phase 4 policy.
- At each small merge point run `.\.venv\Scripts\python.exe -m unittest discover
  -v`, `.\.venv\Scripts\python.exe -m pip check`, the frontend's typecheck and
  build once it exists, and `git diff --check`. Use fake camera/model/tutor
  objects for tests; reserve real camera and Gemini for documented manual checks.
- Do not enable automatic monitoring on page load, browser reload, or machine
  boot. The student starts a session explicitly.

## Phase 2 — local web shell and session lifecycle

**Deliverable:** a web page served by one local Python process, with a single
student-started session that continues while another tab or app has focus.

1. Add a dependency-light `study_session.py` domain layer. Represent idle as
   no session and define `running`, `paused`, `stopped`, and `error` states;
   allow only one active session ID; use a monotonic elapsed clock and a
   camera worker that owns a single
   `VideoCapture`. Inject camera and classifier factories for tests. Load the
   frozen model once per service process, and release the camera on pause,
   stop, worker failure, and service shutdown. A second start must fail cleanly.
2. Define a small event contract: `{session_id, sequence, timestamp, type,
   payload}`. Emit a snapshot after a state change and low-rate inference
   updates containing capture status and tentative class. Keep frames and
   per-frame scores inside Python; send no image bytes through the API.
3. Add a FastAPI service bound only to `127.0.0.1`. Provide `GET /health`,
   `POST /sessions`, `GET /sessions/{id}`, and pause/resume/stop operations.
   Generate an unpredictable token at service startup. Require it on control
   and WebSocket requests, validate loopback Host/Origin, reject oversized
   payloads, and keep CORS closed except for the explicit local development
   origin. `GET` after reload must recover state without starting a worker.
4. Add `WS /sessions/{id}/events` with one current snapshot on connect and
   sequence-numbered subsequent events. Reconnect must not create another
   camera owner. Track active UI connections; after a short documented grace
   period with none, pause capture and release the camera. A hidden but
   connected tab may continue a student-started session.
5. Add a Vite/React/TypeScript shell with subject, goal, duration, Start,
   Pause, Resume, Stop, elapsed time, camera/model status, and a manual-help
   entry point. Show that frames remain local. Use one same-origin production
   build served by FastAPI; development uses a loopback-only proxy. No camera
   permission is requested by the browser because Python owns capture.
6. Provide a synthetic inference stream behind a development-only switch so
   interface work and automated tests need no student image. It must be off in
   the normal packaged launch and visibly labeled when used.
7. Test transition legality, concurrent start, release on every exit path,
   worker errors, browser reconnect, token/Origin rejection, and real-model
   inference with synthetic input. Manually test Windows Chrome/Edge with
   another tab and desktop app foregrounded, unplug/replug, and sleep/resume.

**Gate:** Start uses the saved ResNet50; Pause and Stop release the camera;
reconnect preserves one session; failures remain visible; no frame leaves the
local process. Do not call this gate complete from synthetic tests alone.

## Phase 3 — tutoring and study context

**Deliverable:** a useful guide only after an accepted offer or manual request.

1. Extract the desktop prompt wording into a tutor module shared where useful.
   Preserve distinct Confused, Frustrated, and Bored guide structures: plain
   explanation/example; small steps/next action; analogy/application/challenge.
   An accepted cue is a suggested guide type, never a factual emotion claim.
2. Add a topic and an optional bounded pasted excerpt. Before sending, show
   exactly which text goes to Gemini. The API receives only approved text,
   selected guide type, and a request ID; never a frame, score stream, or key.
   Keep the key in the Python environment and out of browser bundles/logs.
3. Implement one-at-a-time async tutor requests with per-session and per-day
   local budgets, a maximum input/output size, timeout, cancellation behavior,
   and useful messages for missing key, offline service, empty response, and
   quota/HTTP 429. Verify the live model's Free Tier availability and actual
   project limits in AI Studio before turning this on; quotas can change.
4. Add `POST /sessions/{id}/help` for an accepted offer or manual question,
   and `POST /sessions/{id}/feedback` for useful, irrelevant, too easy, too
   hard, incorrect, and wrong-cue responses. Keep only minimal session
   counters by default; transcript persistence requires a separate setting.
5. Test prompts with a fake tutor, request budget boundaries, concurrency,
   duplicate request IDs, key absence, timeout, 429, and secret redaction.
   Manually verify one real request on a free project using non-sensitive text.

**Gate:** each accepted class creates its intended guide; manual help works
with monitoring paused/off; API errors do not break the session or leak a key.

## Phase 4 — model-led offers and interruption control

**Deliverable:** restrained, class-specific offers backed by sustained model
evidence. Keep all thresholds configurable and marked provisional until Phase 5.

1. Add a fixed-duration rolling window over the original six scores, using
   monotonic timestamps rather than frame counts. Ignore camera errors and
   no-face frames; reset a candidate after a documented gap. Aggregate scores
   and apply `classify_probabilities` to the aggregate; record the contributing
   sample count and window duration for debugging without storing images.
2. Map sustained Confused, Frustrated, Bored, and optionally Drowsy to distinct
   offer cards. Focused, Looking Away, Uncertain, and no face never assert an
   emotion or create a study-guide offer. Keep a separate break suggestion for
   Drowsy. Require a hold interval, then one offer per cue episode.
3. Add dismiss, wrong-cue, snooze, alternate guide type, clear interval,
   cooldown, and per-class enable switches. The policy state belongs in the
   backend so a hidden/reloaded tab cannot retrigger an offer. Never submit a
   Gemini request merely because an offer appears.
4. Show a quiet card in the web UI. Optional desktop notifications require
   explicit opt-in and should link back to the card. Avoid topmost windows,
   alarms, and unvalidated attention percentages.
5. Test irregular frame rates, threshold ties, flickering predictions,
   no-face gaps, pause/resume, dismissal, snooze, cooldown, class switches,
   disabled classes, and reconnect. Use a synthetic probability timeline and
   assert offer counts and timing rather than just state strings.

**Gate:** offers require sustained evidence and occur at most once per
episode; interruption controls hold across reconnect; low-confidence or lost
camera produces no emotion claim.

## Phase 5 — baseline evaluation and pilot

**Deliverable:** defensible evidence for the unchanged model and for the full
model → offer → guide chain. This is research work; do not invent metrics.

1. Inventory the local `Student Dataset` outside Git: path, label, hash,
   dimensions, source/subject/video identity where known, and rights/consent
   status. Resolve the known 35 Bored/Drowsy conflicting duplicate groups and
   audit near duplicates. Do not publish images or a manifest that identifies
   students without permission.
2. Recover training details from original artifacts. Document unknowns. Make
   a deterministic evaluation script using the same crop/preprocessing and
   class-order assumption. Keep the original checkpoint immutable.
3. Make subject- or video-separated train/validation/test splits when source
   identity exists. If it does not, clearly label the existing folder split
   as limited and obtain a stronger dataset before reporting generalization.
   Tune thresholds only on validation; lock the test split until final review.
4. Report per-class precision/recall/F1, macro F1, confusion matrix,
   calibration/reliability, no-face rate, CPU latency, and coverage by lighting,
   pose, and device. Include uncertainty intervals and label uncertainty.
5. With consenting participants, measure unwanted offers/hour, missed useful
   offers, corrections, acceptance, guide usefulness/correctness, interruptions,
   and quota use. Compare with manual-help and random-timing controls. Define
   guide-usefulness target before the pilot. The provisional unwanted-offer
   gate is at most one per study hour; do not claim success from a tiny sample.
6. Review each class separately. Disable automatic offers for weak classes
   while keeping their model outputs for research. Propose any retraining,
   detector replacement, or new model as a separate reviewed experiment.

**Gate:** publish aggregate findings and limits, not student media; decide
explicitly which classes can safely trigger offers. Missing provenance or an
independent test split blocks strong accuracy claims.

## Phase 6 — Windows packaging and release

**Deliverable:** a local installation that starts and exits cleanly without a
terminal, including the original Git LFS model and built web assets.

1. Choose and document a Windows launcher/packager after measuring bundled
   TensorFlow size and startup time. Pin the Python and Node toolchains and
   produce repeatable local build commands. Verify the packaged model's SHA-256
   against `MODEL_BASELINE.md` and ensure Git LFS has supplied bytes, not a
   pointer file.
2. On launch, bind only to loopback, create a new token, start one service,
   open the local browser, and show camera/model/key readiness without exposing
   secrets. Do not auto-start monitoring. Provide an exit control that stops
   worker and tutor tasks and releases the camera before terminating.
3. Store preferences and optional aggregate summaries in a user-local
   directory with a schema version and migration tests. Keep frames out of
   storage. Document how to clear data and change camera or API key.
4. Run the full automated suite, typecheck/build, dependency checks, and a
   source/bundle/history secret scan. Test on a fresh Windows user profile:
   start/pause/resume/stop, hidden tab, another active app, no camera, camera
   unplugged, no key, 429/offline, sleep/resume, and normal/forced exit.
5. Prepare a release checklist with measured startup, inference latency, and
   package size; installation and uninstall steps; accessibility pass; and
   clear disclosures about the limits of facial inference and Gemini text use.

**Gate:** a new user can control a session without a terminal; exit releases
the camera; unavailable camera/API still permits manual study setup and
appropriate help recovery. Publish only after Phase 5's model-offer gate is
reviewed.
