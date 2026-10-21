# Saved model baseline

The application uses the unchanged `study_buddy_resnet_final.keras` checkpoint through Git LFS. This document records its inference contract, not a measured accuracy claim.

| Property | Recorded value |
| --- | --- |
| SHA-256 of the 216,958,783-byte model file | `5152239c776ad4cb2f00de26b3a5d16c7f104dc0659d0b188299e0351da75e3f` |
| Input shape | `(None, 224, 224, 3)` |
| Output shape | `(None, 6)` |
| Output class order used by the application | `Bored`, `Confused`, `Drowsy`, `Focused`, `Frustrated`, `Looking Away` |

## Inference path preserved from the desktop prototype

1. OpenCV reads a BGR frame; the desktop app flips it horizontally before inference.
2. A Haar cascade detects faces on a grayscale copy. The largest detected face is selected. No face produces a separate `no_face` result; a failed frame produces `camera_error`.
3. Crop that face from the BGR frame, resize it to 224 × 224, convert BGR to RGB, cast to `float32`, and add the batch dimension.
4. Apply `tensorflow.keras.applications.resnet50.preprocess_input` and run the saved model once for the batch.
5. Pass the six scores to `study_buddy_logic.classify_probabilities`. Its 0.65 top-score and 0.15 top-two margin gates may yield `Uncertain`; that is not a seventh model output. The separate `CueTracker` applies the existing time and cooldown rules.

`study_buddy_inference.py` owns steps 2–5 so a later interface can call the same path. It loads the model and cascade once per classifier instance. Camera frames and face crops are not written to disk or sent to Gemini by this module.

## Known limits and assumptions

- The six-name output order is an **assumption inherited from the original alphabetical `Student Dataset` folder order**. The checkpoint shape confirms six outputs but does not independently prove which name belongs to each output index.
- The original training recipe, subject identities, and independent held-out test results are unavailable in this repository. The existing `model_eval.py` reports a folder validation subset, not a final test result.
- The local dataset contains 35 exact duplicate groups with conflicting Bored and Drowsy labels. It remains outside public Git history pending provenance and redistribution review.
- The score and margin gates are prototype decision settings, not calibrated probabilities. Real webcam behavior, class quality, and prompt usefulness still require consented hardware and pilot testing.

Run the synthetic inference checks without a camera, API key, or student image:

```powershell
.\.venv\Scripts\python.exe -m unittest test_study_buddy_inference -v
```
