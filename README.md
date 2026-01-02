# Surgical Phase Recognition with Sequential Visual Models

## 1. Problem Statement

This project addresses **surgical phase recognition** in laparoscopic cholecystectomy videos.  
Given a stream of RGB video frames, the goal is to **predict the current surgical phase at each time step**.

Formally, this is a **supervised, discriminative sequence modeling problem**: we map visual observations over time to a categorical phase label, without assuming access to surgeon intent, internal state, or hidden variables.

The task is framed as **context estimation**, not evaluation, correction, or decision-making.

---

## 2. Why This Matters

Many downstream surgical systems—visualization, logging, indexing, safety monitoring—benefit from **knowing what phase of a procedure is currently happening**.

Accurate phase recognition enables:
- phase-aware interfaces and analytics
- temporal alignment of surgical events
- uncertainty-aware downstream systems

This project does **not** judge surgical performance or recommend actions.  
It strictly estimates **procedural context from visual data**.

---

## 3. Dataset

We use the **Cholec80 dataset**, a canonical benchmark for surgical phase recognition:

- 80 laparoscopic cholecystectomy videos  
- Frame-level annotations for standard surgical phases  
- Widely used in prior literature  

### Preprocessing
- Videos are sampled at a fixed frame rate (1–5 FPS, consistent across splits)
- RGB frames only (no multimodal fusion)
- Standard train / validation / test split
- No relabeling or dataset augmentation

---

## 4. Method

The model follows a **two-stage discriminative pipeline**:

### Visual Encoding
Each video frame is passed through a **pretrained visual encoder** (CNN or Vision Transformer) to extract a compact feature representation.

- Encoder weights may be frozen or lightly fine-tuned
- The classifier head is removed; only the encoder is used

### Temporal Modeling
Encoded frame features are fed into a **sequential model** to capture temporal structure:

- LSTM / GRU / sliding-window temporal models
- Outputs a phase probability distribution at each time step
- Final predictions obtained via softmax over phases

This architecture models **temporal continuity** without assuming explicit state transitions or latent dynamics.

---

## 5. Training Objective

The model is trained end-to-end using:

- **Cross-entropy loss** at the frame level
- Supervised learning with ground-truth phase labels
- Optional temporal smoothing through the sequence model

No reinforcement learning, planning, or policy optimization is used.

---

## 6. Evaluation

We evaluate the model using standard, interpretable metrics:

### Quantitative Metrics
- Frame-level accuracy  
- Per-phase accuracy  
- Confusion matrix  

### Temporal Analysis
- Timeline plots comparing predicted vs. ground-truth phases
- Phase probability trajectories over time
- Qualitative assessment of temporal consistency

Evaluation emphasizes **behavior over time**, not just aggregate accuracy.

---

## 7. Extension: Phase Uncertainty (Optional)

As a conservative extension, we analyze **model uncertainty** via the entropy of the predicted phase distribution.

Observations:
- Uncertainty typically increases near phase boundaries
- Stable phases show low-entropy predictions

This highlights where the model is **less confident**, which is valuable for safety-aware downstream systems.

Importantly, uncertainty is **descriptive**, not prescriptive.

---

## 8. Limitations

- Limited dataset size (80 videos)
- Single surgical procedure type
- Visual-only inputs
- No claims of generalization beyond Cholec80
- No clinical validation or deployment claims

This project demonstrates **methodology**, not clinical readiness.

---

## 9. Future Work

Possible future directions (not implemented here):

- Larger-scale or multi-procedure datasets
- Explicit phase-transition modeling
- World-model or latent-state approaches as internal representations
- Integration into passive surgical analytics systems

These are intentionally left as future explorations.

---

## 10. Summary

This repository presents a **clean, discriminative approach** to surgical phase recognition using:

- pretrained visual encoders  
- sequential temporal models  
- explicit uncertainty analysis  
- conservative evaluation  

The focus is on **procedural context estimation**, not autonomy or intervention.
