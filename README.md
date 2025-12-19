# Surgical Phase Recognition (Cholec80)

## 1. Problem Statement

We model **surgical phase recognition** as a problem of **latent state inference under partial observability**. At any time *t*, the true surgical phase is not directly observed; instead, we observe noisy visual evidence from laparoscopic video. The goal is to infer a **probability distribution over phases** given RGB video frames, and to maintain temporal consistency without assuming perfect observability.

This project focuses on **context estimation**, not action, recommendation, or evaluation of surgical performance.

---

## 2. Why This Matters

Reliable phase awareness is a foundational capability for downstream surgical systems (e.g., logging, indexing, safety monitoring, or phase-aware tooling). In high-stakes environments, models must:

* Be **probabilistic**, not overconfident
* Expose **uncertainty**, especially near transitions
* Be **inspectable** via timelines and plots

This project does **not** claim clinical impact. It demonstrates disciplined system design appropriate for safety-critical domains.

---

## 3. Dataset

**Cholec80**

* 80 laparoscopic cholecystectomy videos
* Frame-level annotations for standard surgical phases
* Canonical benchmark used across the literature

### Preprocessing

* Frames sampled at a fixed FPS (1–5 FPS, held constant across splits)
* Standard train / validation / test split
* RGB frames only
* No new labels, no multimodal fusion

---

## 4. Method

### Inputs

* RGB video frames (single frames or short temporal windows)

### Outputs

* Softmax probability distribution over surgical phases at time *t*

### Architecture (Intentionally Simple)

1. **Visual Encoder**

   * CNN or Vision Transformer (pretrained allowed)
   * Extracts frame-level visual features

2. **Temporal Smoothing**

   * Lightweight temporal model (LSTM, GRU, or sliding window smoothing)
   * Enforces short-term temporal coherence

3. **Classifier Head**

   * Linear layer + softmax over phases

This is **phase inference**, not detection of correctness or skill.

---

## 5. Training & Evaluation

### Metrics (Required)

* **Frame-level accuracy**
* **Confusion matrix**
* **Per-phase accuracy**
* **Temporal consistency** (qualitative)

The goal is **clarity and reliability**, not metric maximization.

---

## 6. Visualizations (Required)

At minimum, the following plots are produced:

1. **Timeline Plot**

   * Ground truth phase vs. predicted phase over time

2. **Phase Probability Over Time**

   * Stacked probabilities or entropy curve

These visualizations are treated as first-class outputs and matter more than architectural complexity.

---

## 7. Extension: Phase Uncertainty (Chosen)

### What Was Added

We compute the **entropy of the phase posterior** at each time step:

* High entropy → model uncertainty
* Low entropy → confident phase assignment

### Observations

* Uncertainty spikes consistently near **phase transitions**
* Confidence stabilizes during steady-state phases

### Why This Is Conservative

* No action is taken based on uncertainty
* No recommendations are made
* Uncertainty is exposed purely for **situational awareness**

This framing aligns with safety-first system design.

---

## 8. Results

Results are reported via:

* Aggregate metrics (accuracy, confusion matrix)
* Per-phase performance breakdown
* Qualitative timeline + uncertainty plots

Discussion focuses on **failure modes**, transition ambiguity, and dataset limitations.

---

## 9. Limitations

* Dataset bias (single procedure type)
* Limited generalization across institutions or tools
* Frame-based modeling misses fine-grained motion cues
* No clinical claims or validation

---

## 10. Future Work

* Longer-horizon temporal models
* Phase-transition anticipation (predictive only)
* Integration into larger **context-aware systems**
* World-model or agent-based extensions as internal research tools only

---

## 11. Project Scope & Constraints

* No reinforcement learning
* No agents acting on surgery
* No simulation
* No multi-dataset fusion
* No claims of clinical utility

The project is intentionally narrow, inspectable, and conservative.

---

## 12. Summary

This repository demonstrates a **probabilistic surgical context-estimation system** with explicit uncertainty and clean evaluation. The focus is on disciplined modeling, transparency, and safety-aligned design — not novelty chasing.
