# Tomato Disease and Pest Surveillance System
## Final Industry Deployment Recommendation

**Document ID**: REQ-10-DEPLOY-REC-2026  
**Audience**: Technical Directors, Agricultural Stakeholders, Academic Review Panel  
**Evaluated Model**: ResNet18 Fine-Tuned with Square-Root Inverse Frequency Class Weighting (`best_resnet18_weighted.pt`)  
**SHA-256 Checksum**: `b86d760146c7e8389600bb56d63dca9c552bb0a649d01e74feb1cb19bfa4a3fa`  
**Test Set Scope**: 2,724 holdout tomato leaf images across 10 pathological categories  
**Date**: November 15, 2026  

---

## 1. Executive Deployment Matrix

| Deployment Scenario | Decision Tier | Key Prerequisite / Constraint | Recommended Hardware |
|:---|:---:|:---|:---|
| **Greenhouse & Controlled Inspection** | **GO** | Standardized lighting, single-leaf presentation | Web Dashboard / Edge Gateway |
| **Mobile Agronomist Assistive App** | **CONDITIONAL GO** | Calibrated $T^*=1.3021$, blur filter, confidence $\ge 85\%$ | Smartphone (iOS / Android) |
| **Edge Video Monitoring (Sorting Line)** | **CONDITIONAL GO** | Fixed focus camera, conveyor belt illumination | Jetson Orin Nano / Raspberry Pi 5 |
| **Pest Object Detection (Field)** | **NO-GO** | Requires in-situ annotated bounding boxes | N/A (Research Benchmark Only) |
| **Autonomous Robotic Pesticide Spraying** | **NO-GO** | Safety & liability risk; lacks field noise immunity | N/A (Strictly Prohibited) |

---

## 2. Evidence-Based Capability Breakdown

### 2.1 Greenhouse & Controlled Facility Diagnostics: **GO**
* **Technical Evidence**:
  - Test Accuracy: **99.23%** (2,703 / 2,724 correct)
  - Macro F1: **0.9885**, Weighted F1: **0.9923**
  - Healthy Foliage Recall: **100.0%** (0 false negatives out of 238 healthy leaves)
  - Calibrated ECE: **0.364%**
  - Single-Sample Latency: **16.57 ms** (60.4 FPS)
* **Operational Scope**:
  - Controlled greenhouse inspection stations.
  - Post-harvest sorting facilities.
  - Agricultural extension laboratory intake.
* **Verdict**: **Approved for immediate commercial pilot deployment.**

---

### 2.2 Mobile Agronomist Assistive Triage: **CONDITIONAL GO**
* **Technical Evidence**:
  - The model maintains high accuracy under moderate illumination shifts (93.20% at +55% brightness, 93.00% at -55% brightness) and JPEG compression (95.20% at quality 20).
  - However, handheld motion blur degrades accuracy to 37.60% at moderate severity and 31.80% at high severity.
* **Mandatory Operational Safeguards**:
  1. **Post-Hoc Calibration**: The client must scale logits using temperature $T^* = 1.3021$ (`artifacts/checkpoints/temperature_scale.json`).
  2. **Pre-Inference Quality Gate**: Implement a Laplacian variance blur detector ($Var(\Delta I) > 100$) before passing frames to the CNN.
  3. **Agronomist Human-in-the-Loop**:
     - Predictions with confidence $\ge 85\%$: Report diagnosis with recommended organic/chemical intervention.
     - Predictions with confidence $< 85\%$: Flag as "Ambiguous — Requires Agronomist Visual Inspection".
* **Verdict**: **Approved for assisted human-in-the-loop deployment only.**

---

### 2.3 Autonomous Drone / Chemical Spraying: **NO-GO**
* **Technical Evidence**:
  - High-frequency Gaussian sensor noise drops accuracy to **13.40%** at Severity 3.
  - Adversarial perturbations ($\epsilon = 0.005$) drop accuracy to **4.69%** while model confidence remains **97.67%**.
  - Misclassifying a fungal disease (*Alternaria solani*) as bacterial (*Xanthomonas*) could lead to improper spraying, destroying crops and violating chemical compliance standards.
* **Verdict**: **Strictly NO-GO for autonomous actuation.** Autonomous chemical dispersal requires multi-spectral imaging, cross-sensor fusion, and strict physical fail-safes.

---

### 2.4 Pest Detection & Lesion Segmentation in Field: **NO-GO**
* **Technical Evidence**:
  - While the U-Net and detection architectures are fully implemented, verified, and testable within the codebase, training was performed on synthetic lesion overlays and synthetic pest boxes. Real PlantVillage images lack polygon segmentation masks and bounding box labels.
* **Verdict**: **NO-GO for commercial field surveillance until real-world annotated field datasets are integrated.**

---

## 3. Deployment Specifications Summary

- **Production Checkpoint**: `artifacts/checkpoints/best_resnet18_weighted.pt`
- **Temperature Scale**: `artifacts/checkpoints/temperature_scale.json` ($T^* = 1.3021$)
- **Input Resolution**: $224 \times 224 \times 3$ RGB
- **Normalization**: ImageNet Mean `[0.485, 0.456, 0.406]`, Std `[0.229, 0.224, 0.225]`
- **Inference Mode**: Deterministic evaluation (`torch.inference_mode()`)
- **Memory Footprint**: 76.15 MB RAM (FP32 weights + forward activations at $B=1$)
- **Real-Time Speed**: 16.57 ms per sample on CPU (P95: 17.75 ms)
