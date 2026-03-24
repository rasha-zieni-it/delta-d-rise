# DELTA-D-RISE

**DELTA-D-RISE: Detection-Loss Based Perturbation Explainability for Object Detection**

---

## Overview

Object detection models are widely used in safety-critical applications, but their predictions remain difficult to interpret.

DELTA-D-RISE is a **perturbation-based explainability method** designed specifically for object detection.  
Unlike existing approaches such as D-RISE, it defines pixel importance using **detection loss deterioration**, directly linking explanations to model performance.

---

## Method

For a given detection, we define a detection loss:

L_det = L_cls + L_loc

- **Classification loss**:  
  L_cls = -log(p_class)

- **Localization loss**:  
  L_loc = 1 - IoU(b_base, b_masked)

We then define importance as:

Δ = max(0, L_masked - L_base)

This measures **how much masking a region degrades detection performance**.

The final saliency map is computed by aggregating normalized Δ scores over random masks.

---

## Key Features

- Detection-aware saliency (classification + localization)
- Model-agnostic (no gradients required)
- Per-detection explanations
- Robustness and deletion-based evaluation
- Compatible with any object detector via wrapper

---


## Installation

```bash
git clone https://github.com/USER/delta-d-rise.git
cd delta-d-rise
pip install -e .

## Demo

A complete usage example is provided in:

notebooks/demo.ipynb

The notebook demonstrates:

- Running DELTA-D-RISE on an object detector
- Generating per-detection saliency maps
- Visualizing heatmaps
- Running the full explainability pipeline

## Paper

This repository accompanies the paper:

**Explainability Metrics for Object Detection Models**

Rasha Zieni et al.

(2026)

[Add link here when available]

## Acknowledgment

This work was supported by the **ENFIELD Project** ("SAFER AI: Secure, Accurate, Fair, Explainable and Robust AI"), funded by the European Union under grant agreement No 101120657.


## Authors

- Rasha Zieni  
- François Picard  
- Leïla Belmerhnia  
- Georgios Spathoulas  
- Walter Quadrini  
- Paolo Giudici  
