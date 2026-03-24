<h1 align="center">DELTA-D-RISE</h1>

This repository contains the code for our explainability method for object detection models:

**DELTA-D-RISE — a detection-loss based perturbation explainability method.**


## Project description

The goal of this project is to explain object detector predictions by measuring how much the model’s performance degrades when parts of the image are removed.

Instead of using heuristic similarity scores (like D-RISE), we define importance using detection loss increase:

- classification loss (confidence drop)
- localization loss (IoU drop)

For each detection, we generate a saliency map highlighting the regions that matter most for that prediction.

We also evaluate explanations using:
- robustness (consistency across runs)
- fidelity (deletion test)
- object focus (saliency inside bounding boxes)


## Demo

A full example is available in:

notebooks/demo.ipynb

The notebook shows how to:
- run DELTA-D-RISE on a detector
- visualize saliency maps and heatmaps
- run the full pipeline with evaluation


## Notes

- The method is model-agnostic: any detector can be used via a wrapper.
- A YOLO wrapper is included as an example.

## Paper

This repository is based on our paper:

Explainability Metrics for Object Detection Models

(Add link when available)


## Acknowledgment

This work was funded by the EC ENFIELD Project. The ”SAFER
AI: Secure, Accurate, Fair, Explainable and Robust AI” project has received
funding from the European Union, via the oc3-2025-TES-01 issued and imple-
mented by the ENFIELD project, under the grant agreement No 101120657.


## How to run

Clone the repository:

```bash
git clone https://github.com/rasha-zieni-it/delta-d-rise.git
cd ddelta-d-rise


