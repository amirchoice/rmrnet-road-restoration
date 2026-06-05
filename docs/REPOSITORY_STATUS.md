# Repository Status and Caveats

This repository is a cleaned research-code release from an active experiment series.

## What Is Included

- Latest v24 model/loss/training code.
- Dataset preparation and evaluation utilities.
- Lightweight v24 audit output.
- Reproducible command scripts.

## What Is Not Included

- Full datasets.
- Trained YOLO detector checkpoints.
- RMR-Net restoration checkpoints.
- Generated restored YOLO image folders.
- Paper ZIPs and large qualitative panels.

## Why Large Files Are Excluded

The local workspace contains many generated experiments, checkpoints, and restored image splits. Uploading those directly would make the repository difficult to clone and may violate dataset/checkpoint redistribution rules.

Recommended release pattern:

- GitHub: source code, configs, docs, lightweight audit CSV/JSON.
- GitHub Releases or Zenodo: optional trained checkpoints.
- Dataset provider pages: raw datasets.
- README/docs: exact commands to reproduce generated splits.

## Current Account Note

The GitHub connector in the development environment authenticated as `amirchoice`, while the requested public repository owner was `AmirNetwork`. Create or push the final repository under the account that should own the project.

