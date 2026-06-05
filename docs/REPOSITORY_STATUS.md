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

## Current Repository Owner

The repository was published under the signed-in/connected GitHub account:

```text
https://github.com/amirchoice/rmrnet-road-restoration
```

The originally requested owner name was `AmirNetwork`. Transfer the repository in GitHub settings if it should live under that account or organization instead.
