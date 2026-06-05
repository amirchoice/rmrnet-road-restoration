# Release Manifest

Prepared on 2026-06-05 for GitHub publication.

## Included

- Source code:
  - `models/`
  - `rcadnet/`
  - `tools/`
  - `baselines/`
  - `losses/`
  - top-level training/inference/benchmark scripts
- Reproducibility:
  - `README.md`
  - `requirements.txt`
  - `requirements-windows-gpu.txt`
  - `environment-windows-gpu.yml`
  - `scripts/*.ps1`
  - `docs/*.md`
- Lightweight audit:
  - `experiments/v24_task_loss_audit/smoke_test.json`
  - `experiments/v24_task_loss_audit/selection_summary.csv`
  - `experiments/v24_task_loss_audit/checkpoints/best_by_val_map.json`

## Excluded

- `.venv/`
- `data/`
- `datasets/`
- `runs/`
- `weights/`
- YOLO checkpoints
- RMR-Net checkpoints
- generated restored images
- paper ZIPs and large qualitative figures

These files are excluded to keep the GitHub repository lightweight and to avoid redistributing datasets/checkpoints with uncertain licenses.

