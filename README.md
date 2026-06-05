# RMR-Net Road Image Restoration for Detection and Boundary Analysis

RMR-Net is a research prototype for road-image restoration under blur, low light, defocus, compression, and vehicle-motion degradations. The code is designed around the practical question that matters in road monitoring: does restoration help downstream perception, especially YOLO defect detection and active-contour boundary extraction?

The repository contains:

- A PyTorch road-restoration model with metadata/degradation conditioning.
- Task-driven losses using frozen YOLO features, detector-safety terms, and differentiable contour regularization.
- YOLO split restoration/evaluation utilities.
- Active-contour boundary metric utilities.
- Reproducibility instructions for Windows GPU and Linux/Jetson-style runs.
- Lightweight audit files from the latest v24 detector-safe rerun.

## Important Status

This is an active research code release. The latest v24 code path compiles, trains, and evaluates cleanly, but the v24 detector-safe rerun did **not** surpass the strongest earlier v22/v23 local results. The v24 audit is included for transparency in `experiments/v24_task_loss_audit`.

For paper-quality claims, use validation-selected checkpoints and held-out test evaluation only. Do not select models using test mAP.

## Repository Layout

```text
.
├── models/
│   └── rmrnet.py                  # RMRNet wrapper and auxiliary outputs
├── rcadnet/
│   ├── model.py                   # Core restoration model
│   ├── dataset.py                 # Restoration dataset loader
│   ├── losses.py                  # Base restoration losses
│   ├── scenario_codes.py          # Synthetic degradation metadata codes
│   ├── synthetic_metadata.py      # Metadata utilities
│   └── task_losses.py             # TDP, Jacobian, contour, detector evidence losses
├── tools/
│   ├── restore_yolo_split.py      # Restore YOLO-format image splits
│   ├── eval_yolo_suite.py         # Evaluate YOLO mAP on multiple splits
│   ├── tune_residual_policy.py    # Validation-only residual policy tuning
│   ├── snake_boundary_metrics.py  # Active contour overlays and boundary metrics
│   └── prepare_*.py               # Dataset preparation helpers
├── baselines/                     # Lightweight DFPIR/NAFNet adapters
├── losses/                        # Earlier task-driven loss implementation
├── scripts/                       # Reproducible Windows PowerShell commands
├── docs/                          # Data, reproduction, results, and caveats
├── experiments/v24_task_loss_audit/
│   ├── smoke_test.json
│   ├── selection_summary.csv
│   └── checkpoints/best_by_val_map.json
├── train_rcadnet.py
├── infer_rcadnet.py
└── benchmark_unified_restoration.py
```

## Quick Start on Windows GPU

The development machine used for the latest rerun was Windows with an NVIDIA RTX 3050 GPU, Python 3.12, PyTorch CUDA, and Ultralytics YOLO.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-windows-gpu.txt
python -m pip install -r requirements.txt
python -m compileall models rcadnet tools train_rcadnet.py
```

Then run a tiny smoke test after preparing data:

```powershell
.\scripts\smoke_test.ps1
```

## Data

Large datasets and generated restored YOLO splits are intentionally not committed. See `docs/DATA.md` for the required folder structure and preparation commands.

Expected restoration layout:

```text
data/<dataset_name>/scenarios/<scenario>/input/*.jpg
data/<dataset_name>/scenarios/<scenario>/gt/*.jpg
```

Expected YOLO layout:

```text
datasets/<split_name>/
├── data.yaml
├── images/{train,val,test}/
└── labels/{train,val,test}/
```

## Reproducing the v24 Audit

The latest v24 rerun used validation mAP50 to choose checkpoints, then evaluated once on held-out test splits.

```powershell
.\scripts\train_ivcnz_v24.ps1
.\scripts\train_pcm_v24.ps1
.\scripts\evaluate_v24_selected.ps1
```

The included audit files summarize the rerun:

- `experiments/v24_task_loss_audit/smoke_test.json`
- `experiments/v24_task_loss_audit/selection_summary.csv`
- `experiments/v24_task_loss_audit/checkpoints/best_by_val_map.json`

## Method Summary

RMR-Net combines road-aware restoration with perception-oriented training:

- Degradation/metadata conditioning from synthetic scenario codes or real telemetry when available.
- Auxiliary contour maps `phi`, `lambda1`, and `lambda2` for train-time geometry losses.
- Frozen YOLO feature hooks for Task-Driven Perceptual loss.
- Hutchinson-estimated detector Jacobian penalty for cascaded stability.
- Detector input anchor and evidence non-regression terms to avoid suppressing road-defect cues.
- Optional validation-tuned residual policy for low-light deployment.

Auxiliary contour maps are train-time signals. They are not deployed as segmentation masks.

## Evaluation Protocol

Use three separate stages:

1. Train restoration on training split only.
2. Select checkpoint/policy on validation mAP50 only.
3. Report restoration metrics and YOLO mAP on held-out test splits only.

Recommended metrics:

- PSNR, SSIM, LPIPS if available.
- YOLO mAP50 and mAP50-95 before/after restoration.
- Runtime per image and backend status.
- Active-contour area, perimeter, compactness, and success/failure where boundary annotations or boxes are available.

## Notes for Reviewers and Reusers

- The repository does not claim that v24 is the final best model.
- The code includes both successful and unsuccessful ablations because they document what was tested.
- Dataset licenses must be respected. Download original datasets from their official sources.
- YOLO checkpoints are not committed. Train or download detectors separately.

## Citation

If this repository helps your research, cite the associated manuscript once available, or use the metadata in `CITATION.cff` as a placeholder.

