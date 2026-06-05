# Reproducibility Guide

## 1. Create Environment

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-windows-gpu.txt
python -m pip install -r requirements.txt
python -m compileall models rcadnet tools train_rcadnet.py
```

Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-windows-gpu.txt
python -m pip install -r requirements.txt
python -m compileall models rcadnet tools train_rcadnet.py
```

For Jetson, install the NVIDIA-provided PyTorch wheel for your JetPack version instead of the Windows CUDA wheel index.

## 2. Verify GPU

```powershell
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
PY
```

## 3. Smoke Test

Prepare a tiny paired dataset in:

```text
data/synthetic_road_smoke/scenarios/motion_horizontal_medium/input
data/synthetic_road_smoke/scenarios/motion_horizontal_medium/gt
```

Then:

```powershell
.\scripts\smoke_test.ps1
```

## 4. Train

The v24 rerun used a pretrained restoration checkpoint as initialization when available:

```text
runs/rmrnet_revised_evidence_30ep/rcadnet_best.pth
```

If that file is missing, remove `--init-weights ...` from the training scripts and train from scratch for more epochs.

```powershell
.\scripts\train_ivcnz_v24.ps1
.\scripts\train_pcm_v24.ps1
```

## 5. Evaluate

The strict selection rule is:

1. Evaluate each epoch on validation YOLO splits.
2. Select the epoch with best mean validation mAP50.
3. Evaluate that selected epoch once on held-out test splits.

```powershell
.\scripts\evaluate_v24_selected.ps1
```

## 6. Boundary Metrics

After running YOLO prediction export, use:

```powershell
python tools/snake_boundary_metrics.py --help
```

The active-contour module is useful for visual overlays and geometry metrics such as area, perimeter, compactness, and success/failure. It should not be treated as a replacement for detection mAP.

