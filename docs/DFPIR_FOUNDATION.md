# Foundation Repository: DFPIR CVPR 2025

I selected **DFPIR: Degradation-Aware Feature Perturbation for All-in-One Image
Restoration** as the new foundation repo for this Windows GPU environment.

Why this repo:

- It is a **CVPR 2025** paper.
- It targets **all-in-one image restoration**, not just one degradation.
- It reports multiple benchmark tasks: denoising, dehazing, deraining, motion
  deblurring, and low-light enhancement.
- Its code is based on PromptIR-style restoration, so RCAD-Net can be compared
  in a familiar supervised restoration setting.
- Its degradation-conditioned design is conceptually close enough to our method
  to make the comparison scientifically meaningful, while still being distinct:
  DFPIR uses CLIP text degradation prompts; RCAD-Net uses road/sensor/degradation
  codes and defect-preserving attention.

Local location:

```text
third_party/DFPIR-main
```

Sources:

- Paper/arXiv: https://arxiv.org/abs/2505.12630
- Code: https://github.com/TxpHome/DFPIR

## Current Windows Setup

The machine has an RTX 3050 with 6 GB VRAM. The local environment is:

```text
.venv
Python 3.12
torch 2.11.0+cu128
torchvision 0.26.0+cu128
```

DFPIR dependencies installed into the same venv include CLIP, einops,
tensorboard, tqdm, matplotlib, scikit-image, opencv-python, and gdown.

Official weights are downloaded here:

```text
weights/dfpir
```

## Practical Benchmark Strategy

Use three tiers:

1. **Pipeline smoke benchmark**
   - synthetic/Kodak pairs
   - RCAD-Net trained briefly
   - DFPIR tiny random smoke model
   - purpose: verify code paths, GPU usage, metrics, timing

2. **Academic restoration benchmark**
   - DFPIR official datasets or generated benchmark-style paired folders
   - official DFPIR weights from the authors' Google Drive
   - RCAD-Net trained on the same train split
   - purpose: fair restoration metrics

3. **Road-specific benchmark**
   - road images with road-relevant degradation scenarios
   - RCAD-Net full training
   - DFPIR official/fine-tuned baseline if possible
   - downstream detector mAP before/after restoration
   - purpose: paper claim about road-defect preservation

## Commands

Train RCAD-Net on the local synthesized Kodak benchmark:

```powershell
.\.venv\Scripts\python.exe train_rcadnet.py --data-root data\kodak_restoration_benchmark --epochs 20 --batch-size 1 --patch-size 128 --width 32 --device cuda --out runs\rcadnet_kodak
```

Run RCAD-Net benchmark:

```powershell
.\.venv\Scripts\python.exe benchmark_unified_restoration.py --data-root data\kodak_restoration_benchmark --model rcadnet --rcadnet-weights runs\rcadnet_kodak\rcadnet_last.pth --limit 4 --max-side 384 --device cuda --out runs\bench_rcadnet
```

Run DFPIR smoke benchmark:

```powershell
.\.venv\Scripts\python.exe benchmark_unified_restoration.py --data-root data\kodak_restoration_benchmark --model dfpir --dfpir-smoke --limit 2 --max-side 128 --device cuda --out runs\bench_dfpir_smoke
```

For real DFPIR comparison, download official weights into `weights/dfpir/`, then:

```powershell
.\.venv\Scripts\python.exe benchmark_unified_restoration.py --data-root data\kodak_restoration_benchmark --model dfpir --dfpir-weights weights\dfpir\DFPIR_5D.pth.tar --dfpir-clip --limit 4 --max-side 384 --device cuda --out runs\bench_dfpir_official
```

Because the downloaded filename includes metric values, this PowerShell-safe
variant finds it automatically:

```powershell
$w=(Get-ChildItem weights\dfpir -Filter '*5D*.pth.tar').FullName
.\.venv\Scripts\python.exe benchmark_unified_restoration.py --data-root data\kodak_restoration_benchmark --model dfpir --dfpir-weights $w --dfpir-clip --scenario motion_horizontal_medium --scenario defocus_medium --limit 4 --max-side 128 --warmup 2 --device cuda --out runs\bench_dfpir_official_4img
```

## Local Smoke Results

These are small sanity checks on four Kodak-derived images resized to long side
128. They are not paper results, but they prove the training/evaluation stack is
working on the RTX 3050.

| model | scenario | PSNR | SSIM | mean runtime ms | backend |
|---|---:|---:|---:|---:|---|
| RCAD-Net, 10 epochs tiny width | motion_horizontal_medium | 22.96 | 0.730 | 13.24 | GPU-confirmed |
| RCAD-Net, 10 epochs tiny width | defocus_medium | 24.21 | 0.805 | 7.23 | GPU-confirmed |
| DFPIR official 5D | motion_horizontal_medium | 30.23 | 0.903 | 106.27 | GPU-confirmed |
| DFPIR official 5D | defocus_medium | 33.11 | 0.942 | 91.37 | GPU-confirmed |

Interpretation: official DFPIR is much stronger on this tiny generic restoration
slice, while the small RCAD-Net checkpoint is faster. The next research step is
to train RCAD-Net at full width and on road-specific degradation pairs, then add
road-defect downstream detection metrics where generic restoration may oversmooth
small defect structures.
