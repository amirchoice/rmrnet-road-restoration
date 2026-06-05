# RCAD-Net Road-Defect Restoration Experiments

Date: 2026-05-24  
Machine: Windows, NVIDIA GeForce RTX 3050 6 GB  
Environment: `.venv`, Python 3.12, `torch 2.11.0+cu128`

## Foundation Baseline

The main academic baseline is official **DFPIR-CVPR2025**:

- Repository: https://github.com/TxpHome/DFPIR
- Paper: https://arxiv.org/abs/2505.12630
- Local repo: `third_party/DFPIR-main`
- Official 5D checkpoint: `weights/dfpir/*5D*.pth.tar`

DFPIR is a strong all-in-one restoration baseline covering denoising, dehazing,
deraining, motion deblurring, and low-light enhancement. This makes it a better
foundation than a narrow single-degradation repo.

## Road-Defect Detection Dataset

Dataset used for downstream detection:

- IVCNZ pothole YOLO dataset release from GitHub
- 1,243 images with YOLO-format labels
- Split created locally:
  - train: 870
  - val: 186
  - test: 187
- Dataset YAML: `datasets/pothole_yolo/data.yaml`

YOLO detector:

- Model: `yolov8n.pt`
- Fine-tuned on clean pothole train split
- Epochs: 12
- Image size: 320
- Best weights: `runs/detect/runs/detect_train/yolov8n_pothole_clean/weights/best.pt`

Clean validation performance:

| split | mAP50 | mAP50-95 | precision | recall |
|---|---:|---:|---:|---:|
| clean val | 0.5949 | 0.2712 | 0.6588 | 0.5339 |
| clean test | 0.5949 | 0.2775 | 0.6560 | 0.5365 |

## Downstream Detection Results

The same trained YOLO detector was evaluated on clean, degraded, and restored
validation images. Labels are unchanged because degradations/restorations do not
move pothole boxes.

| scenario | input to detector | mAP50 | mAP50-95 | precision | recall |
|---|---|---:|---:|---:|---:|
| clean | clean | 0.5949 | 0.2712 | 0.6588 | 0.5339 |
| motion horizontal | degraded | 0.2227 | 0.0849 | 0.3342 | 0.2636 |
| motion horizontal | RCAD-Net restored | 0.3261 | 0.1276 | 0.4544 | 0.3203 |
| motion horizontal | DFPIR restored | 0.4426 | 0.1885 | 0.5617 | 0.4069 |
| defocus | degraded | 0.2759 | 0.1139 | 0.4859 | 0.2569 |
| defocus | RCAD-Net restored | 0.4899 | 0.2080 | 0.6228 | 0.4502 |
| defocus | DFPIR restored | 0.3450 | 0.1421 | 0.5013 | 0.3247 |
| low light | degraded | 0.5155 | 0.2280 | 0.6089 | 0.4545 |
| low light | RCAD-Net restored | 0.5401 | 0.2421 | 0.6455 | 0.4820 |
| low light | DFPIR restored | 0.5238 | 0.2305 | 0.6265 | 0.4622 |

## Detection Improvement

Validation gains:

| scenario | method | mAP50 gain over degraded | mAP50-95 gain over degraded |
|---|---|---:|---:|
| motion horizontal | RCAD-Net | +0.1034 | +0.0427 |
| motion horizontal | DFPIR | +0.2200 | +0.1036 |
| defocus | RCAD-Net | +0.2140 | +0.0941 |
| defocus | DFPIR | +0.0691 | +0.0282 |
| low light | RCAD-Net | +0.0246 | +0.0141 |
| low light | DFPIR | +0.0083 | +0.0025 |

## Held-Out Test Detection Results

The held-out test split was kept separate from the restoration-training and
YOLO-validation workflow. It has 187 pothole images and uses the same detector
weights as the validation table.

| scenario | input to detector | mAP50 | mAP50-95 | precision | recall |
|---|---|---:|---:|---:|---:|
| clean | clean | 0.5949 | 0.2775 | 0.6560 | 0.5365 |
| motion horizontal | degraded | 0.1954 | 0.0716 | 0.3396 | 0.2383 |
| motion horizontal | RCAD-Net restored | 0.3035 | 0.1225 | 0.4335 | 0.3012 |
| motion horizontal | DFPIR restored | 0.4632 | 0.1977 | 0.5695 | 0.4101 |
| defocus | degraded | 0.2505 | 0.0973 | 0.4451 | 0.2295 |
| defocus | RCAD-Net restored | 0.4900 | 0.2174 | 0.6440 | 0.4357 |
| defocus | DFPIR restored | 0.3228 | 0.1289 | 0.4873 | 0.3158 |
| low light | degraded | 0.5132 | 0.2382 | 0.6557 | 0.4343 |
| low light | RCAD-Net restored | 0.5546 | 0.2586 | 0.6211 | 0.5000 |
| low light | DFPIR restored | 0.5193 | 0.2388 | 0.6497 | 0.4591 |

Held-out test gains:

| scenario | method | mAP50 gain over degraded | mAP50-95 gain over degraded |
|---|---|---:|---:|
| motion horizontal | RCAD-Net | +0.1082 | +0.0509 |
| motion horizontal | DFPIR | +0.2679 | +0.1261 |
| defocus | RCAD-Net | +0.2395 | +0.1201 |
| defocus | DFPIR | +0.0722 | +0.0315 |
| low light | RCAD-Net | +0.0414 | +0.0204 |
| low light | DFPIR | +0.0061 | +0.0006 |

Interpretation:

- RCAD-Net improves downstream detection after all tested degradations on both
  validation and held-out test splits.
- Official DFPIR is still stronger for motion blur detection recovery.
- RCAD-Net is stronger for defocus and low-light downstream detection on this
  road-defect dataset.
- This supports the research direction: road-aware restoration should be judged
  by whether it preserves defect evidence, not only by generic PSNR.

## Road Restoration Metrics

Evaluation set: 186 pothole validation images per scenario, resized to long side
320 before degradation. Metrics are full-reference against the clean validation
images. Runtime is per image on RTX 3050.

| scenario | model | PSNR | SSIM | mean runtime ms | backend |
|---|---|---:|---:|---:|---|
| motion horizontal | RCAD-Net | 24.2544 | 0.6071 | 40.68 | GPU-confirmed |
| motion horizontal | DFPIR-CVPR2025 | 25.3166 | 0.6634 | 389.64 | GPU-confirmed |
| defocus | RCAD-Net | 25.0921 | 0.6375 | 40.36 | GPU-confirmed |
| defocus | DFPIR-CVPR2025 | 24.2514 | 0.5608 | 384.48 | GPU-confirmed |
| low light | RCAD-Net | 28.6333 | 0.8372 | 40.19 | GPU-confirmed |
| low light | DFPIR-CVPR2025 | 15.6527 | 0.7615 | 387.19 | GPU-confirmed |

Interpretation:

- RCAD-Net is about 9.5x faster than DFPIR at this 320 px validation scale.
- DFPIR is better on motion-blur PSNR/SSIM and downstream mAP.
- RCAD-Net is better on defocus and low-light road restoration metrics here.
- The strongest current claim is not "RCAD-Net dominates every case"; it is:
  **RCAD-Net is a lightweight road-specific restorer that improves pothole
  detection under realistic degradations and beats a strong generic CVPR
  baseline on defocus/low-light road cases while running much faster.**

## RCAD-Net++ Robustness Update

After the first RCAD-Net experiment, I added a self-contained method variant:
**RCAD-Net++**. The new component is a blind image-derived degradation encoder
that predicts the same soft degradation code used by the scenario/metadata
conditioner. During training, this estimated code is fused with the synthetic
scenario code and lightly supervised. At inference time, the model can run from
the image alone, or it can fuse future IMU/camera metadata if available.

This is the better final paper method because it no longer depends on future
sensor data and it improves the actual downstream task.

Held-out test detection comparison:

| scenario | input to detector | mAP50 | mAP50-95 | precision | recall |
|---|---|---:|---:|---:|---:|
| motion horizontal | degraded | 0.1954 | 0.0716 | 0.3396 | 0.2383 |
| motion horizontal | RCAD-Net | 0.3035 | 0.1225 | 0.4335 | 0.3012 |
| motion horizontal | RCAD-Net++ | 0.4568 | 0.2026 | 0.5681 | 0.4123 |
| motion horizontal | DFPIR | 0.4632 | 0.1977 | 0.5695 | 0.4101 |
| defocus | degraded | 0.2505 | 0.0973 | 0.4451 | 0.2295 |
| defocus | RCAD-Net | 0.4900 | 0.2174 | 0.6440 | 0.4357 |
| defocus | RCAD-Net++ | 0.4942 | 0.2156 | 0.6442 | 0.4401 |
| defocus | DFPIR | 0.3228 | 0.1289 | 0.4873 | 0.3158 |
| low light | degraded | 0.5132 | 0.2382 | 0.6557 | 0.4343 |
| low light | RCAD-Net | 0.5546 | 0.2586 | 0.6211 | 0.5000 |
| low light | RCAD-Net++ | 0.5572 | 0.2598 | 0.6515 | 0.4781 |
| low light | DFPIR | 0.5193 | 0.2388 | 0.6497 | 0.4591 |

Held-out test restoration/runtime comparison:

| scenario | model | PSNR | SSIM | mean runtime ms | backend |
|---|---|---:|---:|---:|---|
| motion horizontal | RCAD-Net | 24.5370 | 0.6321 | 34.67 | GPU-confirmed |
| motion horizontal | RCAD-Net++ | 23.7167 | 0.5858 | 44.42 | GPU-confirmed |
| motion horizontal | DFPIR-CVPR2025 | 26.4397 | 0.7096 | 419.37 | GPU-confirmed |
| defocus | RCAD-Net | 23.1633 | 0.6632 | 39.82 | GPU-confirmed |
| defocus | RCAD-Net++ | 22.5157 | 0.6478 | 42.30 | GPU-confirmed |
| defocus | DFPIR-CVPR2025 | 25.9821 | 0.6722 | 410.24 | GPU-confirmed |
| low light | RCAD-Net | 31.9727 | 0.9175 | 39.62 | GPU-confirmed |
| low light | RCAD-Net++ | 31.0204 | 0.9173 | 47.30 | GPU-confirmed |
| low light | DFPIR-CVPR2025 | 16.2565 | 0.8310 | 402.47 | GPU-confirmed |

Updated interpretation:

- RCAD-Net++ is the strongest downstream-detection method variant.
- On motion blur, RCAD-Net++ almost closes the gap to DFPIR in mAP50
  (0.4568 vs 0.4632) and slightly exceeds DFPIR in mAP50-95.
- On defocus and low light, RCAD-Net++ beats DFPIR clearly in downstream mAP.
- DFPIR remains stronger on full-reference motion/defocus PSNR and SSIM.
- The paper should emphasize task-driven restoration: better road-defect
  visibility and detector recovery, not universal PSNR dominance.

Paper-ready assets were generated here:

- `paper_assets/rcadnetpp_2026-05-24/tables/`
- `paper_assets/rcadnetpp_2026-05-24/figures/`
- `paper_assets/rcadnetpp_2026-05-24/PAPER_ASSET_SUMMARY.md`

## Kodak-24 Appendix Benchmark

To avoid evaluating only on one image source, I also prepared a full Kodak-24
synthetic restoration benchmark at 384 px max side and evaluated RCAD-Net++ and
DFPIR at 320 px long side. This is an appendix-style sanity check, not the main
road-defect result.

| scenario | model | PSNR | SSIM | mean runtime ms |
|---|---|---:|---:|---:|
| defocus | RCAD-Net++ | 24.6873 | 0.7499 | 31.15 |
| defocus | DFPIR-CVPR2025 | 27.0201 | 0.7835 | 379.51 |
| gaussian sigma3 | RCAD-Net++ | 18.3228 | 0.7398 | 38.27 |
| gaussian sigma3 | DFPIR-CVPR2025 | 34.2521 | 0.9422 | 416.95 |
| jpeg40 motion | RCAD-Net++ | 18.8148 | 0.6678 | 39.15 |
| jpeg40 motion | DFPIR-CVPR2025 | 27.5711 | 0.8043 | 424.32 |
| low light | RCAD-Net++ | 23.9535 | 0.8761 | 40.65 |
| low light | DFPIR-CVPR2025 | 15.4984 | 0.8235 | 412.55 |
| motion horizontal | RCAD-Net++ | 24.5699 | 0.6937 | 40.74 |
| motion horizontal | DFPIR-CVPR2025 | 27.4675 | 0.8152 | 417.41 |
| motion vertical | RCAD-Net++ | 17.4420 | 0.5812 | 41.06 |
| motion vertical | DFPIR-CVPR2025 | 27.5927 | 0.7984 | 413.79 |

Interpretation:

- DFPIR is much stronger on generic Kodak denoising/motion/defocus restoration.
- RCAD-Net++ is not a universal natural-image restoration replacement.
- RCAD-Net++ is faster and remains strong for low-light enhancement.
- This supports a focused paper claim: road-task-aware restoration for defect
  detection, not generic all-in-one restoration dominance.

## Key Commands

Train detector:

```powershell
.\.venv\Scripts\python.exe -c "from ultralytics import YOLO; model=YOLO('yolov8n.pt'); model.train(data='datasets/pothole_yolo/data.yaml', epochs=12, imgsz=320, batch=8, device=0, project='runs/detect_train', name='yolov8n_pothole_clean', workers=0, patience=5)"
```

Train road-specific RCAD-Net:

```powershell
.\.venv\Scripts\python.exe train_rcadnet.py --data-root data\pothole_restoration --scenario motion_horizontal_medium --scenario defocus_medium --scenario lowlight_medium --epochs 12 --batch-size 4 --patch-size 192 --width 32 --device cuda --out runs\rcadnet_pothole_road_12ep --num-workers 0
```

Evaluate restoration:

```powershell
$w=(Get-ChildItem weights\dfpir -Filter '*5D*.pth.tar').FullName
.\.venv\Scripts\python.exe benchmark_unified_restoration.py --data-root data\pothole_restoration_val --model rcadnet --model dfpir --rcadnet-weights runs\rcadnet_pothole_road_12ep\rcadnet_last.pth --dfpir-weights $w --dfpir-clip --scenario motion_horizontal_medium --scenario defocus_medium --scenario lowlight_medium --max-side 320 --warmup 2 --device cuda --out runs\bench_pothole_restoration_val
```
