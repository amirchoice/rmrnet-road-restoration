# V25 Evidence-Preserving Detail Audit

This audit evaluates the v25 structural update to RMR-Net: a bounded evidence-preserving high-frequency skip placed in the model decoder rather than another detector-heavy auxiliary loss.

## What Changed

- Added `EvidencePreservingDetailSkip` in `rcadnet/model.py`.
- The module predicts a bounded spatial gate from decoder features plus image evidence cues: edge, local contrast, dark-region evidence, and saturation.
- The output is `restored_base + max_gain * gate * highpass(input)`, clamped to `[0, 1]`.
- The deployed inference output remains a restored RGB image. The detail gate is an internal diagnostic, not a mask used for downstream supervision.

## Training Protocol

- Training data: `data/pothole_restoration` and `data/pcm_restoration_train`.
- Validation data: `data/pothole_restoration_val` and `data/pcm_restoration_test`.
- Scenarios: `motion_horizontal_medium`, `defocus_medium`, `lowlight_medium`.
- Initialization: `runs/rmrnet_revised_evidence_30ep/rcadnet_best.pth`.
- Objective: base restoration loss with L1, edge, frequency, defect, and visibility terms plus auxiliary degradation-code supervision.
- Task losses were disabled in the v25 run. Current `train_rcadnet.py` logs effective task weights as zero when task losses are disabled; the historical v25 `history.json` contains older nonzero labels and should be read together with `audit_config.json`, where `task_losses_enabled=false`.

## Validation Selection

- Pothole detector: `runs/detect/runs/detect_train/yolov8n_pothole_clean/weights/best.pt`.
- PCM detector: `runs/detect/runs/detect_train/yolov8n_pcm_clean_25ep/weights/best.pt`.
- YOLO evaluation image size: 320.
- Pothole selected checkpoint: epoch 001, detail gain 0.20.
- PCM selected checkpoint: epoch 003, detail gain 0.20.
- Detail-gain calibration tested `0.20`, `0.35`, and `0.50`; stronger gains did not improve mean validation mAP50.

## Held-Out Detection Result

Against degraded inputs, v25 produces large detector recovery on all six held-out rows. Against v24 under the same current detectors and 320-pixel evaluation, v25 is a modest structural improvement rather than a new main benchmark jump:

- Pothole motion: mAP50 0.541 vs v24 0.540; mAP50-95 0.245 vs 0.244.
- Pothole defocus: mAP50 0.518 vs v24 0.519; mAP50-95 0.232 vs 0.229.
- Pothole low-light: mAP50 0.569 vs v24 0.570; mAP50-95 0.262 vs 0.260.
- PCM motion: mAP50 0.253 vs v24 0.247; mAP50-95 0.090 vs 0.090.
- PCM defocus: mAP50 0.257 vs v24 0.254; mAP50-95 0.093 vs 0.092.
- PCM low-light: mAP50 0.285 vs v24 0.285; mAP50-95 0.107 vs 0.105.

## Restoration Quality

V25 improves PCM PSNR/SSIM across all three held-out restoration scenarios. On pothole, v25 is essentially tied with v24 in PSNR, with slightly higher SSIM in all three rows.

## Decision

Do not overwrite the main 640-pixel paper tables with this audit. The result is useful and clean, but it is best treated as an ablation or supplement until the full 640-pixel baseline suite is rerun with the same detector checkpoints.
