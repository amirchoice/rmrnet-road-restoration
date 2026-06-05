# v24 Rerun Commands

Environment: `.\.venv\Scripts\python.exe` on Windows PowerShell, CUDA device `NVIDIA GeForce RTX 3050` verified through PyTorch/Ultralytics logs.

Main outputs:
- IVCNZ training: `runs/rmrnet_v24_detector_safe_ivcnz`
- PCM training: `runs/rmrnet_v24_detector_safe_pcm`
- Validation detection: `runs/detection_eval_v24_val`, `runs/detection_eval_v24_pcm_val`
- Held-out test detection: `runs/detection_eval_v24_test`, `runs/detection_eval_v24_pcm_test`
- Residual policy: `runs/residual_policy_v24`

Selection rule: choose epoch by mean validation YOLO mAP50 across motion, defocus, and lowlight. Test mAP is reported only after that selection.

Selected checkpoints:
- IVCNZ: epoch 006 by validation mAP50
- PCM: epoch 002 by validation mAP50
