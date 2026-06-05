# Included v24 Audit Results

The latest detector-safe v24 rerun was selected using validation mAP50 and then evaluated on held-out test splits.

## Validation Selection

See:

```text
experiments/v24_task_loss_audit/selection_summary.csv
experiments/v24_task_loss_audit/checkpoints/best_by_val_map.json
```

Selected checkpoints:

| Dataset | Selection metric | Selected epoch |
|---|---|---:|
| IVCNZ-style pothole split | mean validation YOLO mAP50 over motion, defocus, lowlight | 6 |
| PCM split | mean validation YOLO mAP50 over motion, defocus, lowlight | 2 |

## Held-Out Test mAP50

| Dataset | Motion | Defocus | Lowlight |
|---|---:|---:|---:|
| IVCNZ raw restored v24 | 0.233 | 0.157 | 0.340 |
| PCM raw restored v24 | 0.303 | 0.270 | 0.414 |

## Validation-Tuned Low-Light Residual Policy

| Dataset | Raw lowlight mAP50 | Policy lowlight mAP50 |
|---|---:|---:|
| IVCNZ | 0.340 | 0.376 |
| PCM | 0.414 | 0.414 |

## Interpretation

The v24 code path is cleaner and safer, but it did not beat the strongest earlier local v22/v23 detector results. Treat this audit as evidence of a tested ablation, not as the final headline result.

