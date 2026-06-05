# Data Preparation

This repository does not include full datasets, generated degraded splits, restored images, or YOLO checkpoints. Those files are large and may have separate licenses.

## Restoration Dataset Layout

Training and restoration evaluation expect paired degraded/clean images:

```text
data/<dataset_name>/scenarios/<scenario>/input/
data/<dataset_name>/scenarios/<scenario>/gt/
```

Example scenarios used in the v24 rerun:

```text
motion_horizontal_medium
defocus_medium
lowlight_medium
```

Example local datasets used during development:

```text
data/pothole_restoration
data/pothole_restoration_val
data/pothole_restoration_test
data/pcm_restoration_train
data/pcm_restoration_test
data/synthetic_road_smoke
```

## YOLO Dataset Layout

YOLO detection evaluation expects:

```text
datasets/<dataset_name>/data.yaml
datasets/<dataset_name>/images/train/*.jpg
datasets/<dataset_name>/images/val/*.jpg
datasets/<dataset_name>/images/test/*.jpg
datasets/<dataset_name>/labels/train/*.txt
datasets/<dataset_name>/labels/val/*.txt
datasets/<dataset_name>/labels/test/*.txt
```

## Preparing Road-Defect YOLO Splits

Use the helper scripts as templates:

```powershell
python tools/prepare_pothole_yolo.py --help
python tools/prepare_pcm_yolo.py --help
```

Then synthesize restoration pairs from YOLO images:

```powershell
python tools/make_restoration_from_yolo.py --help
python tools/make_degraded_yolo_split.py --help
```

## Restoring YOLO Splits

```powershell
python tools/restore_yolo_split.py `
  --checkpoint runs/rmrnet_v24_detector_safe_ivcnz/rcadnet_epoch006.pth `
  --input-data datasets/pothole_yolo_lowlight_test/data.yaml `
  --output-root datasets/pothole_yolo_lowlight_test_rmrnet_v24_epoch006 `
  --split test `
  --device cuda
```

## External Data Suggestions

Use public road-defect datasets such as pothole/crack/road-damage YOLO datasets, plus synthetic road-relevant degradations. For real metadata experiments, pair frames with telemetry such as speed, exposure, IMU/gyro/accelerometer estimates, or scenario-derived synthetic metadata when real telemetry is absent.

Always keep train, validation, and test frames disjoint.

