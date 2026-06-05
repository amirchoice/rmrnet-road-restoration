$ErrorActionPreference = "Stop"

# Restore and evaluate the validation-selected v24 checkpoints.
# This script assumes YOLO data.yaml files already exist under datasets/.

python tools\eval_yolo_suite.py `
  --weights runs\detect\runs\detect_train\yolov8n_pothole_clean\weights\best.pt `
  --data datasets\pothole_yolo_motion_test_rmrnet_v24_epoch006\data.yaml `
  --name motion `
  --data datasets\pothole_yolo_defocus_test_rmrnet_v24_epoch006\data.yaml `
  --name defocus `
  --data datasets\pothole_yolo_lowlight_test_rmrnet_v24_epoch006\data.yaml `
  --name lowlight `
  --out-csv runs\detection_eval_v24_test\pothole_v24_epoch006_test.csv `
  --imgsz 640 `
  --batch 8 `
  --device 0 `
  --workers 0

python tools\eval_yolo_suite.py `
  --weights runs\detect\runs\detect_train\yolov8n_pcm_clean_25ep\weights\best.pt `
  --data datasets\pcm_yolo_motion_test_rmrnet_v24_epoch002\data.yaml `
  --name motion `
  --data datasets\pcm_yolo_defocus_test_rmrnet_v24_epoch002\data.yaml `
  --name defocus `
  --data datasets\pcm_yolo_lowlight_test_rmrnet_v24_epoch002\data.yaml `
  --name lowlight `
  --out-csv runs\detection_eval_v24_pcm_test\pcm_v24_epoch002_test.csv `
  --imgsz 640 `
  --batch 8 `
  --device 0 `
  --workers 0

