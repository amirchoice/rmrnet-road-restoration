$ErrorActionPreference = "Stop"

python train_rcadnet.py `
  --data-root data\synthetic_road_smoke `
  --scenario motion_horizontal_medium `
  --epochs 1 `
  --batch-size 1 `
  --patch-size 64 `
  --lr 1e-4 `
  --width 8 `
  --device cuda `
  --out runs\rmrnet_v24_detector_safe_smoke `
  --num-workers 0 `
  --code-source metadata_fused `
  --block-type evidence `
  --attention-type task `
  --conditioning gated_basis `
  --metadata-noise 0.01 `
  --metadata-dropout 0.10 `
  --edge-weight 0.05 `
  --freq-weight 0.02 `
  --defect-weight 0.05 `
  --use-task-losses `
  --lambda-tdp 0.002 `
  --lambda-jacobian 0.00002 `
  --lambda-active-contour 0.005 `
  --lambda-detector-input-anchor 0.0005 `
  --lambda-evidence-nonregression 0.02 `
  --task-loss-warmup-epochs 1 `
  --tdp-yolo-weights runs\detect\runs\detect_train\yolov8n_pothole_clean\weights\best.pt `
  --tdp-layers 2,4 `
  --tdp-layer-weights 0.5,1.0 `
  --detector-input-size 320 `
  --jacobian-probes 1 `
  --cqmix-grid 4 `
  --debug-first-batches 0 `
  --smoke-test

