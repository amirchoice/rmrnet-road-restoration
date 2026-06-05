# RCAD-Net Experiment Protocol

## Research Motivation

Generic deblurring benchmarks usually emphasize camera shake or synthetic motion
blur. Road monitoring has a different failure profile: vehicle vibration,
forward motion, defocus, low light, compression, and small road-defect structures
all interact. A restoration model that oversmooths cracks or pothole boundaries
can score well on PSNR while hurting downstream road-damage detection.

RCAD-Net++ is designed around that gap. It restores images while explicitly
conditioning on road-relevant degradation context, predicting missing
degradation context from the image itself, and preserving high-frequency defect
cues.

## Architecture

The model is intentionally small and publishable as a practical baseline:

1. **Restoration backbone:** lightweight U-Net with depthwise separable blocks.
2. **Degradation conditioning:** 8-D code injected with FiLM at every stage.
3. **Blind code estimator:** image encoder that predicts the degradation code
   when metadata or scenario labels are unavailable.
4. **Defect attention:** image-gradient path that boosts cracks, pothole rims,
   lane markings, patches, and road texture before restoration.
5. **Losses:** L1 reconstruction, edge loss, FFT magnitude loss,
   defect-weighted reconstruction loss, and auxiliary code-estimation loss.

During synthetic training, the degradation code comes from scenario names such
as `motion_horizontal_medium`, `lowlight_strong`, or `mixed_vibration_noise`.
At inference, RCAD-Net++ can estimate the code from the image alone. Later,
`rcadnet/scenario_codes.py` can also receive real IMU/speed/exposure metadata
through `code_from_metadata(...)`.

## Training Plan

Start with the existing benchmark datasets:

- `data/metaspatial_blur_benchmark`
- `data/road_defect_700_blur_benchmark`

Recommended final training command on the Windows GPU:

```powershell
.\.venv\Scripts\python.exe train_rcadnet.py --data-root data\pothole_restoration --scenario motion_horizontal_medium --scenario defocus_medium --scenario lowlight_medium --epochs 12 --batch-size 3 --patch-size 192 --width 32 --device cuda --out runs\rcadnetpp_pothole_road_12ep --num-workers 0 --code-source fused --aux-code-weight 0.05
```

For 6 GB VRAM, reduce `--batch-size` to `1` or `2` before reducing model width.
Use all scenarios for the main model unless you want an ablation by degradation
type.

## Baseline Comparison

For a robust but feasible first paper table, compare:

- NAFNet
- Restormer
- FFTformer
- Uformer or MPRNet
- LoFormer or AdaRevD if already working in the benchmark
- DarkIR only if backend is verified
- RCAD-Net

Report all available current metrics:

- PSNR
- SSIM
- LPIPS if installed
- runtime per image
- mean runtime
- visual panels: blurred input | restored output | ground truth

Use both datasets and every scenario in the existing controller. Do not mix CPU
and GPU timings in one speed ranking.

## Runtime and Backend Reporting

Every table should include:

- model
- runtime backend: `CPU-forced`, `GPU-confirmed`, `GPU-enabled likely`,
  `GPU-intended`, or `uncertain`
- include in GPU speed ranking: `yes`, `no`, or `only if verified`
- runtime per image
- mean runtime
- peak GPU memory if available
- Jetson utilization evidence if available

On Windows:

```powershell
.\.venv\Scripts\python.exe tools\probe_runtime_backend.py --weights runs\rcadnet_road700\rcadnet_last.pth --scenario motion_horizontal_medium --height 512 --width 512 --device cuda
```

On Jetson, run the benchmark with `tegrastats` in a second terminal and record
GPU utilization:

```bash
sudo tegrastats --interval 1000 --logfile rcadnet_tegrastats.log
python scripts/benchmark_all_models.py --model RCAD-Net --dataset road_defect_700_blur_benchmark
```

## Downstream Detection Extension

After restoration metrics are stable, add a second evaluation stage:

1. Run road-defect detector on blurred input.
2. Run the same detector on restored output.
3. Compare mAP, per-class AP, and confidence changes for cracks, potholes,
   patches, and other damage classes.
4. Add a detection-guided loss only after the restoration-only model is trained.

The strongest research claim should not be only "higher PSNR"; it should be that
RCAD-Net preserves road-defect evidence under realistic mobile sensing
degradations while remaining edge-feasible.
