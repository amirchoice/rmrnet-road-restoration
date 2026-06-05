# Baseline Review Matrix

This file records the baseline policy for the road-restoration paper so the
experimental setup is defensible and easy to extend.

## Executed Baselines In This Workspace

| method | role | status | why included |
|---|---|---|---|
| Degraded input | no-restoration lower bound | executed | Shows detector collapse caused by degradation. |
| RCAD-Net | ablation of proposed method | executed | Scenario-code conditioning + defect attention, no blind code estimator. |
| RCAD-Net++ | proposed method | executed | Adds blind image-derived degradation-code estimator and fused conditioning. |
| DFPIR | strong recent all-in-one restoration baseline | executed | CVPR 2025 method with official code and weights; covers deblurring and low-light restoration. |

## Recommended Strong Baselines To Add Before Submission

| method | venue | task coverage | code/weights availability | priority |
|---|---|---|---|---|
| Restormer | CVPR 2022 Oral | motion blur, defocus blur, denoising, deraining | official repo and pretrained models | high |
| NAFNet | ECCV 2022 | efficient image deblurring and denoising | official repo and pretrained models | high |
| MPRNet | CVPR 2021 | deblurring, deraining, denoising | official repo; older dependency stack | medium |
| FFTformer | CVPR 2023 | high-quality motion deblurring | official repo and RealBlur pretrained model | high for motion-blur table |
| DarkIR | CVPR 2025 | low-light, noise, blur restoration | official repo and weights | high for low-light table |
| InstructIR | ECCV 2024 | prompt-guided all-in-one restoration | repo, HF/demo weights, benchmark data | optional; useful all-in-one comparison |

## Dataset Policy

The current local experiments use:

| dataset | role | images | labels | status |
|---|---|---:|---|---|
| IVCNZ pothole YOLO dataset | main road-defect detection and restoration dataset | 1,243 | YOLO pothole boxes | executed |
| Kodak-24 restoration benchmark | academic natural-image sanity check | 24 | no boxes | executed as appendix |

Recommended before submission:

| dataset | role | reason |
|---|---|---|
| RDD2022 | road-damage detection generalization | multi-national, larger road damage dataset with multiple classes. |
| UAV-PDD2023 | viewpoint generalization | UAV pavement distress images; useful to show the method is not only dashcam-like. |
| GoPro / RealBlur | standard motion-deblurring sanity check | Lets restoration reviewers compare to common deblurring literature. |
| DPDD | standard defocus sanity check | Useful for defocus-specific reviewers. |
| LOLBlur / LOL / LSRW | low-light restoration sanity check | Relevant because road monitoring often has night/low-light blur. |

## Reviewer-Safe Claim

Do not claim universal restoration dominance. The evidence supports:

> RCAD-Net++ is a task-driven road restoration model that consistently improves
> downstream pothole detection under motion blur, defocus, and low light. It is
> competitive with DFPIR on motion detection recovery, stronger on defocus and
> low-light detection recovery, and much faster on the tested RTX 3050 setup.

DFPIR remains stronger on full-reference motion/defocus PSNR and SSIM, so the
paper should explicitly frame the contribution around road-defect evidence
preservation and edge-feasible downstream utility.
