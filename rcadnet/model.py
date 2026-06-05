from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """Small edge-friendly convolution block used throughout RCAD-Net."""

    def __init__(self, channels: int, expansion: int = 2) -> None:
        super().__init__()
        hidden = channels * expansion
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class FiLM(nn.Module):
    """Feature-wise conditioning from degradation or IMU metadata codes."""

    def __init__(self, code_dim: int, channels: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(code_dim, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels * 2),
        )

    def forward(self, x: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.proj(code).chunk(2, dim=1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        return x * (1.0 + gamma) + beta


class RCADBlock(nn.Module):
    def __init__(self, channels: int, code_dim: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.film = FiLM(code_dim, channels)
        self.conv = DepthwiseSeparableConv(channels)

    def forward(self, x: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
        return self.conv(self.film(self.norm(x), code))


class EvidenceConditionedBlock(nn.Module):
    """Conditioned residual block with code-aware channel selection.

    The original RCAD block only shifted features with FiLM. This stronger block
    also lets the metadata/image code decide which restoration channels should
    be emphasized after local mixing, which is closer to a small hypernetwork
    while remaining edge-device friendly.
    """

    def __init__(self, channels: int, code_dim: int, expansion: int = 2) -> None:
        super().__init__()
        hidden = channels * expansion
        self.norm = nn.GroupNorm(1, channels)
        self.film = FiLM(code_dim, channels)
        self.local = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
        )
        self.channel_gate = nn.Sequential(
            nn.Linear(channels + code_dim, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
        mixed = self.local(self.film(self.norm(x), code))
        pooled = mixed.mean(dim=(2, 3))
        gate = self.channel_gate(torch.cat([pooled, code], dim=1))[:, :, None, None]
        return x + mixed * gate


class DefectAttention(nn.Module):
    """Highlights crack/pothole/lane-marking edges without requiring labels."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(1, channels // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels // 2, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        gray = image.mean(dim=1, keepdim=True)
        dx = F.pad(gray[:, :, :, 1:] - gray[:, :, :, :-1], (0, 1, 0, 0))
        dy = F.pad(gray[:, :, 1:, :] - gray[:, :, :-1, :], (0, 0, 0, 1))
        edge = torch.sqrt(dx.square() + dy.square() + 1e-6)
        edge = edge / (edge.amax(dim=(2, 3), keepdim=True) + 1e-6)
        attention = self.refine(edge)
        return features * (1.0 + attention)


class TaskEvidenceAttention(nn.Module):
    """Label-free visibility gate for detector-relevant road evidence.

    It combines edge strength, local contrast, dark-region evidence, and
    saturation cues. Cracks, pothole rims, patches, and lane markings are often
    thin high-frequency structures, but low-light and compression also change
    local contrast. This gate gives the backbone a richer task proxy than a
    plain gradient mask.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(channels // 2, 8)
        self.refine = nn.Sequential(
            nn.Conv2d(4, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        gray = image.mean(dim=1, keepdim=True)
        dx = F.pad(gray[:, :, :, 1:] - gray[:, :, :, :-1], (0, 1, 0, 0))
        dy = F.pad(gray[:, :, 1:, :] - gray[:, :, :-1, :], (0, 0, 0, 1))
        edge = torch.sqrt(dx.square() + dy.square() + 1e-6)
        blur_pool = F.avg_pool2d(gray, kernel_size=9, stride=1, padding=4)
        contrast = torch.abs(gray - blur_pool)
        dark = (1.0 - gray).clamp(0.0, 1.0)
        saturation = image.amax(dim=1, keepdim=True) - image.amin(dim=1, keepdim=True)
        cues = torch.cat([edge, contrast, dark, saturation], dim=1)
        cues = cues / (cues.amax(dim=(2, 3), keepdim=True) + 1e-6)
        attention = self.refine(cues)
        return features * (1.0 + attention)


class IdentityDefectAttention(nn.Module):
    """Ablation module that keeps the backbone identical except defect gating."""

    def forward(self, features: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        return features


class DegradationEncoder(nn.Module):
    """Predicts a soft degradation code directly from the input image.

    This keeps RCAD-Net end-to-end when scenario names, IMU, or camera metadata
    are unavailable. During synthetic training the prediction can be supervised
    by known scenario codes; with real data it can run as a blind estimator.
    """

    def __init__(self, code_dim: int, width: int) -> None:
        super().__init__()
        hidden = max(width, 16)
        self.features = nn.Sequential(
            nn.Conv2d(3, hidden, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, stride=2, padding=1, groups=hidden),
            nn.Conv2d(hidden, hidden * 2, 1),
            nn.GELU(),
            nn.Conv2d(hidden * 2, hidden * 2, 3, stride=2, padding=1, groups=hidden * 2),
            nn.Conv2d(hidden * 2, hidden * 2, 1),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, code_dim),
            nn.Sigmoid(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(image))


class CodeBasisFusion(nn.Module):
    """Reliability-gated fusion of metadata and image-estimated codes.

    A raw 8-D code is intentionally compact, but a linear FiLM projection can be
    too weak. This module expands each code with simple nonlinear basis terms and
    learns a per-dimension gate between metadata and image evidence. If metadata
    is missing or zeroed during metadata-dropout training, the gate can fall back
    to the blind image estimate.
    """

    def __init__(self, code_dim: int) -> None:
        super().__init__()
        basis_dim = code_dim * 4
        self.embed = nn.Sequential(
            nn.Linear(basis_dim, code_dim * 3),
            nn.GELU(),
            nn.Linear(code_dim * 3, code_dim),
            nn.Sigmoid(),
        )
        self.gate = nn.Sequential(
            nn.Linear(code_dim * 3 + 1, code_dim * 2),
            nn.GELU(),
            nn.Linear(code_dim * 2, code_dim),
            nn.Sigmoid(),
        )

    def _basis(self, code: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                code,
                code.square(),
                torch.sin(math.pi * code),
                torch.cos(math.pi * code),
            ],
            dim=1,
        )

    def forward(self, metadata_code: torch.Tensor, estimated_code: torch.Tensor) -> torch.Tensor:
        meta_has_signal = (metadata_code.abs().sum(dim=1, keepdim=True) > 1e-6).to(metadata_code.dtype)
        meta = self.embed(self._basis(metadata_code))
        est = self.embed(self._basis(estimated_code))
        gate = self.gate(torch.cat([meta, est, torch.abs(meta - est), meta_has_signal], dim=1))
        return torch.clamp(gate * meta + (1.0 - gate) * est, 0.0, 1.0)


class Down(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Up(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(channels, channels // 2, 1)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.proj(x) + skip


class RCADNet(nn.Module):
    """Road-Context Adaptive Defect-Preserving restoration network."""

    def __init__(
        self,
        width: int = 32,
        code_dim: int = 8,
        blocks_per_stage: int = 2,
        use_defect_attention: bool = True,
        use_estimated_code: bool = False,
        code_fusion: str = "scenario",
        block_type: str = "simple",
        attention_type: str = "edge",
        conditioning: str = "film",
        use_tdac_head: bool = False,
    ) -> None:
        super().__init__()
        self.code_dim = code_dim
        self.use_estimated_code = use_estimated_code
        self.code_fusion = code_fusion
        self.block_type = block_type
        self.attention_type = attention_type
        self.conditioning = conditioning
        self.use_tdac_head = use_tdac_head
        self.stem = nn.Conv2d(3, width, 3, padding=1)
        if not use_defect_attention or attention_type == "none":
            self.defect_attention = IdentityDefectAttention()
        elif attention_type == "task":
            self.defect_attention = TaskEvidenceAttention(width)
        else:
            self.defect_attention = DefectAttention(width)
        self.code_encoder = DegradationEncoder(code_dim, width) if use_estimated_code else None
        self.code_fuser = CodeBasisFusion(code_dim) if conditioning == "gated_basis" else None

        self.enc1 = self._make_blocks(width, code_dim, blocks_per_stage)
        self.down1 = Down(width)
        self.enc2 = self._make_blocks(width * 2, code_dim, blocks_per_stage)
        self.down2 = Down(width * 2)
        self.mid = self._make_blocks(width * 4, code_dim, blocks_per_stage + 1)
        self.up2 = Up(width * 4)
        self.dec2 = self._make_blocks(width * 2, code_dim, blocks_per_stage)
        self.up1 = Up(width * 2)
        self.dec1 = self._make_blocks(width, code_dim, blocks_per_stage)
        self.head = nn.Conv2d(width, 3, 3, padding=1)
        # Train-time auxiliary head for differentiable active-contour loss.
        # Channel order is phi_0, lambda_1, lambda_2. It is disabled by
        # default so older checkpoints and inference adapters keep the original
        # restored-image-only behavior.
        self.tdac_head = nn.Conv2d(width, 3, 3, padding=1) if use_tdac_head else None

    def _make_blocks(self, channels: int, code_dim: int, count: int) -> nn.ModuleList:
        block_cls = EvidenceConditionedBlock if self.block_type == "evidence" else RCADBlock
        return nn.ModuleList([block_cls(channels, code_dim) for _ in range(count)])

    def _run_blocks(self, x: torch.Tensor, code: torch.Tensor, blocks: nn.ModuleList) -> torch.Tensor:
        for block in blocks:
            x = block(x, code)
        return x

    def estimate_code(self, image: torch.Tensor) -> torch.Tensor:
        if self.code_encoder is None:
            return torch.zeros(image.shape[0], self.code_dim, device=image.device, dtype=image.dtype)
        return self.code_encoder(image)

    def _prepare_code(self, image: torch.Tensor, code: torch.Tensor | None) -> torch.Tensor:
        estimated = self.estimate_code(image) if self.use_estimated_code else None
        if code is None:
            if estimated is not None:
                return estimated
            return torch.zeros(image.shape[0], self.code_dim, device=image.device, dtype=image.dtype)

        if code.ndim == 1:
            code = code[None, :].expand(image.shape[0], -1)
        code = code.to(device=image.device, dtype=image.dtype)
        if self.code_fuser is not None:
            if estimated is None:
                estimated = torch.zeros_like(code)
            return self.code_fuser(code, estimated)
        if estimated is None or self.code_fusion in {"scenario", "metadata"}:
            return code
        if self.code_fusion == "estimated":
            return estimated
        return torch.clamp(0.5 * code + 0.5 * estimated, 0.0, 1.0)

    def _decode(self, image: torch.Tensor, code: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        x1 = self.defect_attention(self.stem(image), image)
        x1 = self._run_blocks(x1, code, self.enc1)
        x2 = self._run_blocks(self.down1(x1), code, self.enc2)
        x3 = self._run_blocks(self.down2(x2), code, self.mid)
        y = self._run_blocks(self.up2(x3, x2), code, self.dec2)
        y = self._run_blocks(self.up1(y, x1), code, self.dec1)
        residual = self.head(y)
        restored = torch.clamp(image + residual, 0.0, 1.0)
        aux = self.tdac_head(y) if self.tdac_head is not None else None
        return restored, aux

    @staticmethod
    def unpack_tdac_aux(aux: torch.Tensor | None) -> dict[str, torch.Tensor | None]:
        """Convert raw TDAC channels to named differentiable maps.

        The active-contour loss consumes an unconstrained level-set map and
        positive spatial energy maps. We expose both through the model result
        dictionary so training code and downstream probes do not need to know
        the packed channel convention.
        """

        if aux is None:
            return {"phi": None, "lambda1": None, "lambda2": None}
        if aux.shape[1] < 3:
            raise ValueError("TDAC auxiliary head must output phi, lambda1 and lambda2 channels")
        return {
            "phi": torch.tanh(aux[:, 0:1]),
            # Bounded spatial energies avoid active-contour collapse or
            # exploding region costs during task-driven fine-tuning.
            "lambda1": torch.sigmoid(aux[:, 1:2]) * 4.9 + 0.1,
            "lambda2": torch.sigmoid(aux[:, 2:3]) * 4.9 + 0.1,
        }

    def forward(
        self,
        image: torch.Tensor,
        code: torch.Tensor | None = None,
        *,
        return_aux: bool = False,
        return_dict: bool = False,
        return_tuple: bool = False,
        gate_threshold: float | None = None,
        gate_softness: float = 0.03,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor] | dict[str, torch.Tensor | None]:
        code = self._prepare_code(image, code)
        if gate_threshold is not None and gate_softness <= 0 and not return_aux:
            severity = code[:, -1].clamp(0.0, 1.0)
            if bool((severity < gate_threshold).all().detach().cpu()):
                return image
        restored, aux = self._decode(image, code)
        gate = None
        if gate_threshold is not None:
            severity = code[:, -1].clamp(0.0, 1.0)
            if gate_softness <= 0:
                gate = (severity >= gate_threshold).to(image.dtype)
            else:
                gate = torch.sigmoid((severity - gate_threshold) / gate_softness).to(image.dtype)
            gate = gate[:, None, None, None]
            restored = image + gate * (restored - image)
            restored = torch.clamp(restored, 0.0, 1.0)
        if not return_aux and not return_dict and not return_tuple:
            return restored
        tdac_maps = self.unpack_tdac_aux(aux)
        if return_tuple:
            return restored, tdac_maps["phi"], tdac_maps["lambda1"], tdac_maps["lambda2"], code[:, -1]
        return {
            "restored": restored,
            "aux": aux,
            "phi": tdac_maps["phi"],
            "lambda1": tdac_maps["lambda1"],
            "lambda2": tdac_maps["lambda2"],
            "code": code,
            "severity": code[:, -1],
            "z_severity": code[:, -1],
            "gate": gate,
        }
