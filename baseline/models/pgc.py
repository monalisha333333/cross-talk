import logging
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder.residual_stream import ResidualStream
from .encoder.rgb_stream import RgbStream
from .lora.lora import apply_lora_to_linear_layers, get_lora_params
from .pgcm.peak_calibration import PeakGuidedCalibrationModule


logger = logging.getLogger(__name__)


# Default LoRA target modules inside a DINO ViT block.
_DEFAULT_LORA_TARGETS = ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"]

# Channel dimension of the residual stream output (matches paper).
_RESIDUAL_DIM = 256


class PGCNetwork(nn.Module):


    def __init__(
        self,
        opt: argparse.Namespace,
        dino_variant: str,
        num_classes: int = 1,
        lora_rank: int = 8,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.1,
        lora_targets=None,
        pretrained_root=None,
        tau_rgb: float = 0.5,
        tau_res: float = 0.5,
    ):
        super().__init__()
        self.opt=opt

        # --- (a) Feature Encoding ---
        self.rgb_stream = RgbStream(dino_variant, pretrained_root)
        self.residual_stream = ResidualStream(out_channels=_RESIDUAL_DIM)

        # --- (b) PGCM ---
        self.pgcm = PeakGuidedCalibrationModule(
            rgb_dim=self.rgb_stream.embed_dim,
            residual_dim=_RESIDUAL_DIM,
            tau_rgb=tau_rgb,
            tau_res=tau_res,
        )

        # --- (c) Global classifier head ---
        if self.opt.only_residual:
            in_dim = _RESIDUAL_DIM
        elif self.opt.only_rgb:
            in_dim = self.rgb_stream.embed_dim
        else:
            in_dim = self.rgb_stream.embed_dim + _RESIDUAL_DIM
        
        self.classifier = nn.Linear(in_dim, num_classes)
        nn.init.normal_(self.classifier.weight, 0.0, 0.02)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

        # --- LoRA injection inside the ViT backbone ---
        targets = _DEFAULT_LORA_TARGETS if lora_targets is None else list(lora_targets)
        logger.info(
            "[PGCNetwork] Injecting LoRA (rank=%d, alpha=%.3f, dropout=%.3f, "
            "targets=%s) into RGB backbone.",
            lora_rank, lora_alpha, lora_dropout, targets,
        )
        if self.opt.lora:
            # Patch matching nn.Linear layers in-place.
            self.rgb_stream.backbone = apply_lora_to_linear_layers(
                self.rgb_stream.backbone,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
                target_modules=targets,
                trainable_orig=False,
            )

        # Freeze every backbone parameter, then re-enable ONLY the LoRA params.
        # Mirrors the old project: prevents accidental partial unfreezing that
        # leads to overfitting.
        for p in self.rgb_stream.backbone.parameters():
            p.requires_grad = False
        
        if self.opt.lora:
            for p in get_lora_params(self.rgb_stream.backbone):
                p.requires_grad = True

        # Persist trainable hyperparameters for downstream tooling / logging.
        self.dino_variant = dino_variant
        if self.opt.lora:
            self.lora_rank = lora_rank
            self.lora_alpha = lora_alpha
            self.lora_dropout = lora_dropout
            self.lora_targets = targets

    # ------------------------------------------------------------------ #
    # Forward                                                             #
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor, return_feature: bool = False):

        if not self.opt.only_residual:
            rgb = x[:, :3, :, :]        # [B, 3, H, W]
            # --- RGB stream ---
            f_rgb_cls, f_rgb_tokens = self.rgb_stream(rgb)              # [B,D],[B,N,D]

        if not self.opt.only_rgb:
            residual = x[:, 3:, :, :]   # [B, 3, H, W]

            # --- Residual stream ---
            f_residual_map = self.residual_stream(residual)             # [B,C,Hr,Wr]
            f_res_pooled = torch.flatten(
                F.adaptive_avg_pool2d(f_residual_map, 1), 1
            )                                                            # [B, C]

        if self.opt.pgc or self.opt.only_pgc:
            # --- Local calibration bias from PGCM ---
            z_local = self.pgcm(f_rgb_tokens, f_residual_map)            # [B, 1]
        elif self.opt.only_residual:
            z_local = f_residual_map.new_zeros(f_residual_map.size(0),1)
        else:
            z_local = f_rgb_tokens.new_zeros(f_rgb_tokens.size(0),1)

        # --- Global feature + classifier (Eq. 5 LHS) ---
        if self.opt.only_residual:
            f_global = f_res_pooled
        elif self.opt.only_rgb:
            f_global = f_rgb_cls
        elif self.opt.RFD_separate:
            print("shape of rgb:",f_rgb_cls.shape, " Shape of res:", f_res_pooled.shape)
            exit()
            f_global = torch.cat([f_rgb_cls, f_res_pooled], dim=1)       # [B, D+C]
        else:
            f_global = torch.cat([f_rgb_cls, f_res_pooled], dim=1)       # [B, D+C]
        # L2-normalize to the unit hypersphere; with small-norm classifier
        # weights this keeps Z_global tight around 0 so that Z_local can
        # decisively shift the decision (paper Sec. 3.3 design intent).
        f_global_normed = F.normalize(f_global, p=2, dim=1)
        z_global = self.classifier(f_global_normed)                  # [B, num_classes]

        if self.opt.only_pgc:
            y_pred = z_local
        else:    
            # --- Final additive calibration (paper Eq. 5) ---
            y_pred = z_global + z_local
            
        if return_feature:
            return f_global, y_pred, z_local, z_global
        return y_pred

    # ------------------------------------------------------------------ #
    # Trainable parameter collection                                      #
    # ------------------------------------------------------------------ #

    def get_trainable_params(self):

        seen = set()
        out = []

        def _add(params):
            for p in params:
                if id(p) in seen:
                    continue
                seen.add(id(p))
                out.append(p)

        if self.opt.lora:
            _add(get_lora_params(self.rgb_stream.backbone))
        _add(self.residual_stream.parameters())
        _add(self.classifier.parameters())
        _add(self.pgcm.parameters())  # includes score heads + lambda_rgb

        return out
