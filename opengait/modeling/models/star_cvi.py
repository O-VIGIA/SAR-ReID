import os
import math
import json
import inspect
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf

warnings.filterwarnings("ignore", message="The PyTorch API of nested tensors is in prototype stage")

from ..base_model import BaseModel
from ..modules import SeparateFCs, SeparateBNNecks
from ..model_clip.make_model_clipreid import make_model
from .star_cvi_texts import build_semantic_dictionary

# Register STAR-CVI's additive OpenGait extensions without overwriting upstream
# transform.py or evaluator.py. OpenGait imports every model module before it
# constructs the data loaders, so these names are available when configs are read.
from data import transform as _opengait_transform
from data.star_cvi_transform import TRANSFORM_REGISTRY
from evaluation import evaluator as _opengait_evaluator
from evaluation.star_cvi_evaluator import EVALUATOR_REGISTRY

for _name, _component in TRANSFORM_REGISTRY.items():
    setattr(_opengait_transform, _name, _component)
for _name, _component in EVALUATOR_REGISTRY.items():
    setattr(_opengait_evaluator, _name, _component)

_CLIP_CFG_PATH = Path(__file__).resolve().parents[1] / "model_clip" / "config_clip" / "cfg.yaml"
clip_cfg = OmegaConf.load(_CLIP_CFG_PATH)


# =========================================================================================
# Utility modules
# =========================================================================================
class TextFeatureAdapter(nn.Module):
    """Light residual adapter that moves CLIP text prototypes toward the AG-ReID domain."""

    def __init__(self, dim=512, hidden_dim=128, dropout=0.1, init_scale=1e-3, use_layernorm=True):
        super().__init__()
        self.norm = nn.LayerNorm(dim) if use_layernorm else nn.Identity()
        self.down_proj = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up_proj = nn.Linear(hidden_dim, dim)
        self.scale = nn.Parameter(torch.ones(1) * float(init_scale))

    def forward(self, x):
        h = self.up_proj(self.dropout(self.act(self.down_proj(self.norm(x)))))
        return F.normalize(x + self.scale * h, p=2, dim=-1)


class MaskedSequencePooling(nn.Module):
    """Temporal pooling with seqL support."""

    def __init__(self, pool_type="mean"):
        super().__init__()
        if pool_type not in ["mean", "max"]:
            raise ValueError(f"Unsupported pool_type: {pool_type}")
        self.pool_type = pool_type

    @staticmethod
    def _build_mask(seqL, T, device, B):
        if seqL is None:
            return None
        seqL = seqL.to(device) if torch.is_tensor(seqL) else torch.tensor(seqL, device=device)
        seqL = seqL.squeeze(-1).long() if seqL.dim() > 1 else seqL.long()
        return torch.arange(T, device=device).unsqueeze(0).expand(B, T) < seqL.unsqueeze(1)

    def forward(self, x, seqL=None):
        B, T, D = x.shape
        mask = self._build_mask(seqL, T, x.device, B)
        if mask is None:
            return x.mean(dim=1) if self.pool_type == "mean" else x.max(dim=1)[0]

        if self.pool_type == "mean":
            mask_f = mask.unsqueeze(-1).float()
            denom = mask_f.sum(dim=1).clamp(min=1e-6)
            return (x * mask_f).sum(dim=1) / denom

        x_masked = x.masked_fill(~mask.unsqueeze(-1), -1e9)
        return x_masked.max(dim=1)[0].clamp(min=-1e4, max=1e4)


class LatentViewBranch(nn.Module):
    """Implicit view/capture-condition encoder used by saliency and context calibration."""

    def __init__(self, dim=512, hidden_dim=256, dropout=0.1, use_layernorm=True):
        super().__init__()
        self.norm = nn.LayerNorm(dim) if use_layernorm else nn.Identity()
        self.shared = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.view_head = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return F.normalize(self.view_head(self.shared(self.norm(x))), p=2, dim=-1)


class IdentityTextCalibration(nn.Module):
    """No-op text calibration with the same call signature as router calibration blocks."""

    def forward(self, query, key_value=None):
        return query


class AxisConditionedCLSTextCalibration(nn.Module):
    """Axis-specific CLS calibration for text queries.

    The previous single-key cross-attention sends the same CLS-derived residual to every
    semantic axis. This module keeps frame-level visual adaptation, but conditions the
    residual direction and gate on each text axis, so calibration does not erase axis
    differences before text-to-visual routing.
    """

    def __init__(
        self,
        embed_dim=512,
        hidden_dim=256,
        dropout=0.0,
        init_scale=1e-3,
        max_scale=0.20,
        use_layernorm=True,
    ):
        super().__init__()
        self.max_scale = float(max_scale)
        self.text_norm = nn.LayerNorm(embed_dim) if use_layernorm else nn.Identity()
        self.cls_norm = nn.LayerNorm(embed_dim) if use_layernorm else nn.Identity()
        in_dim = embed_dim * 3
        self.delta_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        init_scale = min(max(float(init_scale), 1e-6), self.max_scale - 1e-6)
        init_ratio = init_scale / self.max_scale
        self.scale_logit = nn.Parameter(torch.tensor(math.log(init_ratio / (1.0 - init_ratio)), dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self):
        # Start close to identity; the model can learn visual adaptation when useful.
        nn.init.zeros_(self.delta_mlp[-1].weight)
        nn.init.zeros_(self.delta_mlp[-1].bias)
        nn.init.zeros_(self.gate_mlp[-1].weight)
        nn.init.zeros_(self.gate_mlp[-1].bias)

    def forward(self, query, key_value):
        if key_value is None:
            return query
        cls = key_value[:, :1, :].expand(-1, query.size(1), -1)
        qn = self.text_norm(query)
        cn = self.cls_norm(cls)
        h = torch.cat([qn, cn, qn * cn], dim=-1)
        delta = self.delta_mlp(h)
        gate = torch.sigmoid(self.gate_mlp(h))
        scale = self.max_scale * torch.sigmoid(self.scale_logit)
        return query + scale * gate * delta


class CrossAttentionFFNBlock(nn.Module):
    """Cross-attention residual block without extra bias."""

    def __init__(self, embed_dim=512, num_heads=8, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.norm_ffn = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=attn_drop, batch_first=True)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.GELU(), nn.Dropout(proj_drop),
            nn.Linear(hidden_dim, embed_dim), nn.Dropout(proj_drop)
        )

    def forward(self, query, key_value):
        attn_out, _ = self.cross_attn(self.norm_q(query), self.norm_kv(key_value), self.norm_kv(key_value), need_weights=False)
        x = query + attn_out
        return x + self.ffn(self.norm_ffn(x))


class CrossAttentionFFNBlockWithBias(nn.Module):
    """Cross-attention block with optional patch-logit bias, attention return, and axis-wise competition."""

    def __init__(self, embed_dim=512, num_heads=8, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}")
        self.num_heads = int(num_heads)
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.norm_ffn = nn.LayerNorm(embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.GELU(), nn.Dropout(proj_drop),
            nn.Linear(hidden_dim, embed_dim), nn.Dropout(proj_drop)
        )

    @staticmethod
    def _apply_axis_competition(attn, logits, cfg):
        """Blend patch-softmax attention with patch-to-axis competition inside each role.

        attn/logits: [B,H,M,N], where M is joint text axes and N is visual patches.
        The ordinary T2V softmax is over patches. Axis competition adds a second assignment:
        for each patch, semantic axes inside the same role compete for explanation ownership.
        """
        if cfg is None or not bool(cfg.get("enable", False)):
            return attn
        Mi = int(cfg.get("num_id_axes", 0))
        Mc = int(cfg.get("num_context_axes", max(0, attn.shape[-2] - Mi)))
        if Mi + Mc != attn.shape[-2]:
            return attn

        pieces = []
        offset = 0
        for role, M in [("id", Mi), ("ctx", Mc)]:
            if M <= 0:
                continue
            role_attn = attn[..., offset:offset + M, :]
            role_logits = logits[..., offset:offset + M, :]
            tau = float(cfg.get(f"temperature_{role}", 0.7 if role == "id" else 0.5))
            mix = float(cfg.get(f"mix_ratio_{role}", 0.15 if role == "id" else 0.25))
            tau = max(tau, 1e-4)
            mix = min(max(mix, 0.0), 1.0)
            if mix > 0:
                # For each visual patch, distribute ownership over axes in the same role.
                axis_prob = torch.softmax(role_logits.transpose(-2, -1) / tau, dim=-1).transpose(-2, -1)
                if bool(cfg.get("detach_axis_prob", False)):
                    axis_prob = axis_prob.detach()
                comp = role_attn * axis_prob
                comp = comp / comp.sum(dim=-1, keepdim=True).clamp(min=1e-6)
                role_attn = (1.0 - mix) * role_attn + mix * comp
                role_attn = role_attn / role_attn.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            pieces.append(role_attn)
            offset += M
        return torch.cat(pieces, dim=-2) if pieces else attn

    def forward(self, query, key_value, attn_bias=None, return_attn=False, detach_attn=True, axis_competition=None):
        B, Lq, D = query.shape
        _, Lk, _ = key_value.shape
        q = self.q_proj(self.norm_q(query)).view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(self.norm_kv(key_value)).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(self.norm_kv(key_value)).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)

        query_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attn_bias is not None:
            if attn_bias.dim() == 2:          # [B, Lk]
                bias = attn_bias[:, None, None, :]
            elif attn_bias.dim() == 3:        # [B, Lq, Lk]
                bias = attn_bias[:, None, :, :]
            else:
                raise ValueError(f"Unsupported attn_bias shape: {tuple(attn_bias.shape)}")
            biased_logits = query_logits + bias
        else:
            biased_logits = query_logits

        query_attn = torch.softmax(query_logits, dim=-1)
        biased_attn = torch.softmax(biased_logits, dim=-1)
        biased_attn = self._apply_axis_competition(biased_attn, biased_logits, axis_competition)
        attn = self.attn_drop(biased_attn)
        attn_out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, Lq, D)
        x = query + self.out_proj(attn_out)
        x = x + self.ffn(self.norm_ffn(x))

        if return_attn:
            if detach_attn:
                query_attn_out = query_attn.detach()
                biased_attn_out = biased_attn.detach()
            else:
                query_attn_out = query_attn
                biased_attn_out = biased_attn
            aux = {
                "query_attn": query_attn_out,
                "biased_attn": biased_attn_out,
            }
            return x, aux
        return x


class ViewConditionedTokenSaliencyBias(nn.Module):
    """Branch-specific shared patch prior conditioned on frame CLS and video-level z_view."""

    def __init__(self, embed_dim=512, hidden_dim=256, dropout=0.1, init_bias_scale=0.25):
        super().__init__()
        self.patch_norm = nn.LayerNorm(embed_dim)
        self.cls_norm = nn.LayerNorm(embed_dim)
        self.view_norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1)
        )
        self.bias_scale = nn.Parameter(torch.ones(1) * float(init_bias_scale))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, cls_tokens, patch_tokens, z_view, seqL=None):
        B, T, N, D = patch_tokens.shape
        cls_seq = cls_tokens.squeeze(2) if cls_tokens.dim() == 4 else cls_tokens
        feat = torch.cat([
            self.patch_norm(patch_tokens),
            self.cls_norm(cls_seq).unsqueeze(2).expand(B, T, N, D),
            self.view_norm(z_view).unsqueeze(1).unsqueeze(2).expand(B, T, N, D),
        ], dim=-1)
        attn_bias = self.bias_scale * torch.tanh(self.mlp(feat).squeeze(-1))
        if seqL is not None:
            seq_t = seqL.to(patch_tokens.device) if torch.is_tensor(seqL) else torch.tensor(seqL, device=patch_tokens.device)
            seq_t = seq_t.view(-1).long()
            frame_mask = torch.arange(T, device=patch_tokens.device).unsqueeze(0).expand(B, T) < seq_t.unsqueeze(1)
            attn_bias = attn_bias * frame_mask.unsqueeze(-1).float()
        return attn_bias


# =========================================================================================
# ID/Context router and calibration
# =========================================================================================
class JointIDContextSemanticRouter(nn.Module):
    """Joint ID/CTX router with visual-to-semantic role competition and axis-specialized T2V.

    Compared with the previous separated ID/CTX router, this module concatenates ID and CTX
    text axes inside the router. Patches compete over joint semantic axes in Visual2Text,
    while T2V can additionally use axis-wise competition inside each role to avoid all axes
    focusing on the same region.
    """

    def __init__(
        self,
        embed_dim=512,
        num_heads=8,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        competition_bias_enable=True,
        competition_bias_scale=0.35,
        competition_bias_clip=2.0,
        detach_competition=True,
        axis_competition_cfg=None,
        semantic_calibration_cfg=None,
        final_use_shared_patch_prior=True,
        axis_specific_final_bias_enable=False,
        axis_specific_final_bias_scale=0.15,
        axis_specific_final_bias_clip=1.5,
        detach_axis_specific_final_bias=True,
    ):
        super().__init__()
        self.competition_bias_enable = bool(competition_bias_enable)
        self.competition_bias_scale = float(competition_bias_scale)
        self.competition_bias_clip = float(competition_bias_clip)
        self.detach_competition = bool(detach_competition)
        self.axis_competition_cfg = dict(axis_competition_cfg or {})
        self.axis_competition_enable = bool(self.axis_competition_cfg.get("enable", False))
        self.axis_competition_apply_final = bool(self.axis_competition_cfg.get("apply_to_final", False))
        self.final_use_shared_patch_prior = bool(final_use_shared_patch_prior)
        self.axis_specific_final_bias_enable = bool(axis_specific_final_bias_enable)
        self.axis_specific_final_bias_scale = float(axis_specific_final_bias_scale)
        self.axis_specific_final_bias_clip = float(axis_specific_final_bias_clip)
        self.detach_axis_specific_final_bias = bool(detach_axis_specific_final_bias)

        semantic_calibration_cfg = dict(semantic_calibration_cfg or {})
        calibration_type = str(semantic_calibration_cfg.get("type", "axis_conditioned_cls")).lower()
        if calibration_type in {"none", "identity", "off"}:
            self.semantic_recalibration = IdentityTextCalibration()
        elif calibration_type in {"legacy_cross_attention", "cross_attention", "mha"}:
            self.semantic_recalibration = CrossAttentionFFNBlock(embed_dim, num_heads, mlp_ratio, attn_drop, proj_drop)
        elif calibration_type in {"axis_conditioned", "axis_conditioned_cls", "axis_cls"}:
            self.semantic_recalibration = AxisConditionedCLSTextCalibration(
                embed_dim=embed_dim,
                hidden_dim=semantic_calibration_cfg.get("hidden_dim", 256),
                dropout=semantic_calibration_cfg.get("dropout", 0.0),
                init_scale=semantic_calibration_cfg.get("init_scale", 1e-3),
                max_scale=semantic_calibration_cfg.get("max_scale", 0.20),
                use_layernorm=semantic_calibration_cfg.get("use_layernorm", True),
            )
        else:
            raise ValueError(f"Unsupported semantic calibration type: {calibration_type}")

        self.text_to_visual = CrossAttentionFFNBlockWithBias(embed_dim, num_heads, mlp_ratio, attn_drop, proj_drop)
        self.visual_to_text = CrossAttentionFFNBlockWithBias(embed_dim, num_heads, mlp_ratio, attn_drop, proj_drop)
        self.joint_final_refine = CrossAttentionFFNBlockWithBias(embed_dim, num_heads, mlp_ratio, attn_drop, proj_drop)

    @staticmethod
    def _expand_text(text_feats, B, T, D, device, dtype):
        text_feats = text_feats.to(device=device, dtype=dtype)
        if text_feats.dim() == 2:
            return text_feats.unsqueeze(0).expand(B * T, -1, -1).contiguous()
        if text_feats.dim() == 3:
            if text_feats.shape[0] == B:
                return text_feats.unsqueeze(1).expand(B, T, -1, -1).reshape(B * T, text_feats.shape[1], D).contiguous()
            if text_feats.shape[0] == B * T:
                return text_feats.contiguous()
        raise ValueError(f"Unsupported text feature shape: {tuple(text_feats.shape)}")

    @staticmethod
    def _reshape_patch_bias(bias, B, T, N, device, dtype):
        if bias is None:
            return None
        bias = bias.to(device=device, dtype=dtype)
        if bias.dim() == 3:
            assert bias.shape == (B, T, N), f"bias shape {tuple(bias.shape)} != {(B, T, N)}"
            return bias.reshape(B * T, N).contiguous()
        if bias.dim() == 2 and bias.shape[0] == B * T:
            assert bias.shape[1] == N
            return bias.contiguous()
        raise ValueError(f"Unsupported patch bias shape: {tuple(bias.shape)}")

    @staticmethod
    def _pack_joint_attn(aux, B, T, Mi):
        if aux is None:
            return None, None
        out = {}
        for k, v in aux.items():
            v = v.mean(dim=1).view(B, T, v.shape[-2], v.shape[-1]).detach()
            out[k] = v
        id_out = {k: v[:, :, :Mi, :] for k, v in out.items()}
        ctx_out = {k: v[:, :, Mi:, :] for k, v in out.items()}
        return id_out, ctx_out

    @staticmethod
    def _pack_v2t_aux(aux, B, T, Mi, Mc, competition_delta=None):
        if aux is None:
            return None
        token_attn = aux["biased_attn"].mean(dim=1).view(B, T, aux["biased_attn"].shape[-2], aux["biased_attn"].shape[-1]).detach()
        id_mass = token_attn[..., :Mi].sum(dim=-1) if Mi > 0 else torch.zeros_like(token_attn[..., 0])
        ctx_mass = token_attn[..., Mi:].sum(dim=-1) if Mc > 0 else torch.zeros_like(id_mass)
        packed = {"token_attn": token_attn, "id_mass": id_mass, "ctx_mass": ctx_mass}
        if competition_delta is not None:
            packed["competition_delta"] = competition_delta.view(B, T, -1).detach()
        return packed

    @staticmethod
    def _build_joint_patch_bias(bias_id, bias_ctx, Mi, Mc):
        if bias_id is None and bias_ctx is None:
            return None
        if bias_id is None:
            bias_id = torch.zeros_like(bias_ctx)
        if bias_ctx is None:
            bias_ctx = torch.zeros_like(bias_id)
        parts = []
        if Mi > 0:
            parts.append(bias_id.unsqueeze(1).expand(-1, Mi, -1))
        if Mc > 0:
            parts.append(bias_ctx.unsqueeze(1).expand(-1, Mc, -1))
        return torch.cat(parts, dim=1) if parts else None

    def _axis_competition(self, Mi, Mc, apply_final=False):
        if not self.axis_competition_enable:
            return None
        cfg = dict(self.axis_competition_cfg)
        cfg["num_id_axes"] = Mi
        cfg["num_context_axes"] = Mc
        if apply_final and not self.axis_competition_apply_final:
            return None
        if apply_final:
            # Final refinement is more sensitive; use a conservative half-mix if enabled.
            cfg["mix_ratio_id"] = float(cfg.get("final_mix_ratio_id", 0.5 * float(cfg.get("mix_ratio_id", 0.15))))
            cfg["mix_ratio_ctx"] = float(cfg.get("final_mix_ratio_ctx", 0.5 * float(cfg.get("mix_ratio_ctx", 0.25))))
        return cfg

    def _build_competition_bias(self, v2t_aux, Mi, Mc):
        if (not self.competition_bias_enable) or v2t_aux is None or Mc <= 0:
            return None, None
        token_attn = v2t_aux["biased_attn"].mean(dim=1)  # [BT,N,Mi+Mc]
        id_mass = token_attn[..., :Mi].sum(dim=-1)
        ctx_mass = token_attn[..., Mi:].sum(dim=-1)
        delta = torch.log((id_mass + 1e-6) / (ctx_mass + 1e-6))
        delta = torch.clamp(delta, min=-self.competition_bias_clip, max=self.competition_bias_clip)
        if self.detach_competition:
            delta = delta.detach()
        id_bias = self.competition_bias_scale * delta.unsqueeze(1).expand(-1, Mi, -1)
        ctx_bias = -self.competition_bias_scale * delta.unsqueeze(1).expand(-1, Mc, -1)
        return torch.cat([id_bias, ctx_bias], dim=1), delta

    def _build_axis_specific_final_bias(self, t2v_aux, Mi, Mc):
        if (not self.axis_specific_final_bias_enable) or t2v_aux is None:
            return None
        attn = t2v_aux["biased_attn"].mean(dim=1).clamp(min=1e-6)  # [BT,M,N]
        if self.detach_axis_specific_final_bias:
            attn = attn.detach()
        log_attn = torch.log(attn)
        parts = []
        offset = 0
        for M in [Mi, Mc]:
            if M <= 0:
                continue
            role_log = log_attn[:, offset:offset + M, :]
            if M == 1:
                role_bias = torch.zeros_like(role_log)
            else:
                role_bias = role_log - role_log.mean(dim=1, keepdim=True)
            parts.append(role_bias)
            offset += M
        if not parts:
            return None
        bias = torch.cat(parts, dim=1)
        bias = torch.clamp(bias, min=-self.axis_specific_final_bias_clip, max=self.axis_specific_final_bias_clip)
        return self.axis_specific_final_bias_scale * bias

    def forward(
        self,
        id_text_feats,
        context_text_feats,
        patch_tokens,
        cls_tokens,
        patch_attn_bias_id=None,
        patch_attn_bias_context=None,
        return_visualization=False,
    ):
        B, T, N, D = patch_tokens.shape
        device, dtype = patch_tokens.device, patch_tokens.dtype
        visual_seq = patch_tokens.reshape(B * T, N, D)
        cls_seq = cls_tokens.reshape(B * T, 1, D)
        id_text = self._expand_text(id_text_feats, B, T, D, device, dtype)
        ctx_text = self._expand_text(context_text_feats, B, T, D, device, dtype)
        Mi, Mc = id_text.shape[1], ctx_text.shape[1]
        if Mi <= 0:
            raise ValueError("At least one ID text axis is required for retrieval.")

        joint_text = torch.cat([id_text, ctx_text], dim=1) if Mc > 0 else id_text
        bias_id = self._reshape_patch_bias(patch_attn_bias_id, B, T, N, device, dtype)
        bias_ctx = self._reshape_patch_bias(patch_attn_bias_context, B, T, N, device, dtype)
        joint_patch_bias = self._build_joint_patch_bias(bias_id, bias_ctx, Mi, Mc)

        joint_text_calib = self.semantic_recalibration(query=joint_text, key_value=cls_seq)

        need_t2v_aux = return_visualization or self.axis_specific_final_bias_enable
        if need_t2v_aux:
            joint_text_1, t2v_aux = self.text_to_visual(
                joint_text_calib, visual_seq, joint_patch_bias, return_attn=True,
                detach_attn=(return_visualization or self.detach_axis_specific_final_bias),
                axis_competition=self._axis_competition(Mi, Mc, apply_final=False)
            )
        else:
            t2v_aux = None
            joint_text_1 = self.text_to_visual(
                joint_text_calib, visual_seq, joint_patch_bias, return_attn=False,
                axis_competition=self._axis_competition(Mi, Mc, apply_final=False)
            )

        need_v2t_aux = return_visualization or (self.competition_bias_enable and Mc > 0)
        if need_v2t_aux:
            visual_response, v2t_aux = self.visual_to_text(
                visual_seq, joint_text_1, attn_bias=None, return_attn=True, detach_attn=self.detach_competition
            )
        else:
            v2t_aux = None
            visual_response = self.visual_to_text(visual_seq, joint_text_1, attn_bias=None, return_attn=False)

        competition_bias, competition_delta = self._build_competition_bias(v2t_aux, Mi, Mc)
        final_bias = joint_patch_bias if self.final_use_shared_patch_prior else None
        if competition_bias is not None:
            final_bias = competition_bias if final_bias is None else final_bias + competition_bias
        axis_specific_bias = self._build_axis_specific_final_bias(t2v_aux, Mi, Mc)
        if axis_specific_bias is not None:
            final_bias = axis_specific_bias if final_bias is None else final_bias + axis_specific_bias

        if return_visualization:
            joint_text_2, final_aux = self.joint_final_refine(
                joint_text_1, visual_response, final_bias, return_attn=True, detach_attn=True,
                axis_competition=self._axis_competition(Mi, Mc, apply_final=True)
            )
        else:
            final_aux = None
            joint_text_2 = self.joint_final_refine(
                joint_text_1, visual_response, final_bias, return_attn=False,
                axis_competition=self._axis_competition(Mi, Mc, apply_final=True)
            )

        routed_id = joint_text_2[:, :Mi, :].view(B, T, Mi, D)
        routed_ctx = joint_text_2[:, Mi:, :].view(B, T, Mc, D) if Mc > 0 else joint_text_2[:, :0, :].view(B, T, 0, D)

        aux = {}
        if return_visualization:
            id_t2v, ctx_t2v = self._pack_joint_attn(t2v_aux, B, T, Mi)
            id_final, ctx_final = self._pack_joint_attn(final_aux, B, T, Mi)
            aux["id"] = {"t2v": id_t2v, "final": id_final}
            aux["context"] = {"t2v": ctx_t2v, "final": ctx_final}
            packed_v2t = self._pack_v2t_aux(v2t_aux, B, T, Mi, Mc, competition_delta)
            if packed_v2t is not None:
                aux["joint"] = {"v2t": packed_v2t}
            if axis_specific_bias is not None:
                aux.setdefault("joint", {})["axis_specific_final_bias"] = axis_specific_bias.view(B, T, Mi + Mc, N).detach()
        return routed_id, routed_ctx, aux


# Backward-compatible alias: old references now build the joint router.
S2ASharedIDContextRouter = JointIDContextSemanticRouter


class ContextConditionedIDCalibration(nn.Module):
    """Use routed context tokens and z_view to conservatively calibrate routed ID tokens."""

    def __init__(
        self,
        embed_dim=512,
        num_id_axes=20,
        num_context_axes=10,
        hidden_dim=256,
        dropout=0.1,
        init_scale=0.03,
        max_scale=0.20,
        detach_condition=True,
    ):
        super().__init__()
        self.detach_condition = bool(detach_condition)
        self.max_scale = float(max_scale)
        self.id_norm = nn.LayerNorm(embed_dim)
        self.ctx_norm = nn.LayerNorm(embed_dim)
        self.view_norm = nn.LayerNorm(embed_dim)
        self.ctx_score = nn.Linear(embed_dim, 1)
        self.cond_mlp = nn.Sequential(
            nn.LayerNorm(embed_dim * 2), nn.Linear(embed_dim * 2, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim * 2)
        )
        self.gate_mlp = nn.Sequential(
            nn.LayerNorm(embed_dim * 4), nn.Linear(embed_dim * 4, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        init_scale = min(max(float(init_scale), 1e-6), self.max_scale - 1e-6)
        init_ratio = init_scale / self.max_scale
        self.scale_logit = nn.Parameter(torch.tensor(math.log(init_ratio / (1.0 - init_ratio)), dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.cond_mlp[-1].weight, std=1e-3)
        nn.init.zeros_(self.cond_mlp[-1].bias)
        nn.init.zeros_(self.gate_mlp[-1].weight)
        nn.init.zeros_(self.gate_mlp[-1].bias)
        nn.init.zeros_(self.ctx_score.bias)

    def forward(self, id_feats, ctx_feats, z_view, return_aux=False):
        B, T, Mi, D = id_feats.shape
        ctx_cond = ctx_feats.detach() if self.detach_condition else ctx_feats
        view_cond = z_view.detach() if self.detach_condition else z_view

        ctx_norm = self.ctx_norm(ctx_cond)
        ctx_weight = torch.softmax(self.ctx_score(ctx_norm), dim=2)      # [B,T,Mc,1]
        ctx_summary = (ctx_weight * ctx_norm).sum(dim=2)                # [B,T,D]
        view_summary = self.view_norm(view_cond).unsqueeze(1).expand(B, T, D)

        gamma_beta = torch.tanh(self.cond_mlp(torch.cat([ctx_summary, view_summary], dim=-1)))
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma, beta = gamma.unsqueeze(2), beta.unsqueeze(2)

        id_norm = self.id_norm(id_feats)
        ctx_expand = ctx_summary.unsqueeze(2).expand(B, T, Mi, D)
        view_expand = view_summary.unsqueeze(2).expand(B, T, Mi, D)
        gate_input = torch.cat([id_norm, ctx_expand, view_expand, id_norm * ctx_expand], dim=-1)
        gate = torch.sigmoid(self.gate_mlp(gate_input))                 # [B,T,Mi,1]
        scale = self.max_scale * torch.sigmoid(self.scale_logit)
        calibrated = id_feats + scale * gate * (gamma * id_norm + beta)

        if return_aux:
            return calibrated, {"ctx_weight": ctx_weight.detach(), "calib_gate": gate.detach(), "calib_scale": scale.detach()}
        return calibrated


class SemanticCrossFrameAttentionBlock(nn.Module):
    """Semantic-wise cross-frame set interaction. No temporal order is assumed."""

    def __init__(self, dim=512, bottleneck_dim=128, num_heads=2, depth=1, dropout=0.1):
        super().__init__()
        self.norm_in = nn.LayerNorm(dim)
        self.down_proj = nn.Linear(dim, bottleneck_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=bottleneck_dim, nhead=num_heads, dim_feedforward=bottleneck_dim * 4,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.up_proj = nn.Linear(bottleneck_dim, dim)
        self.out_drop = nn.Dropout(dropout)

    def forward(self, cls_feat, group_feats, seqL=None):
        B, T, M, D = group_feats.shape
        tokens = torch.cat([cls_feat.unsqueeze(2), group_feats], dim=2)  # [B,T,1+M,D]
        x = self.down_proj(self.norm_in(tokens))
        x = x.permute(0, 2, 1, 3).reshape(B * (1 + M), T, -1)
        mask = None
        if seqL is not None:
            seq_t = seqL.to(x.device) if torch.is_tensor(seqL) else torch.tensor(seqL, device=x.device)
            seq_t = seq_t.view(-1).long()
            frame_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
            mask = (frame_ids >= seq_t.unsqueeze(1)).unsqueeze(1).expand(B, 1 + M, T).reshape(B * (1 + M), T)
        x = self.encoder(x, src_key_padding_mask=mask)
        x = x.view(B, 1 + M, T, -1).permute(0, 2, 1, 3)
        out = tokens + self.out_drop(self.up_proj(x))
        return out[:, :, 0, :], out[:, :, 1:, :]


class PreRouterTemporalPatchBlock(nn.Module):
    """Light temporal refinement before semantic routing.

    This block mirrors the post-router SemanticCrossFrameAttentionBlock, but it
    operates on CLIP visual tokens before text-axis routing.

    Key design choices:
    - It models only the same token index across frames, not full space-time
      attention, so the CLIP patch layout is preserved.
    - It is residual with a very small learnable scale, so the initial behavior
      is close to identity and should not abruptly disturb CLIP image-text
      alignment.
    - CLS and patch tokens are refined together, keeping their video context
      consistent before z_view estimation and token saliency bias generation.
    """

    def __init__(
        self,
        dim=512,
        bottleneck_dim=128,
        num_heads=2,
        depth=1,
        dropout=0.1,
        init_scale=1e-3,
        refine_cls=True,
        refine_patch=True,
    ):
        super().__init__()
        self.refine_cls = bool(refine_cls)
        self.refine_patch = bool(refine_patch)
        self.norm_in = nn.LayerNorm(dim)
        self.down_proj = nn.Linear(dim, bottleneck_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=bottleneck_dim,
            nhead=num_heads,
            dim_feedforward=bottleneck_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.up_proj = nn.Linear(bottleneck_dim, dim)
        self.out_drop = nn.Dropout(dropout)
        self.gamma = nn.Parameter(torch.ones(1) * float(init_scale))

    def forward(self, cls_tokens, patch_tokens, seqL=None):
        """
        cls_tokens:   [B,T,1,D] or [B,T,D]
        patch_tokens: [B,T,N,D]
        """
        B, T, N, D = patch_tokens.shape
        cls_is_4d = (cls_tokens.dim() == 4)
        cls_feat = cls_tokens.squeeze(2) if cls_is_4d else cls_tokens

        tokens = torch.cat([cls_feat.unsqueeze(2), patch_tokens], dim=2)  # [B,T,1+N,D]
        x = self.down_proj(self.norm_in(tokens))
        x = x.permute(0, 2, 1, 3).reshape(B * (1 + N), T, -1)

        mask = None
        if seqL is not None:
            seq_t = seqL.to(x.device) if torch.is_tensor(seqL) else torch.tensor(seqL, device=x.device)
            seq_t = seq_t.view(-1).long()
            frame_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
            mask = (frame_ids >= seq_t.unsqueeze(1)).unsqueeze(1).expand(B, 1 + N, T)
            mask = mask.reshape(B * (1 + N), T)

        x = self.encoder(x, src_key_padding_mask=mask)
        x = x.view(B, 1 + N, T, -1).permute(0, 2, 1, 3)
        refined = tokens + self.gamma * self.out_drop(self.up_proj(x))

        cls_out = refined[:, :, 0, :] if self.refine_cls else cls_feat
        patch_out = refined[:, :, 1:, :] if self.refine_patch else patch_tokens

        if seqL is not None:
            seq_t = seqL.to(patch_tokens.device) if torch.is_tensor(seqL) else torch.tensor(seqL, device=patch_tokens.device)
            seq_t = seq_t.view(-1).long()
            valid = torch.arange(T, device=patch_tokens.device).unsqueeze(0).expand(B, T) < seq_t.unsqueeze(1)
            cls_out = torch.where(valid.unsqueeze(-1), cls_out, cls_feat)
            patch_out = torch.where(valid.unsqueeze(-1).unsqueeze(-1), patch_out, patch_tokens)

        return cls_out.unsqueeze(2) if cls_is_4d else cls_out, patch_out


class ClassAttentionRefiner(nn.Module):
    """Use semantic ID tokens to conservatively refine frame CLS tokens."""

    def __init__(self, embed_dim=512, num_heads=8, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0, init_scale=1e-3):
        super().__init__()
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.norm_ffn = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=attn_drop, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)), nn.GELU(), nn.Dropout(proj_drop),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim), nn.Dropout(proj_drop)
        )
        self.gamma_attn = nn.Parameter(torch.ones(1) * float(init_scale))
        self.gamma_ffn = nn.Parameter(torch.ones(1) * float(init_scale))

    def forward(self, cls_feat, group_feats, seqL=None):
        B, T, D = cls_feat.shape
        M = group_feats.shape[2]
        q = self.norm_q(cls_feat.reshape(B * T, 1, D))
        kv = self.norm_kv(group_feats.reshape(B * T, M, D))
        attn_out, _ = self.cross_attn(q, kv, kv, need_weights=False)
        x = cls_feat.reshape(B * T, 1, D) + self.gamma_attn * attn_out
        x = (x + self.gamma_ffn * self.ffn(self.norm_ffn(x))).reshape(B, T, D)
        if seqL is not None:
            seq_t = seqL.to(x.device) if torch.is_tensor(seqL) else torch.tensor(seqL, device=x.device)
            seq_t = seq_t.view(-1).long()
            mask = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T) < seq_t.unsqueeze(1)
            x = torch.where(mask.unsqueeze(-1), x, cls_feat)
        return x


class SemanticTemporalPooling(nn.Module):
    """Masked temporal max pooling for semantic parts."""

    def forward(self, x, seqL=None):
        B, T, M, D = x.shape
        if seqL is not None:
            seq_t = seqL.to(x.device) if torch.is_tensor(seqL) else torch.tensor(seqL, device=x.device)
            seq_t = seq_t.view(-1).long()
            mask = (torch.arange(T, device=x.device).unsqueeze(0).expand(B, T) < seq_t.unsqueeze(1)).unsqueeze(-1).unsqueeze(-1)
            x = x.masked_fill(~mask.expand(-1, -1, M, D), -1e9)
        return x.max(dim=1)[0].clamp(min=-1e4, max=1e4)


class SemanticAxisDropout(nn.Module):
    """Drop a small subset of ID axes during training to avoid single-axis over-reliance."""

    def __init__(self, drop_prob=0.10, max_drop_ratio=0.15):
        super().__init__()
        self.drop_prob = float(drop_prob)
        self.max_drop_ratio = float(max_drop_ratio)

    def forward(self, group_feats):
        if (not self.training) or self.drop_prob <= 0 or torch.rand(1, device=group_feats.device).item() > self.drop_prob:
            return group_feats
        B, M, D = group_feats.shape
        max_drop = max(1, int(round(M * self.max_drop_ratio)))
        num_drop = torch.randint(1, max_drop + 1, (1,), device=group_feats.device).item()
        mask = torch.ones(B, M, 1, device=group_feats.device, dtype=group_feats.dtype)
        for b in range(B):
            mask[b, torch.randperm(M, device=group_feats.device)[:num_drop], :] = 0.0
        return group_feats * mask / mask.mean(dim=1, keepdim=True).clamp(min=1e-6)


# =========================================================================================
# Visualization helpers
# =========================================================================================
class RouterAttentionVisualizer:
    """Save training-debug combo figures or paper-ready separated/combined figures."""

    def __init__(self, cfg, save_root, id_axis_names, context_axis_names, clip_cfg_obj):
        self.cfg = dict(cfg or {})
        self.enable = bool(self.cfg.get("enable", False))
        self.save_root = Path(save_root) / "router_visualization"
        self.id_axis_names = list(id_axis_names)
        self.context_axis_names = list(context_axis_names)
        self.events = {"train": 0, "eval": 0}
        self.clip_cfg_obj = clip_cfg_obj

    def should_save(self, is_training, step=0):
        if not self.enable:
            return False
        phase = "train" if is_training else "eval"
        if is_training and not bool(self.cfg.get("save_during_train", False)):
            return False
        if (not is_training) and not bool(self.cfg.get("save_during_eval", False)):
            return False
        if self.events[phase] >= int(self.cfg.get("max_events", 0)):
            return False
        start_step = int(self.cfg.get("start_step", 0))
        if int(step) < start_step:
            return False
        interval = int(self.cfg.get("train_interval" if is_training else "eval_interval", 1))
        interval = max(1, interval)
        return (int(step) - start_step) % interval == 0

    def mark_saved(self, is_training):
        self.events["train" if is_training else "eval"] += 1

    @staticmethod
    def _is_main_process():
        return (not torch.distributed.is_available()) or (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0

    def _get_input_size(self):
        size = list(self.clip_cfg_obj.INPUT.SIZE_TRAIN)
        return int(size[0]), int(size[1])

    def _get_stride_size(self):
        stride = list(self.clip_cfg_obj.MODEL.STRIDE_SIZE)
        return int(stride[0]), int(stride[1])

    def _infer_patch_grid(self, num_patches):
        patch_grid = self.cfg.get("patch_grid", "auto_clip")
        if isinstance(patch_grid, (list, tuple)) and len(patch_grid) == 2:
            gh, gw = int(patch_grid[0]), int(patch_grid[1])
        elif str(patch_grid).lower() == "auto_clip":
            H, W = self._get_input_size()
            sh, sw = self._get_stride_size()
            patch_size = int(self.cfg.get("patch_size", 16))
            gh = (H - patch_size) // sh + 1
            gw = (W - patch_size) // sw + 1
        else:
            side = int(round(math.sqrt(num_patches)))
            gh, gw = side, num_patches // max(1, side)
        if gh * gw != int(num_patches):
            raise ValueError(f"Patch grid mismatch: grid={gh}x{gw}, N={num_patches}. Check INPUT.SIZE_TRAIN and STRIDE_SIZE.")
        return gh, gw

    @staticmethod
    def _recover_rgb_image(img, image_range="clip_norm", channel_order="rgb"):
        if torch.is_tensor(img):
            img = img.detach().float().cpu()
        if img.ndim == 3 and img.shape[0] in [1, 3]:
            img = img.permute(1, 2, 0)
        if torch.is_tensor(img):
            img = img.numpy()
        img = np.asarray(img, dtype=np.float32)
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        if str(channel_order).lower() == "bgr":
            img = img[..., ::-1]

        image_range = str(image_range).lower()
        if image_range == "clip_norm":
            mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
            std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
            img = img * std + mean
        elif image_range in ["zero_one", "0_1"]:
            pass
        elif image_range in ["minus_one_one", "-1_1"]:
            img = (img + 1.0) * 0.5
        elif image_range in ["zero_255", "0_255"]:
            img = img / 255.0
        else:
            raise ValueError(f"Unsupported image_range={image_range}. Use clip_norm, zero_one, minus_one_one, or zero_255.")
        return (np.clip(img, 0.0, 1.0) * 255.0).round().astype(np.uint8)

    @staticmethod
    def _normalize_attention(attn, method="percentile", lo_p=1.0, hi_p=99.5):
        attn = np.asarray(attn, dtype=np.float32)
        if method == "percentile":
            lo = np.percentile(attn, float(lo_p))
            hi = np.percentile(attn, float(hi_p))
        else:
            lo, hi = float(attn.min()), float(attn.max())
        return np.clip((attn - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

    @staticmethod
    def _resize_map(attn, grid_hw, image_hw, interpolation="bicubic"):
        gh, gw = grid_hw
        H, W = image_hw
        if torch.is_tensor(attn):
            attn = attn.detach().float().cpu()
        if isinstance(attn, np.ndarray):
            attn = torch.from_numpy(attn.astype(np.float32))
        if attn.ndim == 1:
            attn = attn.view(gh, gw)
        attn = attn.view(1, 1, gh, gw)
        mode = "bicubic" if str(interpolation).lower() == "bicubic" else "bilinear"
        out = F.interpolate(attn, size=(H, W), mode=mode, align_corners=False)
        return out.squeeze().numpy()

    @staticmethod
    def _boundary_from_mask(mask):
        mask = mask.astype(bool)
        pad = np.pad(mask, 1, mode="constant", constant_values=False)
        center = pad[1:-1, 1:-1]
        eroded = center & pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:]
        return center & (~eroded)

    def _overlay(self, rgb_uint8, attn_map, is_prior=False):
        from PIL import Image
        import matplotlib.cm as cm

        alpha = float(self.cfg.get("overlay_alpha", 0.26))
        lo_p = float(self.cfg.get("percentile_low", 1.0))
        hi_p = float(self.cfg.get("percentile_high", 99.5))
        method = self.cfg.get("normalization", "percentile")
        min_visible = float(self.cfg.get("min_visible", 0.12))
        gamma = float(self.cfg.get("attention_gamma", 0.75))

        if is_prior:
            scale = np.percentile(np.abs(attn_map), float(self.cfg.get("prior_percentile", 99.0)))
            norm = np.clip(attn_map / max(scale, 1e-6), -1.0, 1.0)
            vis = (norm + 1.0) * 0.5
            cmap = cm.get_cmap("coolwarm")
        else:
            vis = self._normalize_attention(attn_map, method, lo_p, hi_p)
            vis = np.power(vis, gamma)
            cmap = cm.get_cmap("turbo")

        heat = (cmap(vis)[..., :3] * 255.0).astype(np.float32)
        rgb = rgb_uint8.astype(np.float32)
        if is_prior:
            mask = np.ones_like(vis)[..., None]
            local_alpha = min(alpha, 0.35)
        else:
            mask = np.clip((vis - min_visible) / max(1.0 - min_visible, 1e-6), 0.0, 1.0)[..., None]
            local_alpha = alpha
        out = rgb * (1.0 - local_alpha * mask) + heat * (local_alpha * mask)

        if (not is_prior) and bool(self.cfg.get("draw_attention_contour", True)):
            threshold = np.percentile(vis, float(self.cfg.get("contour_percentile", 85.0)))
            boundary = self._boundary_from_mask(vis >= threshold)
            out[boundary] = 255.0
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

    @staticmethod
    def _draw_title(tile, title, height=48):
        from PIL import Image, ImageDraw
        w, h = tile.size
        canvas = Image.new("RGB", (w, h + height), "white")
        canvas.paste(tile, (0, height))
        draw = ImageDraw.Draw(canvas)
        draw.text((4, 4), str(title)[:80], fill=(0, 0, 0))
        return canvas

    @staticmethod
    def _make_grid(tiles, max_columns=4):
        from PIL import Image
        if not tiles:
            return None
        widths, heights = zip(*[im.size for im in tiles])
        tw, th = max(widths), max(heights)
        cols = min(int(max_columns), len(tiles))
        rows = int(math.ceil(len(tiles) / cols))
        canvas = Image.new("RGB", (cols * tw, rows * th), "white")
        for idx, im in enumerate(tiles):
            r, c = divmod(idx, cols)
            canvas.paste(im, (c * tw, r * th))
        return canvas


    @staticmethod
    def _get_font(size=10):
        """Get a readable font without depending on project-specific font files."""
        from PIL import ImageFont
        try:
            return ImageFont.truetype("DejaVuSans.ttf", int(size))
        except Exception:
            return ImageFont.load_default()

    @staticmethod
    def _short_axis_name(name, max_chars=32):
        name = str(name)
        if len(name) <= int(max_chars):
            return name
        return name[:max(1, int(max_chars) - 3)] + "..."

    @staticmethod
    def _normalize_matrix_values(mat, value_min=0.0, value_max=100.0, method="percentile", lo_p=1.0, hi_p=99.5):
        """Normalize an attention matrix to a display range, default [0, 100]."""
        mat = np.asarray(mat, dtype=np.float32)
        if mat.size == 0:
            return mat.copy()
        method = str(method).lower()
        if method == "percentile":
            lo = np.percentile(mat, float(lo_p))
            hi = np.percentile(mat, float(hi_p))
        elif method in ["minmax", "min_max"]:
            lo, hi = float(np.min(mat)), float(np.max(mat))
        elif method == "row_minmax":
            lo = np.min(mat, axis=1, keepdims=True)
            hi = np.max(mat, axis=1, keepdims=True)
            norm = (mat - lo) / np.maximum(hi - lo, 1e-6)
            return np.clip(norm * (float(value_max) - float(value_min)) + float(value_min), float(value_min), float(value_max))
        elif method == "row_percentile":
            lo = np.percentile(mat, float(lo_p), axis=1, keepdims=True)
            hi = np.percentile(mat, float(hi_p), axis=1, keepdims=True)
            norm = (mat - lo) / np.maximum(hi - lo, 1e-6)
            return np.clip(norm * (float(value_max) - float(value_min)) + float(value_min), float(value_min), float(value_max))
        else:
            lo, hi = float(np.min(mat)), float(np.max(mat))
        norm = (mat - lo) / max(float(hi - lo), 1e-6)
        out = norm * (float(value_max) - float(value_min)) + float(value_min)
        return np.clip(out, float(value_min), float(value_max))

    def _draw_axis_patch_attention_matrix(
        self,
        matrix,
        axis_names,
        title,
        patch_grid_hw=None,
        role_split_idx=None,
    ):
        """Draw a full semantic-axis by patch attention matrix.

        Rows are semantic axes, columns are visual patches. Each cell is filled with
        the normalized value in [0, 100] and colored by a cold-to-warm colormap.
        """
        from PIL import Image, ImageDraw
        import matplotlib.cm as cm

        if torch.is_tensor(matrix):
            matrix_np = matrix.detach().float().cpu().numpy()
        else:
            matrix_np = np.asarray(matrix, dtype=np.float32)
        if matrix_np.ndim != 2:
            raise ValueError(f"Expected a 2D attention matrix [num_axes, num_patches], got {matrix_np.shape}")

        num_axes, num_patches = matrix_np.shape
        axis_names = list(axis_names)
        if len(axis_names) != num_axes:
            axis_names = [f"axis_{i:02d}" for i in range(num_axes)]

        value_min = float(self.cfg.get("attention_matrix_value_min", 0.0))
        value_max = float(self.cfg.get("attention_matrix_value_max", 100.0))
        norm_method = self.cfg.get("attention_matrix_normalization", "percentile")
        lo_p = float(self.cfg.get("attention_matrix_percentile_low", self.cfg.get("percentile_low", 1.0)))
        hi_p = float(self.cfg.get("attention_matrix_percentile_high", self.cfg.get("percentile_high", 99.5)))
        values = self._normalize_matrix_values(
            matrix_np,
            value_min=value_min,
            value_max=value_max,
            method=norm_method,
            lo_p=lo_p,
            hi_p=hi_p,
        )

        cell_w = int(self.cfg.get("attention_matrix_cell_width", 28))
        cell_h = int(self.cfg.get("attention_matrix_cell_height", 20))
        cell_w = max(12, cell_w)
        cell_h = max(12, cell_h)

        left_margin = int(self.cfg.get("attention_matrix_left_margin", 250))
        top_margin = int(self.cfg.get("attention_matrix_top_margin", 74))
        right_margin = int(self.cfg.get("attention_matrix_right_margin", 92))
        bottom_margin = int(self.cfg.get("attention_matrix_bottom_margin", 54))
        title_font_size = int(self.cfg.get("attention_matrix_title_font_size", 13))
        axis_font_size = int(self.cfg.get("attention_matrix_axis_font_size", 8))
        value_font_size = int(self.cfg.get("attention_matrix_value_font_size", 7))
        tick_font_size = int(self.cfg.get("attention_matrix_tick_font_size", 8))
        axis_name_max_chars = int(self.cfg.get("attention_matrix_axis_label_max_chars", 34))
        x_tick_step = max(1, int(self.cfg.get("attention_matrix_x_tick_step", 5)))
        show_values = bool(self.cfg.get("attention_matrix_show_values", True))
        show_grid = bool(self.cfg.get("attention_matrix_show_grid", True))
        value_fmt = str(self.cfg.get("attention_matrix_value_format", "{:.0f}"))
        cmap_name = str(self.cfg.get("attention_matrix_cmap", self.cfg.get("colorbar_cmap", "turbo")))

        try:
            cmap = cm.get_cmap(cmap_name)
        except Exception:
            cmap = cm.get_cmap("turbo")

        matrix_w = num_patches * cell_w
        matrix_h = num_axes * cell_h
        canvas_w = left_margin + matrix_w + right_margin
        canvas_h = top_margin + matrix_h + bottom_margin

        canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
        draw = ImageDraw.Draw(canvas)
        title_font = self._get_font(title_font_size)
        axis_font = self._get_font(axis_font_size)
        value_font = self._get_font(value_font_size)
        tick_font = self._get_font(tick_font_size)

        gh, gw = patch_grid_hw if patch_grid_hw is not None else (None, None)
        subtitle = f"axes={num_axes}, patches={num_patches}"
        if gh is not None and gw is not None:
            subtitle += f" ({gh}x{gw})"
        subtitle += f", normalized to [{value_min:.0f}, {value_max:.0f}] by {norm_method}"

        draw.text((8, 8), str(title), fill=(0, 0, 0), font=title_font)
        draw.text((8, 30), subtitle, fill=(60, 60, 60), font=tick_font)
        draw.text((left_margin, top_margin - 28), "Patch index", fill=(0, 0, 0), font=tick_font)

        # X tick labels.
        for p in range(num_patches):
            if p % x_tick_step == 0 or p == num_patches - 1:
                x = left_margin + p * cell_w + cell_w // 2
                draw.text((x - 6, top_margin - 14), str(p), fill=(0, 0, 0), font=tick_font)

        denom = max(value_max - value_min, 1e-6)
        for r in range(num_axes):
            y0 = top_margin + r * cell_h
            y1 = y0 + cell_h
            # Axis labels.
            label = self._short_axis_name(axis_names[r], axis_name_max_chars)
            draw.text((8, y0 + max(0, (cell_h - axis_font_size) // 2)), label, fill=(0, 0, 0), font=axis_font)

            for c in range(num_patches):
                x0 = left_margin + c * cell_w
                x1 = x0 + cell_w
                v = float(values[r, c])
                color_pos = np.clip((v - value_min) / denom, 0.0, 1.0)
                rgb = tuple((np.asarray(cmap(color_pos)[:3]) * 255.0).astype(np.uint8).tolist())
                draw.rectangle([x0, y0, x1, y1], fill=rgb)

                if show_values:
                    # White text on dark colors; black text on bright colors.
                    luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
                    txt_color = (255, 255, 255) if luminance < 115 else (0, 0, 0)
                    try:
                        txt = value_fmt.format(v)
                    except Exception:
                        txt = f"{v:.0f}"
                    # Keep dense matrices readable by centering compact integer labels.
                    bbox = draw.textbbox((0, 0), txt, font=value_font)
                    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    if tw <= cell_w - 1 and th <= cell_h - 1:
                        draw.text((x0 + (cell_w - tw) / 2, y0 + (cell_h - th) / 2 - 1), txt, fill=txt_color, font=value_font)

                if show_grid:
                    draw.rectangle([x0, y0, x1, y1], outline=(235, 235, 235), width=1)

        # Role separator between ID and CTX axes.
        if role_split_idx is not None and 0 < int(role_split_idx) < num_axes:
            split_y = top_margin + int(role_split_idx) * cell_h
            draw.line([(0, split_y), (left_margin + matrix_w, split_y)], fill=(0, 0, 0), width=2)
            draw.text((8, split_y + 2), "CTX axes", fill=(0, 0, 0), font=tick_font)
            draw.text((8, top_margin + 2), "ID axes", fill=(0, 0, 0), font=tick_font)

        # Colorbar.
        bar_x0 = left_margin + matrix_w + 24
        bar_w = int(self.cfg.get("attention_matrix_colorbar_width", 18))
        bar_h = min(matrix_h, int(self.cfg.get("attention_matrix_colorbar_height", 260)))
        bar_y0 = top_margin
        for yy in range(bar_h):
            pos = 1.0 - yy / max(bar_h - 1, 1)
            rgb = tuple((np.asarray(cmap(pos)[:3]) * 255.0).astype(np.uint8).tolist())
            draw.rectangle([bar_x0, bar_y0 + yy, bar_x0 + bar_w, bar_y0 + yy + 1], fill=rgb)
        draw.rectangle([bar_x0, bar_y0, bar_x0 + bar_w, bar_y0 + bar_h], outline=(0, 0, 0), width=1)

        tick_values = self.cfg.get("attention_matrix_colorbar_ticks", [0, 25, 50, 75, 100])
        for tv in tick_values:
            tv = float(tv)
            pos = np.clip((tv - value_min) / denom, 0.0, 1.0)
            y = bar_y0 + int(round((1.0 - pos) * (bar_h - 1)))
            draw.line([(bar_x0 + bar_w, y), (bar_x0 + bar_w + 5, y)], fill=(0, 0, 0), width=1)
            draw.text((bar_x0 + bar_w + 8, y - 5), f"{tv:.0f}", fill=(0, 0, 0), font=tick_font)

        draw.text((left_margin, canvas_h - 36), "Each cell: normalized attention value for one semantic axis and one visual patch.", fill=(60, 60, 60), font=tick_font)
        return canvas, values

    def _axis_indices(self, names, requested, topk, scores=None):
        name_to_idx = {n: i for i, n in enumerate(names)}
        indices = []
        for name in requested:
            if name in name_to_idx:
                indices.append(name_to_idx[name])
        if int(topk) > 0 and scores is not None:
            top = torch.topk(scores.detach().float().cpu(), k=min(int(topk), len(names))).indices.tolist()
            indices += top
        out = []
        for idx in indices:
            if idx not in out:
                out.append(idx)
        return out

    def save(self, rgb_bcthw, seqL, vis_aux, typs=None, labels=None, is_training=True, step=0):
        if (not self.enable) or (not self._is_main_process()):
            return
        try:
            from PIL import Image
        except Exception:
            return

        phase = "train" if is_training else "eval"
        event_id = self.events[phase]
        root = self.save_root / phase / f"event_{event_id:04d}_step_{int(step):07d}"
        root.mkdir(parents=True, exist_ok=True)

        B, C, T, H, W = rgb_bcthw.shape
        samples = min(int(self.cfg.get("samples_per_event", 1)), B)
        frames_per_sample = min(int(self.cfg.get("frames_per_sample", 3)), T)
        tile_width = int(self.cfg.get("tile_width", 160))
        title_height = int(self.cfg.get("title_height", 48))
        max_columns = int(self.cfg.get("max_columns", 4))
        stages = list(self.cfg.get("attention_stages", ["final"]))
        layout = str(self.cfg.get("layout", "combo")).lower()
        raw_attention = {}

        N = int(vis_aux["id"][stages[0]]["biased_attn"].shape[-1])
        grid_hw = self._infer_patch_grid(N)
        selected_samples = list(range(samples))
        for b in selected_samples:
            valid_T = int(seqL[b].item()) if torch.is_tensor(seqL) else int(seqL[b]) if seqL is not None else T
            valid_T = max(1, min(valid_T, T))
            if frames_per_sample == 1:
                frame_ids = [valid_T // 2]
            else:
                frame_ids = np.linspace(0, valid_T - 1, frames_per_sample).round().astype(int).tolist()

            sample_root = root / f"sample_{b:02d}"
            sample_root.mkdir(parents=True, exist_ok=True)
            for tt in frame_ids:
                rgb = self._recover_rgb_image(
                    rgb_bcthw[b, :, tt], self.cfg.get("image_range", "clip_norm"), self.cfg.get("channel_order", "rgb")
                )
                if bool(self.cfg.get("save_clean_original", True)):
                    Image.fromarray(rgb).save(sample_root / f"frame_{tt:03d}_original.png")

                # Scores for optional top-k are average peak attention over selected stages.
                id_scores = vis_aux["id"][stages[-1]]["biased_attn"][b, tt].amax(dim=-1)
                ctx_scores = vis_aux["context"][stages[-1]]["biased_attn"][b, tt].amax(dim=-1)
                id_indices = self._axis_indices(self.id_axis_names, self.cfg.get("id_axes", []), self.cfg.get("topk_id", 0), id_scores)
                ctx_indices = self._axis_indices(self.context_axis_names, self.cfg.get("context_axes", []), self.cfg.get("topk_context", 0), ctx_scores)

                tiles = [self._draw_title(Image.fromarray(rgb).resize((tile_width, int(round(tile_width * H / W)))), "Original", title_height)]

                # ------------------------------------------------------------------
                # Full axis-patch attention matrix visualization.
                # Rows: semantic axes (ID first, then CTX). Columns: visual patches.
                # Each cell stores a normalized 0-100 attention value and uses a
                # cold-to-warm background color to show magnitude.
                # ------------------------------------------------------------------
                if bool(self.cfg.get("include_attention_matrix", False)):
                    matrix_attn_type_default = "query_attn" if bool(self.cfg.get("use_query_only_for_paper", False)) else "biased_attn"
                    matrix_attn_type = str(self.cfg.get("attention_matrix_attn_type", matrix_attn_type_default))
                    matrix_stages = list(self.cfg.get("attention_matrix_stages", stages))
                    matrix_root = sample_root / f"frame_{tt:03d}_attention_matrices"
                    matrix_root.mkdir(parents=True, exist_ok=True)
                    matrix_axis_names = self.id_axis_names + self.context_axis_names
                    for matrix_stage in matrix_stages:
                        matrix_parts = []
                        if (
                            "id" in vis_aux and matrix_stage in vis_aux["id"]
                            and matrix_attn_type in vis_aux["id"][matrix_stage]
                        ):
                            matrix_parts.append(vis_aux["id"][matrix_stage][matrix_attn_type][b, tt])
                        if (
                            "context" in vis_aux and matrix_stage in vis_aux["context"]
                            and matrix_attn_type in vis_aux["context"][matrix_stage]
                            and len(self.context_axis_names) > 0
                        ):
                            matrix_parts.append(vis_aux["context"][matrix_stage][matrix_attn_type][b, tt])
                        if len(matrix_parts) == 0:
                            continue
                        matrix_tensor = torch.cat(matrix_parts, dim=0)
                        matrix_img, matrix_values = self._draw_axis_patch_attention_matrix(
                            matrix_tensor,
                            matrix_axis_names[: int(matrix_tensor.shape[0])],
                            title=f"{matrix_stage.upper()} {matrix_attn_type} axis-patch attention matrix",
                            patch_grid_hw=grid_hw,
                            role_split_idx=len(self.id_axis_names),
                        )
                        matrix_png = matrix_root / f"frame_{tt:03d}_{matrix_stage}_{matrix_attn_type}_axis_patch_matrix.png"
                        matrix_img.save(matrix_png)
                        if bool(self.cfg.get("attention_matrix_save_pdf", False)):
                            matrix_img.save(matrix_root / f"frame_{tt:03d}_{matrix_stage}_{matrix_attn_type}_axis_patch_matrix.pdf")
                        raw_attention[f"sample{b}_frame{tt}_{matrix_stage}_{matrix_attn_type}_axis_patch_matrix_raw"] = matrix_tensor.detach().cpu().numpy()
                        raw_attention[f"sample{b}_frame{tt}_{matrix_stage}_{matrix_attn_type}_axis_patch_matrix_0_100"] = matrix_values.astype(np.float32)

                if bool(self.cfg.get("include_shared_prior", True)):
                    for branch, key in [("ID prior", "id"), ("CTX prior", "context")]:
                        prior = vis_aux["prior"][key][b, tt]
                        prior_map = self._resize_map(prior, grid_hw, (H, W), self.cfg.get("heatmap_interpolation", "bicubic"))
                        im = self._overlay(rgb, prior_map, is_prior=True).resize((tile_width, int(round(tile_width * H / W))))
                        tiles.append(self._draw_title(im, branch, title_height))
                        raw_attention[f"sample{b}_frame{tt}_{key}_prior"] = prior.detach().cpu().numpy()

                if bool(self.cfg.get("include_v2t_competition", True)) and "joint" in vis_aux and "v2t" in vis_aux["joint"]:
                    v2t = vis_aux["joint"]["v2t"]
                    for map_key, title, signed in [
                        ("id_mass", "V2T ID mass", False),
                        ("ctx_mass", "V2T CTX mass", False),
                        ("competition_delta", "V2T ID-CTX competition", True),
                    ]:
                        if map_key in v2t:
                            vmap = v2t[map_key][b, tt]
                            attn_map = self._resize_map(vmap, grid_hw, (H, W), self.cfg.get("heatmap_interpolation", "bicubic"))
                            im = self._overlay(rgb, attn_map, is_prior=signed).resize((tile_width, int(round(tile_width * H / W))))
                            tiles.append(self._draw_title(im, title, title_height))
                            raw_attention[f"sample{b}_frame{tt}_{map_key}"] = vmap.detach().cpu().numpy()

                for stage in stages:
                    for idx in id_indices:
                        axis = self.id_axis_names[idx]
                        attn_type = "query_attn" if bool(self.cfg.get("use_query_only_for_paper", False)) else "biased_attn"
                        attn = vis_aux["id"][stage][attn_type][b, tt, idx]
                        attn_map = self._resize_map(attn, grid_hw, (H, W), self.cfg.get("heatmap_interpolation", "bicubic"))
                        im = self._overlay(rgb, attn_map, is_prior=False).resize((tile_width, int(round(tile_width * H / W))))
                        tiles.append(self._draw_title(im, f"ID {stage}: {axis}", title_height))
                        raw_attention[f"sample{b}_frame{tt}_id_{stage}_{axis}_{attn_type}"] = attn.detach().cpu().numpy()
                    for idx in ctx_indices:
                        axis = self.context_axis_names[idx]
                        attn_type = "query_attn" if bool(self.cfg.get("use_query_only_for_paper", False)) else "biased_attn"
                        attn = vis_aux["context"][stage][attn_type][b, tt, idx]
                        attn_map = self._resize_map(attn, grid_hw, (H, W), self.cfg.get("heatmap_interpolation", "bicubic"))
                        im = self._overlay(rgb, attn_map, is_prior=False).resize((tile_width, int(round(tile_width * H / W))))
                        tiles.append(self._draw_title(im, f"CTX {stage}: {axis}", title_height))
                        raw_attention[f"sample{b}_frame{tt}_ctx_{stage}_{axis}_{attn_type}"] = attn.detach().cpu().numpy()

                grid = self._make_grid(tiles, max_columns=max_columns)
                if grid is not None and layout in ["combo", "both", "paper_combo"]:
                    grid.save(sample_root / f"frame_{tt:03d}_combo.png")
                    if bool(self.cfg.get("save_pdf", False)):
                        grid.save(sample_root / f"frame_{tt:03d}_combo.pdf")
                if layout in ["separate", "both", "paper_separate"]:
                    sep_root = sample_root / f"frame_{tt:03d}_tiles"
                    sep_root.mkdir(parents=True, exist_ok=True)
                    for i, im in enumerate(tiles):
                        im.save(sep_root / f"tile_{i:03d}.png")

        if bool(self.cfg.get("save_raw_attention", True)):
            np.savez_compressed(root / "raw_attention.npz", **raw_attention)
        meta = {"phase": phase, "step": int(step), "grid_hw": grid_hw, "id_axes": self.id_axis_names, "context_axes": self.context_axis_names}
        (root / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        self.mark_saved(is_training)


# =========================================================================================
# Main model
# =========================================================================================
class PointGait(BaseModel):
    def build_network(self, model_cfg, embed_dim=512):
        self.model_cfg = model_cfg
        num_desc = int(clip_cfg.MODEL.NON_VIEW_PROMPT.NUM_DESCRIPTIONS_PER_GROUP)
        dict_cfg = model_cfg.get("semantic_dictionary_cfg", {})
        dict_gen = build_semantic_dictionary(num_desc, **dict_cfg)
        self.group_prompts_non_view = dict_gen.get_group_prompts("non_view")
        self.group_names = dict_gen.get_group_names()
        self.group_roles = dict_gen.get_group_roles()
        self.num_groups = len(self.group_names)
        self.id_group_indices = [i for i, n in enumerate(self.group_names) if str(n).startswith("id_")]
        self.context_group_indices = [i for i, n in enumerate(self.group_names) if str(n).startswith("ctx_")]
        self.id_group_names = [self.group_names[i] for i in self.id_group_indices]
        self.context_group_names = [self.group_names[i] for i in self.context_group_indices]
        self.num_id_axes = len(self.id_group_indices)
        self.num_context_axes = len(self.context_group_indices)
        self.parts_num = 1 + self.num_id_axes

        model_cfg["SeparateFCs"]["parts_num"] = self.parts_num
        model_cfg["SeparateBNNecks"]["parts_num"] = self.parts_num
        self.FCs = SeparateFCs(**model_cfg["SeparateFCs"])
        self.BNNecks = SeparateBNNecks(**model_cfg["SeparateBNNecks"])

        text_cfg = model_cfg.get("text_adapter_cfg", {})
        self.text_adapter = TextFeatureAdapter(
            dim=embed_dim,
            hidden_dim=text_cfg.get("hidden_dim", 128),
            dropout=text_cfg.get("dropout", 0.1),
            init_scale=text_cfg.get("init_scale", 1e-3),
            use_layernorm=text_cfg.get("use_layernorm", True),
        )


        view_cfg = model_cfg.get("latent_view_cfg", {})
        self.pre_router_cls_pool = MaskedSequencePooling(view_cfg.get("pool_type", "mean"))
        self.latent_view_branch = LatentViewBranch(
            dim=embed_dim,
            hidden_dim=view_cfg.get("hidden_dim", 256),
            dropout=view_cfg.get("dropout", 0.1),
            use_layernorm=view_cfg.get("use_layernorm", True),
        )

        sal_cfg = model_cfg.get("token_saliency_cfg", {})
        self.token_saliency_bias_ID = ViewConditionedTokenSaliencyBias(
            embed_dim=embed_dim,
            hidden_dim=sal_cfg.get("hidden_dim", 256),
            dropout=sal_cfg.get("dropout", 0.1),
            init_bias_scale=sal_cfg.get("init_bias_scale", 0.25),
        )
        self.token_saliency_bias_CTX = ViewConditionedTokenSaliencyBias(
            embed_dim=embed_dim,
            hidden_dim=sal_cfg.get("hidden_dim", 256),
            dropout=sal_cfg.get("dropout", 0.1),
            init_bias_scale=sal_cfg.get("init_bias_scale", 0.25),
        )

        router_cfg = model_cfg.get("semantic_router_cfg", {})
        self.semantic_router = S2ASharedIDContextRouter(
            embed_dim=embed_dim,
            num_heads=router_cfg.get("num_heads", 8),
            mlp_ratio=router_cfg.get("mlp_ratio", 4.0),
            attn_drop=router_cfg.get("attn_drop", 0.0),
            proj_drop=router_cfg.get("proj_drop", 0.0),
            competition_bias_enable=router_cfg.get("competition_bias_enable", True),
            competition_bias_scale=router_cfg.get("competition_bias_scale", 0.35),
            competition_bias_clip=router_cfg.get("competition_bias_clip", 2.0),
            detach_competition=router_cfg.get("detach_competition", True),
            axis_competition_cfg=model_cfg.get("axis_competition_cfg", {}),
            semantic_calibration_cfg=router_cfg.get("semantic_calibration_cfg", {}),
            final_use_shared_patch_prior=router_cfg.get("final_use_shared_patch_prior", True),
            axis_specific_final_bias_enable=router_cfg.get("axis_specific_final_bias_enable", False),
            axis_specific_final_bias_scale=router_cfg.get("axis_specific_final_bias_scale", 0.15),
            axis_specific_final_bias_clip=router_cfg.get("axis_specific_final_bias_clip", 1.5),
            detach_axis_specific_final_bias=router_cfg.get("detach_axis_specific_final_bias", True),
        )

        self.cross_frame_enable = bool(model_cfg.get("cross_frame_enable", True))
        if self.cross_frame_enable:
            cross_cfg = model_cfg.get("cross_frame_cfg", {})
            self.cross_frame_block = SemanticCrossFrameAttentionBlock(
                dim=embed_dim,
                bottleneck_dim=cross_cfg.get("bottleneck_dim", 128),
                num_heads=cross_cfg.get("num_heads", 2),
                depth=cross_cfg.get("depth", 1),
                dropout=cross_cfg.get("dropout", 0.1),
            )

        self.class_attention_refiner_enable = bool(model_cfg.get("class_attention_refiner_enable", True))
        if self.class_attention_refiner_enable:
            ref_cfg = model_cfg.get("class_attention_refiner_cfg", {})
            self.class_refiner = ClassAttentionRefiner(
                embed_dim=embed_dim,
                num_heads=ref_cfg.get("num_heads", 8),
                mlp_ratio=ref_cfg.get("mlp_ratio", 4.0),
                attn_drop=ref_cfg.get("attn_drop", 0.0),
                proj_drop=ref_cfg.get("proj_drop", 0.0),
                init_scale=ref_cfg.get("init_scale", 1e-3),
            )

        drop_cfg = model_cfg.get("axis_dropout_cfg", {})
        self.axis_dropout = SemanticAxisDropout(
            drop_prob=drop_cfg.get("drop_prob", 0.10),
            max_drop_ratio=drop_cfg.get("max_drop_ratio", 0.15),
        )
        self.temporal_pool = SemanticTemporalPooling()
        self.cls_temporal_pool = SemanticTemporalPooling()
        self.context_temporal_pool = MaskedSequencePooling(pool_type="mean")

        self.num_platforms = int(model_cfg.get("num_platforms", 6))
        self.platform_label_offset = int(model_cfg.get("platform_label_offset", 1))
        self.z_view_loss_enable = bool(model_cfg.get("z_view_loss_enable", True))
        self.context_platform_loss_enable = bool(model_cfg.get("context_platform_loss_enable", True)) and self.num_context_axes > 0

        self.router_counterfactual_enable = bool(
            model_cfg.get("router_counterfactual_view_intervention_enable", False)
        )
        self.router_counterfactual_cfg = dict(
            model_cfg.get("router_counterfactual_view_intervention_cfg", {})
        )
        self.router_counterfactual_detach_z = bool(
            self.router_counterfactual_cfg.get("detach_counterfactual_z", True)
        )
        self.router_counterfactual_mix_ratio = float(
            self.router_counterfactual_cfg.get("mix_ratio", 1.0)
        )

        self.xview_axis_consistency_enable = bool(
            model_cfg.get("cross_view_axis_consistency_enable", False)
        )

        if self.z_view_loss_enable:
            self.z_view_classifier = nn.Linear(embed_dim, self.num_platforms)
        if self.context_platform_loss_enable:
            self.context_platform_classifier = nn.Linear(embed_dim, self.num_platforms)

        # Initialize new modules before CLIP is created. CLIP is loaded afterwards and is not touched by init_parameters().
        self.init_parameters()
        self._reset_stable_initializations()
        self.clip_model = make_model(clip_cfg, num_groups=self.num_groups)
        self.clip_model.setup_semantic_groups(group_prompts_non_view=self.group_prompts_non_view)

        for param in self.clip_model.parameters():
            param.requires_grad = False
        for name, param in self.clip_model.named_parameters():
            if "prompt" in name or "image_encoder" in name:
                param.requires_grad = True

        self.router_visualizer = RouterAttentionVisualizer(
            model_cfg.get("router_visualization_cfg", {}),
            save_root=getattr(self, "save_path", "output/router_vis"),
            id_axis_names=self.id_group_names,
            context_axis_names=self.context_group_names,
            clip_cfg_obj=clip_cfg,
        )

    def init_parameters(self):
        """Initialize STAR-CVI modules once while preserving pretrained CLIP.

        OpenGait calls ``init_parameters`` after ``build_network``. STAR-CVI
        initializes its newly introduced modules immediately before creating
        the CLIP backbone, so the later framework call must be idempotent.
        """
        if getattr(self, "_star_cvi_parameters_initialized", False):
            return
        super().init_parameters()
        self._star_cvi_parameters_initialized = True

    def _reset_stable_initializations(self):
        self.token_saliency_bias_ID.reset_parameters()
        self.token_saliency_bias_CTX.reset_parameters()
        if hasattr(self, "context_calibrator"):
            self.context_calibrator.reset_parameters()
        if hasattr(self, "z_view_classifier"):
            nn.init.normal_(self.z_view_classifier.weight, std=0.001)
            nn.init.zeros_(self.z_view_classifier.bias)
        if hasattr(self, "context_platform_classifier"):
            nn.init.normal_(self.context_platform_classifier.weight, std=0.001)
            nn.init.zeros_(self.context_platform_classifier.bias)

    def get_param_groups(self, optimizer_cfg):
        base_lr = float(optimizer_cfg.get("lr", 1e-5))
        prompt_lr = base_lr * float(optimizer_cfg.get("prompt_lr_multiplier", 2.0))
        new_lr = base_lr * float(optimizer_cfg.get("head_lr_multiplier", 3.0))
        image_params, prompt_params, new_params = [], [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if "clip_model" in name and "image_encoder" in name:
                image_params.append(p)
            elif "clip_model" in name and "prompt" in name:
                prompt_params.append(p)
            else:
                new_params.append(p)
        groups = []
        if image_params:
            groups.append({"params": image_params, "lr": base_lr, "name": "clip_image_encoder"})
        if prompt_params:
            groups.append({"params": prompt_params, "lr": prompt_lr, "name": "clip_prompt"})
        if new_params:
            groups.append({"params": new_params, "lr": new_lr, "name": "pointgait_new_modules"})
        return groups

    def get_optimizer(self, optimizer_cfg):
        """Build the optimizer with the paper's three learning-rate groups."""
        self.msg_mgr.log_info(optimizer_cfg)
        optimizer_cls = getattr(torch.optim, optimizer_cfg["solver"])
        valid_keys = set(inspect.signature(optimizer_cls.__init__).parameters)
        optimizer_kwargs = {
            key: value
            for key, value in optimizer_cfg.items()
            if key not in {"solver", "prompt_lr_multiplier", "head_lr_multiplier"}
            and key in valid_keys
            and key != "params"
        }
        return optimizer_cls(self.get_param_groups(optimizer_cfg), **optimizer_kwargs)

    def _parse_platform_labels(self, typs_batch, device):
        labels = []
        for typ in typs_batch:
            s = str(typ)
            token = s.split("-")[0].strip()
            if token.startswith("C"):
                token = token[1:]
            value = int(token) - self.platform_label_offset
            if value < 0 or value >= self.num_platforms:
                raise ValueError(
                    f"Parsed platform label {value} from type '{s}', but num_platforms={self.num_platforms}. "
                    f"Check platform_label_offset. Use 1 for C1..C6, 0 for C0..C5."
                )
            labels.append(value)
        return torch.tensor(labels, dtype=torch.long, device=device)

    @staticmethod
    def _ensure_seq_tensor(seqL, device):
        if seqL is None:
            return None
        seq_t = seqL.to(device) if torch.is_tensor(seqL) else torch.tensor(seqL, device=device)
        return seq_t.view(-1).long()

    def _sample_counterfactual_z_view(self, z_view, platform_labels=None):
        """Sample a counterfactual view condition for Router intervention.

        This operation keeps the image tokens and text axes fixed, but replaces
        the video-level view condition used by token saliency bias. It implements
        do(V=v') at the Router-input level rather than adversarially removing view
        information from the final ID embedding.

        z_view:          [B,D]
        platform_labels: [B], optional. If provided, prefer a donor from a
                         different platform; otherwise fall back to batch roll.
        """
        B = z_view.size(0)
        if B <= 1:
            return z_view.detach() if self.router_counterfactual_detach_z else z_view

        device = z_view.device
        perm = torch.arange(B, device=device)

        if platform_labels is not None and platform_labels.numel() == B:
            platform_labels = platform_labels.view(-1)
            for i in range(B):
                candidates = torch.nonzero(
                    platform_labels != platform_labels[i],
                    as_tuple=False
                ).view(-1)
                if candidates.numel() > 0:
                    j = candidates[torch.randint(candidates.numel(), (1,), device=device)]
                    perm[i] = j
                else:
                    perm[i] = (i + 1) % B
        else:
            perm = torch.roll(perm, shifts=1)

        donor = z_view[perm]
        if self.router_counterfactual_detach_z:
            donor = donor.detach()

        mix = min(max(float(self.router_counterfactual_mix_ratio), 0.0), 1.0)
        if mix < 1.0:
            base = z_view.detach() if self.router_counterfactual_detach_z else z_view
            donor = (1.0 - mix) * base + mix * donor

        # z_delta = (donor - z_view).abs().mean()
        # print(f"Router counterfactual view intervention: mean |delta|={z_delta:.6f}, mix_ratio={mix:.3f}")
        return donor

    def _id_head_forward(
        self,
        cls_tokens,
        routed_id,
        routed_ctx,
        z_view,
        seqL=None,
        return_calib_aux=False,
        apply_axis_dropout=True,
    ):
        """Run the complete downstream ReID head from routed semantic ID tokens.

        This helper is intentionally shared by the factual and counterfactual
        Router branches. The counterfactual path therefore cannot stop at
        ``routed_id_cf``: it goes through the same optional context calibration,
        cross-frame interaction, class-attention refinement, temporal pooling,
        axis dropout, SeparateFCs, and BNNecks as the main branch.
        """
        calib_aux = {}

        cls_frame = cls_tokens.squeeze(2) if cls_tokens.dim() == 4 else cls_tokens
        group_frame = routed_id
        if self.cross_frame_enable:
            cls_frame, group_frame = self.cross_frame_block(cls_frame, group_frame, seqL)
        if self.class_attention_refiner_enable:
            cls_frame = self.class_refiner(cls_frame, group_frame, seqL)

        group_pool = self.temporal_pool(group_frame, seqL)
        cls_pool = self.cls_temporal_pool(cls_frame.unsqueeze(2), seqL).squeeze(1)
        group_main = (
            self.axis_dropout(group_pool)
            if self.training and apply_axis_dropout
            else group_pool
        )

        joint_tokens = torch.cat([cls_pool.unsqueeze(1), group_main], dim=1)
        joint_tokens = joint_tokens.permute(0, 2, 1).contiguous()
        embeds = self.FCs(joint_tokens)
        _, logits = self.BNNecks(embeds)

        return {
            "embeds": embeds,
            "logits": logits,
            "routed_id": routed_id,
            "group_frame": group_frame,
            "group_pool": group_pool,
            "cls_frame": cls_frame,
            "cls_pool": cls_pool,
            "calib_aux": calib_aux,
        }

    def forward(self, inputs):
        ipts, labs, typs_batch, vies, seqL = inputs
        rgb_input = ipts[0]
        if rgb_input.dim() == 5 and rgb_input.shape[-1] in (1, 3):
            rgb = rearrange(rgb_input, "n s h w c -> n c s h w")
        elif rgb_input.dim() == 5 and rgb_input.shape[2] in (1, 3):
            rgb = rearrange(rgb_input, "n s c h w -> n c s h w")
        elif rgb_input.dim() == 4:
            rgb = rgb_input.unsqueeze(1)
        else:
            raise ValueError(
                "Expected RGB input shaped [B,T,H,W,C] or [B,T,C,H,W], "
                f"but received {tuple(rgb_input.shape)}"
            )
        B, _, T, _, _ = rgb.shape

        base_text = self.text_adapter(self.clip_model(get_text=True))
        base_id_text = base_text[self.id_group_indices]
        base_ctx_text = base_text[self.context_group_indices]

        cls_tokens, patch_tokens = self.clip_model(
            image_feats=rearrange(rgb, "b c t h w -> (b t) c h w"), get_image=True
        )
        cls_tokens = rearrange(cls_tokens, "(b t) l c -> b t l c", b=B, t=T)
        patch_tokens = rearrange(patch_tokens, "(b t) n c -> b t n c", b=B, t=T)

        pre_router_cls_summary = self.pre_router_cls_pool(cls_tokens.squeeze(2), seqL)
        z_view = self.latent_view_branch(pre_router_cls_summary)

        platform_labels = None
        if self.training and (
            self.z_view_loss_enable
            or self.context_platform_loss_enable
            or self.router_counterfactual_enable
            or self.xview_axis_consistency_enable
        ):
            platform_labels = self._parse_platform_labels(typs_batch, z_view.device)

        return_vis = self.router_visualizer.should_save(self.training, getattr(self, "iteration", 0))
        patch_attn_bias_id = self.token_saliency_bias_ID(cls_tokens, patch_tokens, z_view, seqL)
        patch_attn_bias_ctx = self.token_saliency_bias_CTX(cls_tokens, patch_tokens, z_view, seqL)

        routed_id, routed_ctx, router_vis_aux = self.semantic_router(
            id_text_feats=base_id_text,
            context_text_feats=base_ctx_text,
            patch_tokens=patch_tokens,
            cls_tokens=cls_tokens,
            patch_attn_bias_id=patch_attn_bias_id,
            patch_attn_bias_context=patch_attn_bias_ctx,
            return_visualization=return_vis,
        )

        routed_id_cf = None
        routed_ctx_cf = None
        z_view_cf = None
        if self.training and self.router_counterfactual_enable:
            z_view_cf = self._sample_counterfactual_z_view(z_view, platform_labels)
            patch_attn_bias_id_cf = self.token_saliency_bias_ID(cls_tokens, patch_tokens, z_view_cf, seqL)
            patch_attn_bias_ctx_cf = self.token_saliency_bias_CTX(cls_tokens, patch_tokens, z_view_cf, seqL)
            routed_id_cf, routed_ctx_cf, _ = self.semantic_router(
                id_text_feats=base_id_text,
                context_text_feats=base_ctx_text,
                patch_tokens=patch_tokens,
                cls_tokens=cls_tokens,
                patch_attn_bias_id=patch_attn_bias_id_cf,
                patch_attn_bias_context=patch_attn_bias_ctx_cf,
                return_visualization=False,
            )


        main_head = self._id_head_forward(
            cls_tokens=cls_tokens,
            routed_id=routed_id,
            routed_ctx=routed_ctx,
            z_view=z_view,
            seqL=seqL,
            return_calib_aux=True,
        )
        embeds = main_head["embeds"]
        logits = main_head["logits"]
        routed_id = main_head["routed_id"]
        group_frame = main_head["group_frame"]
        group_pool = main_head["group_pool"]
        calib_aux = main_head["calib_aux"]

        cf_head = None
        embeds_cf = None
        logits_cf = None
        group_pool_cf = None
        if self.training and self.router_counterfactual_enable and routed_id_cf is not None:
            cf_head = self._id_head_forward(
                cls_tokens=cls_tokens,
                routed_id=routed_id_cf,
                routed_ctx=routed_ctx_cf,
                z_view=z_view_cf,
                seqL=seqL,
                return_calib_aux=False,
            )
            embeds_cf = cf_head["embeds"]
            logits_cf = cf_head["logits"]
            routed_id_cf = cf_head["routed_id"]
            group_pool_cf = cf_head["group_pool"]

        training_feat = {
            "triplet": {"embeddings": embeds, "labels": labs},
            "softmax": {"logits": logits, "labels": labs},
            "id_div": {"prototypes": base_id_text},
        }
        if base_ctx_text.numel() > 0:
            training_feat["context_div"] = {"prototypes": base_ctx_text}

        if self.training and self.router_counterfactual_enable and embeds_cf is not None:
            training_feat["cf_triplet"] = {"embeddings": embeds_cf, "labels": labs}
            training_feat["cf_softmax"] = {"logits": logits_cf, "labels": labs}
            training_feat["cf_embed_consistency"] = {
                "anchor": embeds.detach(),
                "positive": embeds_cf,
            }

            # Kept for backward compatibility/diagnostics if an old config still
            # contains RouterCounterfactualConsistencyLoss. The recommended
            # supervision is now the downstream cf_* losses above.

            # training_feat["router_cf_consistency"] = {
            #     "factual": routed_id,
            #     "counterfactual": routed_id_cf,
            #     "seqL": seqL,
            # }

        if self.training and self.xview_axis_consistency_enable:
            training_feat["xview_axis_consistency"] = {
                "axis_feats": group_pool,
                "labels": labs,
                "platforms": platform_labels,
            }

        if self.training and self.z_view_loss_enable:
            z_view_logits = self.z_view_classifier(z_view).unsqueeze(2)
            training_feat["z_view_softmax"] = {"logits": z_view_logits, "labels": platform_labels}
        if self.training and self.context_platform_loss_enable and routed_ctx.size(2) > 0:
            ctx_frame = routed_ctx.mean(dim=2)
            ctx_video = self.context_temporal_pool(ctx_frame, seqL)
            ctx_logits = self.context_platform_classifier(ctx_video).unsqueeze(2)
            training_feat["ctx_platform_softmax"] = {"logits": ctx_logits, "labels": platform_labels}

        if return_vis:
            router_vis_aux["prior"] = {"id": patch_attn_bias_id.detach(), "context": patch_attn_bias_ctx.detach()}
            router_vis_aux["calibration"] = calib_aux
            self.router_visualizer.save(
                rgb_bcthw=rgb,
                seqL=seqL,
                vis_aux=router_vis_aux,
                typs=typs_batch,
                labels=labs,
                is_training=self.training,
                step=getattr(self, "iteration", 0),
            )

        return {
            "training_feat": training_feat,
            "inference_feat": {"embeddings": embeds},
            "visual_summary": {},
        }


# Public method name used by the release config. Keep PointGait as a checkpoint-
# compatible legacy name for existing experiments.
STARCVI = PointGait
