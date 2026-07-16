import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseLoss, gather_and_scale_wrapper


class PrototypeDiversityLoss(BaseLoss):
    """
    Encourage different text prototypes to stay diverse.

    Input:
        prototypes: [M, D]
    """
    def __init__(self, loss_term_weight=1.0):
        super(PrototypeDiversityLoss, self).__init__(loss_term_weight)

    def forward(self, prototypes):
        """
        prototypes: [M, D]
        """
        prototypes = F.normalize(prototypes.float(), p=2, dim=-1)   # [M, D]
        sim = torch.matmul(prototypes, prototypes.t())              # [M, M]

        eye = torch.eye(sim.size(0), device=sim.device, dtype=sim.dtype)
        offdiag = sim - eye

        loss = (offdiag ** 2).sum() / (sim.size(0) * (sim.size(0) - 1) + 1e-12)

        mean_offdiag = offdiag.abs().sum() / (sim.size(0) * (sim.size(0) - 1) + 1e-12)

        self.info.update({
            'loss': loss.detach().clone(),
            # 'mean_offdiag': mean_offdiag.detach().clone()
        })

        return loss, self.info

class LatentOrthogonalLoss(BaseLoss):
    """
    Encourage z_id_anchor and z_view to be approximately orthogonal.

    Inputs:
        z_id_anchor: [B, D]
        z_view:      [B, D]
    """
    def __init__(self, loss_term_weight=1.0):
        super(LatentOrthogonalLoss, self).__init__(loss_term_weight)

    @gather_and_scale_wrapper
    def forward(self, z_id_anchor, z_view):
        z_id_anchor = F.normalize(z_id_anchor.float(), p=2, dim=-1)
        z_view = F.normalize(z_view.float(), p=2, dim=-1)

        # Square the per-sample cosine similarity.
        cos_sim = torch.sum(z_id_anchor * z_view, dim=-1)    # [B]
        loss = (cos_sim ** 2).mean()

        self.info.update({
            'loss': loss.detach().clone()
        })
        return loss, self.info

class SemanticSlotSupConLoss(BaseLoss):
    """
    View-label-free supervised contrastive loss on semantic slot features.

    Input:
        features: [B,M,D] or [B,D,P]
        labels:   [B]

    It only uses identity labels, no view labels.
    """
    def __init__(
        self,
        loss_term_weight=1.0,
        temperature=0.07,
        mode='flatten'
    ):
        super(SemanticSlotSupConLoss, self).__init__(loss_term_weight)
        self.temperature = temperature
        self.mode = mode

    @gather_and_scale_wrapper
    def forward(self, features, labels):
        labels = labels.view(-1)

        if features.dim() != 3:
            raise ValueError(f"Expected 3D features, got {features.shape}")

        # Accept [B,M,D] or [B,D,P].
        # We recommend passing group_enhanced as [B,M,D].
        if self.mode == 'mean':
            z = features.mean(dim=1)
        else:
            z = features.reshape(features.size(0), -1)

        z = F.normalize(z.float(), p=2, dim=-1)
        B = z.size(0)

        sim = torch.matmul(z, z.t()) / self.temperature

        labels = labels.contiguous().view(-1, 1)
        pos_mask = torch.eq(labels, labels.t()).float().to(z.device)

        logits_mask = torch.ones_like(pos_mask) - torch.eye(B, device=z.device)
        pos_mask = pos_mask * logits_mask

        # numerical stability
        sim = sim - sim.max(dim=1, keepdim=True)[0].detach()

        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

        pos_count = pos_mask.sum(dim=1)
        valid = pos_count > 0

        if valid.sum() == 0:
            loss = z.sum() * 0.0
        else:
            mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / pos_count.clamp(min=1.0)
            loss = -mean_log_prob_pos[valid].mean()

        self.info.update({
            'loss': loss.detach().clone(),
            'valid_ratio': valid.float().mean().detach().clone()
        })

        return loss, self.info

class SemanticChannelReliabilityLoss(BaseLoss):
    """
    Branch-free, training-only semantic channel reliability loss.

    Inputs:
        group_feats: [B, M, D]
        labels:      [B]
        masks:       [B, M, D]

    It does not create a new ID branch.
    It does not modify inference features.
    """
    def __init__(
        self,
        loss_term_weight=1.0,
        align_weight=0.02,
        swap_weight=0.01,
        binary_weight=0.001,
        ratio_weight=0.01,
        target_keep_ratio=0.75,
        detach_target=True,
        eps=1e-6
    ):
        super(SemanticChannelReliabilityLoss, self).__init__(loss_term_weight)

        self.align_weight = align_weight
        self.swap_weight = swap_weight
        self.binary_weight = binary_weight
        self.ratio_weight = ratio_weight
        self.target_keep_ratio = target_keep_ratio
        self.detach_target = detach_target
        self.eps = eps

    @gather_and_scale_wrapper
    def forward(self, group_feats, labels, masks):
        """
        group_feats: [B, M, D]
        labels:      [B]
        masks:       [B, M, D]
        """
        group_feats = group_feats.float()
        masks = masks.float().clamp(0.0, 1.0)
        labels = labels.view(-1)

        B, M, D = group_feats.shape

        unique_labels, inverse = torch.unique(
            labels,
            sorted=True,
            return_inverse=True
        )

        C = unique_labels.numel()

        one_hot = F.one_hot(
            inverse,
            num_classes=C
        ).float().to(group_feats.device)

        counts = one_hot.sum(dim=0).clamp(min=1.0)  # [C]

        class_sum = torch.einsum(
            'bc,bmd->cmd',
            one_hot,
            group_feats
        )  # [C, M, D]

        # ---------------------------------------------------------
        # 1) Exclusive class prototype
        # ---------------------------------------------------------
        # For each sample, use same-ID samples except itself as target.
        # This avoids trivial self-alignment.
        count_per_sample = counts[inverse].view(B, 1, 1)

        proto_sum = class_sum[inverse] - group_feats
        proto_den = (count_per_sample - 1.0).clamp(min=1.0)

        target_proto = proto_sum / proto_den

        valid = (
            count_per_sample.squeeze(-1).squeeze(-1) > 1
        ).float()  # [B]

        if self.detach_target:
            target_proto = target_proto.detach()

        # ---------------------------------------------------------
        # 2) Masked semantic prototype alignment
        # ---------------------------------------------------------
        z_anchor = F.normalize(
            group_feats * masks,
            p=2,
            dim=-1,
            eps=self.eps
        )

        z_proto = F.normalize(
            target_proto * masks,
            p=2,
            dim=-1,
            eps=self.eps
        )

        align_loss = 1.0 - torch.sum(z_anchor * z_proto, dim=-1)  # [B, M]
        align_loss = align_loss * valid.view(B, 1)
        align_loss = align_loss.sum() / (valid.sum() * M + self.eps)

        # ---------------------------------------------------------
        # 3) Positive mask swapping
        # ---------------------------------------------------------
        # Use sample i's mask to filter another same-ID sample j.
        # This simulates cross-view visibility inconsistency.
        arange = torch.arange(B, device=group_feats.device)
        pos_idx = arange.clone()

        for c in range(C):
            idx = torch.nonzero(
                inverse == c,
                as_tuple=False
            ).view(-1)

            if idx.numel() > 1:
                pos_idx[idx] = idx.roll(shifts=-1)

        valid_swap = (pos_idx != arange).float()  # [B]

        pos_feats = group_feats[pos_idx]

        z_swap = F.normalize(
            pos_feats * masks,
            p=2,
            dim=-1,
            eps=self.eps
        )

        z_ref = F.normalize(
            group_feats.detach(),
            p=2,
            dim=-1,
            eps=self.eps
        )

        swap_loss = 1.0 - torch.sum(z_swap * z_ref, dim=-1)  # [B, M]
        swap_loss = swap_loss * valid_swap.view(B, 1)
        swap_loss = swap_loss.sum() / (valid_swap.sum() * M + self.eps)

        # ---------------------------------------------------------
        # 4) Binary regularization
        # ---------------------------------------------------------
        # Encourage masks to be close to 0 or 1 instead of staying around 0.5.
        binary_loss = (masks * (1.0 - masks)).mean()

        # ---------------------------------------------------------
        # 5) Keep-ratio regularization
        # ---------------------------------------------------------
        # Avoid all-one or all-zero masks.
        keep_ratio = masks.mean(dim=-1)  # [B, M]

        ratio_loss = (
            keep_ratio - self.target_keep_ratio
        ).pow(2).mean()

        loss = (
            self.align_weight * align_loss
            + self.swap_weight * swap_loss
            + self.binary_weight * binary_loss
            + self.ratio_weight * ratio_loss
        )

        self.info.update({
            'loss': loss.detach().clone(),
            # 'align': align_loss.detach().clone(),
            # 'swap': swap_loss.detach().clone(),
            # 'binary': binary_loss.detach().clone(),
            # 'ratio': ratio_loss.detach().clone(),
            'keep_ratio': masks.mean().detach().clone(),
            # 'mask_min': masks.min().detach().clone(),
            # 'mask_max': masks.max().detach().clone()
        })

        return loss, self.info

class SemanticRouterHeadDiversityLoss(BaseLoss):
    def __init__(self, loss_term_weight=1.0):
        super(SemanticRouterHeadDiversityLoss, self).__init__(loss_term_weight)

    def forward(self, attn_head, valid_axis_indices=None):
        """
        attn_head: [BT,H,M,N]
        """
        if valid_axis_indices is not None:
            attn_head = attn_head[:, :, valid_axis_indices, :]

        BT, H, M, N = attn_head.shape

        a = attn_head.reshape(BT, H, -1)
        a = F.normalize(a.float(), p=2, dim=-1)

        corr = torch.matmul(a, a.transpose(1, 2))  # [BT,H,H]
        eye = torch.eye(H, device=attn_head.device, dtype=corr.dtype).unsqueeze(0)

        loss = (corr - eye).pow(2).mean()

        self.info.update({
            'loss': loss.detach().clone(),
            # 'mean_offdiag': (corr - eye).abs().mean().detach().clone()
        })

        return loss, self.info

class SemanticIDObsOrthogonalLoss(BaseLoss):
    """
    Encourage identity semantic tokens and observation tokens to be orthogonal.

    Inputs:
        id_feats:  [B, Mi, D]
        obs_feats: [B, Mo, D]

    This is a lightweight regularizer for S2A-MSR.
    """
    def __init__(
        self,
        loss_term_weight=1.0,
        detach_obs=False,
        eps=1e-6
    ):
        super(SemanticIDObsOrthogonalLoss, self).__init__(loss_term_weight)
        self.detach_obs = detach_obs
        self.eps = eps

    @gather_and_scale_wrapper
    def forward(self, id_feats, obs_feats):
        if id_feats is None or obs_feats is None:
            loss = torch.tensor(0.0)
            self.info.update({"loss": loss.detach().clone()})
            return loss, self.info

        id_feats = id_feats.float()
        obs_feats = obs_feats.float()

        if self.detach_obs:
            obs_feats = obs_feats.detach()

        id_feats = F.normalize(id_feats, p=2, dim=-1, eps=self.eps)
        obs_feats = F.normalize(obs_feats, p=2, dim=-1, eps=self.eps)

        # [B, Mi, Mo]
        sim = torch.einsum("bid,bod->bio", id_feats, obs_feats)

        loss = sim.pow(2).mean()

        self.info.update({
            "loss": loss.detach().clone(),
            # "mean_abs_sim": sim.abs().mean().detach().clone(),
            # "max_abs_sim": sim.abs().max().detach().clone()
        })

        return loss, self.info

class GlobalMemoryLoss(BaseLoss):
    """
    Global prototype memory loss.

    Maintain momentum-updated identity prototypes across the dataset so the
    contrastive objective is not limited to negatives in the current batch.
    """
    def __init__(
        self,
        num_classes,
        feat_dim,
        temperature=0.05,
        momentum=0.2,
        loss_term_weight=1.0,
        log_accuracy=True,
        **kwargs
    ):
        super(GlobalMemoryLoss, self).__init__(loss_term_weight)

        self.num_classes = int(num_classes)
        self.feat_dim = int(feat_dim)
        self.temperature = float(temperature)
        self.momentum = float(momentum)
        self.log_accuracy = log_accuracy

        # Buffers are checkpointed but are not updated by the optimizer.
        self.register_buffer("global_memory", torch.zeros(self.num_classes, self.feat_dim))
        self.register_buffer("global_valid", torch.zeros(self.num_classes, dtype=torch.bool))

    def _prepare_embeddings(self, embeddings):
        """
        Convert OpenGait features from [B,C,P] to [B,C] when necessary.
        """
        embeddings = embeddings.float()
        if embeddings.dim() == 3:
            embeddings = embeddings.mean(dim=-1)
        # Map features to the unit hypersphere.
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

    @torch.no_grad()
    def _update_memory(self, embeddings, labels):
        """Update memory prototypes with features from the current batch."""
        embeddings = embeddings.detach()
        labels = labels.detach().long()

        unique_labels = torch.unique(labels)

        for y in unique_labels:
            y = int(y.item())
            if y < 0 or y >= self.num_classes:
                continue

            # Average all current-batch features for this identity.
            mask = (labels == y)
            feat = embeddings[mask].mean(dim=0)
            feat = F.normalize(feat, p=2, dim=0)

            # Initialize or momentum-update the identity prototype.
            if not self.global_valid[y]:
                self.global_memory[y] = feat
                self.global_valid[y] = True
            else:
                self.global_memory[y] = F.normalize(
                    self.momentum * self.global_memory[y] + (1.0 - self.momentum) * feat,
                    p=2, dim=0
                )

    @gather_and_scale_wrapper
    def forward(self, embeddings, labels):
        """
        embeddings: [B, C] or [B, C, P]
        labels:     [B]
        """
        labels = labels.long()

        # 1. Prepare features.
        embeddings = self._prepare_embeddings(embeddings)

        # 2. Momentum-update memory.
        self._update_memory(embeddings, labels)

        # 3. Select initialized memory prototypes.
        valid_ids = self.global_valid.nonzero(as_tuple=False).flatten()
        if valid_ids.numel() <= 1:
            loss = embeddings.new_tensor(0.0)
            self.info.update({'loss': loss.detach().clone()})
            return loss, self.info

        memory = self.global_memory[valid_ids]  # [M_valid, feat_dim]

        # 4. Compute InfoNCE logits against all initialized prototypes.
        logits = torch.matmul(embeddings, memory.t()) / self.temperature

        # 5. Build targets in the compact valid-prototype index space.
        target = torch.full_like(labels, -100)
        for i, y in enumerate(labels):
            pos = (valid_ids == y).nonzero(as_tuple=False)
            if pos.numel() > 0:
                target[i] = pos[0, 0]

        valid_mask = target >= 0
        if valid_mask.sum() == 0:
            loss = embeddings.new_tensor(0.0)
            self.info.update({'loss': loss.detach().clone()})
            return loss, self.info

        # 6. Compute the memory-based contrastive cross-entropy.
        loss = F.cross_entropy(logits[valid_mask], target[valid_mask])

        # 7. Update diagnostics.
        self.info.update({
            "loss": loss.detach().clone(),
            "valid_classes": self.global_valid.float().sum().detach().clone()
        })

        if self.log_accuracy:
            pred = logits[valid_mask].argmax(dim=1)
            accu = (pred == target[valid_mask]).float().mean()
            self.info.update({"accuracy": accu.detach().clone()})

        return loss, self.info

class VDTOrthogonalLoss(BaseLoss):
    """
    VDT orthogonal-decoupling loss for identity and view tokens.

    loss = |<t_m, t_v>| / (||t_m||_2 * ||t_v||_2)
    """
    def __init__(self, loss_term_weight=1.0):
        super(VDTOrthogonalLoss, self).__init__(loss_term_weight)

    @gather_and_scale_wrapper
    def forward(self, t_m, t_v):
        # L2-normalize both token types.
        t_m_norm = F.normalize(t_m.float(), p=2, dim=-1)
        t_v_norm = F.normalize(t_v.float(), p=2, dim=-1)

        # Penalize their absolute cosine similarity.
        cos_sim = torch.sum(t_m_norm * t_v_norm, dim=-1)
        loss = torch.abs(cos_sim).mean()

        self.info.update({
            'loss': loss.detach().clone()
        })
        return loss, self.info

class AxisSemanticContrastLoss(BaseLoss):
    """Align each routed semantic axis with its own text axis while contrasting other axes.

    Inputs:
        axis_feats: [B, M, D]
        text_feats: [M, D]
    """
    def __init__(self, loss_term_weight=1.0, temperature=0.07, detach_text=True):
        super(AxisSemanticContrastLoss, self).__init__(loss_term_weight)
        self.temperature = float(temperature)
        self.detach_text = bool(detach_text)

    def forward(self, axis_feats, text_feats):
        if axis_feats is None or text_feats is None or axis_feats.numel() == 0 or text_feats.numel() == 0:
            loss = torch.tensor(0.0, device=text_feats.device if text_feats is not None else axis_feats.device)
            self.info.update({'loss': loss.detach().clone()})
            return loss, self.info
        z = F.normalize(axis_feats.float(), p=2, dim=-1)
        t = text_feats.float()
        if self.detach_text:
            t = t.detach()
        t = F.normalize(t, p=2, dim=-1)
        B, M, D = z.shape
        if t.shape[0] != M:
            raise ValueError(f"text axis number {t.shape[0]} does not match feature axis number {M}")
        logits = torch.einsum('bmd,nd->bmn', z, t) / max(self.temperature, 1e-6)
        targets = torch.arange(M, device=z.device).view(1, M).expand(B, M).reshape(-1)
        loss = F.cross_entropy(logits.reshape(B * M, M), targets)
        pred = logits.argmax(dim=-1)
        acc = (pred == torch.arange(M, device=z.device).view(1, M)).float().mean()
        self.info.update({'loss': loss.detach().clone(), 'axis_acc': acc.detach().clone()})
        return loss, self.info


class AxisFeatureDiversityLoss(BaseLoss):
    """Prevent routed axes within the same role from collapsing to the same feature.

    A semantic-adaptive margin allows naturally related text axes to remain partially similar.
    Inputs:
        axis_feats: [B, M, D]
        text_feats: [M, D] optional
    """
    def __init__(self, loss_term_weight=1.0, base_margin=0.55, semantic_adaptive_margin=True,
                 semantic_margin_scale=0.20, detach_text=True):
        super(AxisFeatureDiversityLoss, self).__init__(loss_term_weight)
        self.base_margin = float(base_margin)
        self.semantic_adaptive_margin = bool(semantic_adaptive_margin)
        self.semantic_margin_scale = float(semantic_margin_scale)
        self.detach_text = bool(detach_text)

    def forward(self, axis_feats, text_feats=None):
        if axis_feats is None or axis_feats.numel() == 0:
            loss = torch.tensor(0.0, device=axis_feats.device)
            self.info.update({'loss': loss.detach().clone()})
            return loss, self.info
        z = F.normalize(axis_feats.float(), p=2, dim=-1)
        B, M, D = z.shape
        if M <= 1:
            loss = z.sum() * 0.0
            self.info.update({'loss': loss.detach().clone()})
            return loss, self.info
        sim = torch.matmul(z, z.transpose(1, 2))  # [B,M,M]
        eye = torch.eye(M, device=z.device, dtype=torch.bool).unsqueeze(0)
        margin = torch.full((M, M), self.base_margin, device=z.device, dtype=sim.dtype)
        if self.semantic_adaptive_margin and text_feats is not None and text_feats.numel() > 0:
            t = text_feats.float()
            if self.detach_text:
                t = t.detach()
            t = F.normalize(t, p=2, dim=-1)
            text_sim = torch.matmul(t, t.t()).clamp(min=0.0)
            margin = margin + self.semantic_margin_scale * text_sim
        margin = margin.unsqueeze(0)
        penalty = F.relu(sim - margin).pow(2).masked_fill(eye, 0.0)
        denom = B * M * (M - 1) + 1e-12
        loss = penalty.sum() / denom
        mean_offdiag = sim.masked_fill(eye, 0.0).abs().sum() / denom
        self.info.update({'loss': loss.detach().clone(), 'mean_abs_sim': mean_offdiag.detach().clone()})
        return loss, self.info


class FeatureConsistencyLoss(BaseLoss):
    """Cosine consistency between two final embedding tensors.

    Inputs:
        anchor:   factual embedding, usually detached. Shape [B,C,P], [B,P,D], or [B,D].
        positive: counterfactual embedding with the same shape as anchor.

    For OpenGait-style part embeddings [B,C,P], the feature/channel dimension is
    dim=1 by default, producing a per-part cosine distance and then averaging.
    """
    def __init__(
        self,
        loss_term_weight=1.0,
        detach_anchor=True,
        channel_dim=1,
        eps=1e-6,
    ):
        super(FeatureConsistencyLoss, self).__init__(loss_term_weight)
        self.detach_anchor = bool(detach_anchor)
        self.channel_dim = int(channel_dim)
        self.eps = float(eps)

    def forward(self, anchor, positive, valid_mask=None):
        if anchor is None or positive is None:
            device = anchor.device if anchor is not None else positive.device
            loss = torch.tensor(0.0, device=device)
            self.info.update({"loss": loss.detach().clone()})
            return loss, self.info

        a = anchor.float()
        p = positive.float()
        if a.shape != p.shape:
            raise ValueError(f"anchor shape {tuple(a.shape)} != positive shape {tuple(p.shape)}")

        if self.detach_anchor:
            a = a.detach()

        feat_dim = -1 if a.dim() == 2 else self.channel_dim
        if feat_dim < 0:
            feat_dim = a.dim() + feat_dim
        if feat_dim < 0 or feat_dim >= a.dim():
            raise ValueError(f"Invalid channel_dim={self.channel_dim} for tensor shape {tuple(a.shape)}")

        a = F.normalize(a, p=2, dim=feat_dim, eps=self.eps)
        p = F.normalize(p, p=2, dim=feat_dim, eps=self.eps)
        dist = 1.0 - torch.sum(a * p, dim=feat_dim)

        valid_ratio = dist.new_tensor(1.0)
        if valid_mask is not None:
            mask = valid_mask.to(dist.device).float().view(dist.shape[0], *([1] * (dist.dim() - 1)))
            loss = (dist * mask).sum() / (mask.sum() * dist[0].numel() + self.eps)
            valid_ratio = mask.mean()
        else:
            loss = dist.mean()

        self.info.update({
            "loss": loss.detach().clone(),
            "valid_ratio": valid_ratio.detach().clone(),
        })
        return loss, self.info


class RouterCounterfactualConsistencyLoss(BaseLoss):
    """Keep ID semantic axes stable under Router-level view intervention.

    Inputs:
        factual:        [B,T,M,D] or [B,M,D]
        counterfactual: [B,T,M,D] or [B,M,D]
        seqL: optional sequence length for masking [B]

    The counterfactual sample keeps image/text content unchanged but changes the
    view-conditioned Router bias. This loss is therefore a local invariance
    constraint on ID semantic axes, not an adversarial removal of view from the
    final embedding.
    """
    def __init__(
        self,
        loss_term_weight=1.0,
        detach_factual=True,
        eps=1e-6,
        reduction="cosine",
    ):
        super(RouterCounterfactualConsistencyLoss, self).__init__(loss_term_weight)
        self.detach_factual = bool(detach_factual)
        self.eps = float(eps)
        self.reduction = str(reduction)

    def forward(self, factual, counterfactual, seqL=None):
        if factual is None or counterfactual is None:
            device = factual.device if factual is not None else counterfactual.device
            loss = torch.tensor(0.0, device=device)
            self.info.update({"loss": loss.detach().clone()})
            return loss, self.info

        f = factual.float()
        c = counterfactual.float()
        if f.shape != c.shape:
            raise ValueError(f"factual shape {tuple(f.shape)} != counterfactual shape {tuple(c.shape)}")

        if self.detach_factual:
            f = f.detach()

        f = F.normalize(f, p=2, dim=-1, eps=self.eps)
        c = F.normalize(c, p=2, dim=-1, eps=self.eps)
        dist = 1.0 - torch.sum(f * c, dim=-1)  # [B,T,M] or [B,M]

        valid_ratio = dist.new_tensor(1.0)
        if dist.dim() == 3 and seqL is not None:
            B, T, M = dist.shape
            seq_t = seqL.to(dist.device) if torch.is_tensor(seqL) else torch.tensor(seqL, device=dist.device)
            seq_t = seq_t.view(-1).long()
            mask = torch.arange(T, device=dist.device).unsqueeze(0).expand(B, T) < seq_t.unsqueeze(1)
            mask = mask.unsqueeze(-1).float()
            loss = (dist * mask).sum() / (mask.sum() * M + self.eps)
            valid_ratio = mask.mean()
        else:
            loss = dist.mean()

        # print(f"router_cf_consistency_loss={loss.item():.8e}")
        self.info.update({
            "loss": loss.detach().clone(),
            "valid_ratio": valid_ratio.detach().clone(),
        })
        return loss, self.info


class CrossViewSemanticAxisConsistencyLoss(BaseLoss):
    """Pull same-ID semantic axes together across different platforms/views.

    Inputs:
        axis_feats: [B,M,D], usually video-level semantic axis features before
                    axis dropout.
        labels:     [B]
        platforms:  [B], parsed platform labels. If no cross-platform positive
                    exists in a batch, the sample is ignored.

    This is intentionally local to the semantic-axis space, avoiding strong GRL
    constraints on the final ID embedding.
    """
    def __init__(
        self,
        loss_term_weight=1.0,
        detach_target=True,
        eps=1e-6,
    ):
        super(CrossViewSemanticAxisConsistencyLoss, self).__init__(loss_term_weight)
        self.detach_target = bool(detach_target)
        self.eps = float(eps)

    @gather_and_scale_wrapper
    def forward(self, axis_feats, labels, platforms):
        z = axis_feats.float()
        labels = labels.view(-1)
        platforms = platforms.view(-1)

        if z.dim() != 3:
            raise ValueError(f"Expected axis_feats [B,M,D], got {tuple(z.shape)}")

        z = F.normalize(z, p=2, dim=-1, eps=self.eps)
        B, M, D = z.shape

        losses = []
        valid_count = 0
        for i in range(B):
            pos = (labels == labels[i]) & (platforms != platforms[i])
            pos[i] = False
            idx = torch.nonzero(pos, as_tuple=False).view(-1)
            if idx.numel() == 0:
                continue

            target = z[idx].mean(dim=0)
            target = F.normalize(target, p=2, dim=-1, eps=self.eps)
            if self.detach_target:
                target = target.detach()

            sim = torch.sum(z[i] * target, dim=-1)  # [M]
            losses.append(1.0 - sim.mean())
            valid_count += 1

        if valid_count == 0:
            loss = z.sum() * 0.0
            valid_ratio = z.new_tensor(0.0)
        else:
            loss = torch.stack(losses).mean()
            valid_ratio = z.new_tensor(float(valid_count) / float(B))

        self.info.update({
            "loss": loss.detach().clone(),
            "valid_ratio": valid_ratio.detach().clone(),
        })
        return loss, self.info
