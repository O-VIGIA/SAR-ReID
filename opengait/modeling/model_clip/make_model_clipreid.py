import torch
import torch.nn as nn
import numpy as np
import os
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
from .clip import clip

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x

class HierarchicalPromptLearner(nn.Module):
    def __init__(self, dtype: torch.dtype, token_embedding: nn.Module, num_groups: int,
                 n_ctx_shared: int = 4, n_ctx_group: int = 2,
                 template_prefix: str = "a photo of", template_suffix: str = ".",
                 context_position: str = "after_prefix", learner_name: str = "prompt"):
        super().__init__()
        self.dtype = dtype
        self.token_embedding = token_embedding
        self.n_ctx_shared = n_ctx_shared
        self.n_ctx_group = n_ctx_group
        self.total_ctx = n_ctx_shared + n_ctx_group
        self.template_prefix = template_prefix.strip()
        self.template_suffix = template_suffix
        self.context_position = context_position
        self.learner_name = learner_name

        ctx_dim = token_embedding.weight.shape[1]

        # Register learnable context only when its configured length is nonzero.
        if self.n_ctx_shared > 0:
            self.ctx_shared = nn.Parameter(torch.empty(n_ctx_shared, ctx_dim, dtype=dtype))
            nn.init.normal_(self.ctx_shared, std=0.02)
        else:
            self.register_parameter('ctx_shared', None)

        if self.n_ctx_group > 0:
            self.ctx_group = nn.Parameter(torch.empty(num_groups, n_ctx_group, ctx_dim, dtype=dtype))
            nn.init.normal_(self.ctx_group, std=0.02)
        else:
            self.register_parameter('ctx_group', None)

        # Compute the prefix length safely, including an empty-prefix config.
        if self.template_prefix:
            prefix_tokens = clip.tokenize([self.template_prefix])
            self.prefix_token_count = int((prefix_tokens[0] != 0).sum().item()) - 2
        else:
            self.prefix_token_count = 0

    def forward(self, descriptions: list, group_idx: int):
        if not descriptions:
            descriptions = ["a person"]

        device = self.token_embedding.weight.device
        K = len(descriptions)

        # Avoid a leading space when the template prefix is empty.
        texts = [
            f"{self.template_prefix} {desc}{self.template_suffix}".lstrip()
            for desc in descriptions
        ]

        tokenized = clip.tokenize(texts).to(device)

        with torch.no_grad():
            base_embedding = self.token_embedding(tokenized).type(self.dtype)

        if self.context_position == "after_prefix":
            prefix_len = 1 + self.prefix_token_count
            prefix = base_embedding[:, :prefix_len, :]

            # Derive the sequence length from the embedding rather than hard-coding 77.
            max_seq_len = base_embedding.shape[1]
            suffix = base_embedding[:, prefix_len:(max_seq_len - self.total_ctx), :]

            # Assemble only the enabled context components.
            prompts_parts = [prefix]

            if self.n_ctx_shared > 0:
                shared_ctx = self.ctx_shared.unsqueeze(0).expand(K, -1, -1)
                prompts_parts.append(shared_ctx)

            if self.n_ctx_group > 0:
                group_ctx = self.ctx_group[group_idx].unsqueeze(0).expand(K, -1, -1)
                prompts_parts.append(group_ctx)

            prompts_parts.append(suffix)

            prompts = torch.cat(prompts_parts, dim=1)
        else:
            raise ValueError(f"Unsupported context_position: {self.context_position}")

        return prompts, tokenized


class build_transformer(nn.Module):
    def __init__(self, cfg, num_groups=11):
        super(build_transformer, self).__init__()
        self.model_name = cfg.MODEL.NAME

        if self.model_name == 'ViT-B-16':
            self.in_planes = 768
            self.in_planes_proj = 512
        else:
            raise ValueError(f"Unsupported CLIP model: {self.model_name}")

        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0] - 16) // cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1] - 16) // cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = (cfg.MODEL.STRIDE_SIZE[0], cfg.MODEL.STRIDE_SIZE[1])

        # --- Prompt config ---
        non_view_prompt_cfg = getattr(cfg.MODEL, "NON_VIEW_PROMPT", None)
        self.non_view_n_ctx_shared = int(non_view_prompt_cfg.N_CTX_SHARED)
        self.non_view_n_ctx_group = int(non_view_prompt_cfg.N_CTX_GROUP)
        self.non_view_template_prefix = str(non_view_prompt_cfg.TEMPLATE_PREFIX)
        self.non_view_template_suffix = str(non_view_prompt_cfg.TEMPLATE_SUFFIX)
        self.non_view_context_position = str(non_view_prompt_cfg.CONTEXT_POSITION)

        self.dynamic_weighting_enable = getattr(non_view_prompt_cfg, "DYNAMIC_WEIGHTING", False)
        self.num_descriptions = int(non_view_prompt_cfg.NUM_DESCRIPTIONS_PER_GROUP)

        if self.dynamic_weighting_enable and self.num_descriptions > 1:
            self.prompt_weights = nn.Parameter(torch.zeros(num_groups, self.num_descriptions))
            print("[*] Attribute-guided prompt weighting enabled (uniform initialization).")
        else:
            self.prompt_weights = None
            print("[*] Attribute-guided prompt weighting disabled; using mean pooling.")

        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)

        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)

        self.prompt_learner = HierarchicalPromptLearner(
            clip_model.dtype, clip_model.token_embedding, num_groups=num_groups,
            n_ctx_shared=self.non_view_n_ctx_shared, n_ctx_group=self.non_view_n_ctx_group,
            template_prefix=self.non_view_template_prefix, template_suffix=self.non_view_template_suffix,
            context_position=self.non_view_context_position, learner_name="semantic_prompt"
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07), requires_grad=False)
        self.group_prompts_non_view_list = None

    def setup_semantic_groups(self, group_prompts_non_view=None):
        self.group_prompts_non_view_list = group_prompts_non_view

    def _encode_group_prompts(self, group_prompts_list, prompt_learner):
        assert group_prompts_list is not None, "Prompt groups are not initialized."
        group_features = []

        # Compute prompt weights only when dynamic weighting is enabled.
        if self.prompt_weights is not None:
            attn_weights = torch.softmax(self.prompt_weights, dim=-1)  # [17, 5]

        for group_idx, descriptions in enumerate(group_prompts_list):
            text_embeddings, text_tokens = prompt_learner(descriptions, group_idx=group_idx)
            feats = self.text_encoder(text_embeddings, text_tokens)   # [K, D]

            # Select dynamic prompt weighting or parameter-free mean pooling.
            if self.prompt_weights is not None and len(descriptions) > 1:
                w = attn_weights[group_idx].unsqueeze(-1) # [K, 1]
                weighted_feats = (feats * w).sum(dim=0, keepdim=True)
            else:
                weighted_feats = feats.mean(dim=0, keepdim=True)

            # L2-normalize the group prototype.
            weighted_feats = weighted_feats / weighted_feats.norm(dim=-1, keepdim=True)
            group_features.append(weighted_feats)

        return torch.cat(group_features, dim=0)   # [M, D]

    def forward(self, image_feats=None, get_image=False, get_text=False):
        if get_text:
            text_feats_non_view = self._encode_group_prompts(self.group_prompts_non_view_list, self.prompt_learner)
            return text_feats_non_view

        elif get_image:
            image_features_last, image_features, image_features_proj = self.image_encoder(image_feats)
            cls_token = image_features_proj[:, 0:1, :]    # [B,1,D]
            patch_tokens = image_features_proj[:, 1:, :]  # [B,N,D]
            return cls_token, patch_tokens
        else:
            raise ValueError("You must specify either get_image=True or get_text=True !!!")

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

def make_model(cfg, num_groups=11):
    return build_transformer(cfg, num_groups=num_groups)

def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    local_model_path = os.path.join(current_dir, f"{backbone_name}.pt")

    if os.path.exists(local_model_path):
        print(">>>>>>>>>> Loading local model from: " + local_model_path)
        model_path = local_model_path
    else:
        print(f">>>>>>>>>> Local model '{local_model_path}' not found. Downloading...")
        url = clip._MODELS[backbone_name]
        model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)
    return model
