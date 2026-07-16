class AnatomicalPromptGenerator:
    """Human-Basis semantic dictionary for Aerial-Ground video ReID.

    This is a clean text-side coordinate system for from-scratch training.

    Design principles:
    - Remove legacy ID20/CTX10 prompt profiles and old prompt variants.
    - Keep only the new Human-Basis person-semantic axes and 3 broad nuisance
      context anchors.
    - Do not introduce manually assigned reliability types.
    - Keep the ``id_`` prefix for person/retrieval axes and ``ctx_`` for
      nuisance/context axes, because PointGait splits branches by prefix.
    - Axis count is profile-dependent. PointGait dynamically sets
      ``parts_num = 1 + num_id_axes`` before building SeparateFCs/BNNecks.
    """

    ID_GROUP_KEYS_HUMAN_BASIS16 = [
        "id_full_body_silhouette",
        "id_head_shoulder_contour",
        "id_torso_upper_body",
        "id_arm_hand_region",
        "id_leg_foot_region",
        "id_body_proportion",
        "id_upper_lower_layout",
        "id_posture_body_pose",
        "id_upper_body_appearance",
        "id_lower_body_appearance",
        "id_clothing_color_layout",
        "id_clothing_pattern_texture",
        "id_footwear_appearance",
        "id_backpack_bag",
        "id_carried_object",
        "id_hair_head_accessory",
    ]

    ID_GROUP_KEYS_HUMAN_BASIS24 = [
        "id_full_body_silhouette",
        "id_body_outline",
        "id_head_shoulder_contour",
        "id_head_detail",
        "id_torso_structure",
        "id_shoulder_width",
        "id_arm_hand_region",
        "id_leg_region",
        "id_foot_region",
        "id_body_proportion",
        "id_upper_lower_layout",
        "id_posture_body_pose",
        "id_upper_body_appearance",
        "id_lower_body_appearance",
        "id_sleeve_appearance",
        "id_pants_skirt_shape",
        "id_clothing_color_layout",
        "id_clothing_pattern_texture",
        "id_outerwear_appearance",
        "id_footwear_appearance",
        "id_backpack_rear_object",
        "id_handbag_side_bag",
        "id_carried_object",
        "id_visible_accessories",
    ]

    CONTEXT_GROUP_KEYS_CTX3 = [
        "ctx_view_platform_scale",
        "ctx_imaging_quality",
        "ctx_scene_interference",
    ]

    SUPPORTED_PROFILES = {
        "ag_vpreid_human_basis16_ctx3_v1",
        "ag_vpreid_human_basis24_ctx3_v1",
    }

    DEFAULT_PROFILE = "ag_vpreid_human_basis24_ctx3_v1"

    def __init__(
        self,
        num_descriptions_per_group_non_view=1,
        granularity=None,
        include_context=True,
        profile=DEFAULT_PROFILE,
        depersonalize_axis_prompt=True,
        **kwargs,
    ):
        """Build the semantic dictionary.

        ``granularity`` and ``depersonalize_axis_prompt`` are kept in the
        signature for config/API compatibility, but they are intentionally unused
        in the clean Human-Basis dictionary.
        """
        if int(num_descriptions_per_group_non_view) != 1:
            raise ValueError("This dictionary version supports exactly one description per group.")
        if profile not in self.SUPPORTED_PROFILES:
            raise ValueError(
                f"Unsupported semantic dictionary profile: {profile}. "
                f"Supported profiles: {sorted(self.SUPPORTED_PROFILES)}"
            )

        self.profile = str(profile)
        self.include_context = bool(include_context)

        self.groups_keys = list(self._get_id_keys())
        if self.include_context:
            self.groups_keys += list(self.CONTEXT_GROUP_KEYS_CTX3)

        self._validate_group_keys(self.groups_keys)

        self.prompt_bank = {"non_view": self._build_prompt_bank()}
        missing = [k for k in self.groups_keys if k not in self.prompt_bank["non_view"]]
        if missing:
            raise ValueError(f"Missing prompt definitions for groups: {missing}")

    @staticmethod
    def _validate_group_keys(keys):
        duplicates = sorted({k for k in keys if keys.count(k) > 1})
        if duplicates:
            raise ValueError(f"Duplicated semantic group keys: {duplicates}")

        invalid = [
            k for k in keys
            if not (str(k).startswith("id_") or str(k).startswith("ctx_"))
        ]
        if invalid:
            raise ValueError(
                "Semantic group keys must start with 'id_' or 'ctx_' because "
                f"PointGait splits branches by prefix. Invalid keys: {invalid}"
            )

    def _get_id_keys(self):
        if self.profile == "ag_vpreid_human_basis16_ctx3_v1":
            return self.ID_GROUP_KEYS_HUMAN_BASIS16
        if self.profile == "ag_vpreid_human_basis24_ctx3_v1":
            return self.ID_GROUP_KEYS_HUMAN_BASIS24
        raise RuntimeError(f"Unexpected profile after validation: {self.profile}")

    @staticmethod
    def _human_basis_prompts():
        """CLIP-friendly visual phrases for person-semantic evidence axes.

        The prompts intentionally avoid abstract wording such as "invariant
        identity cue". They use concrete visual nouns and attributes that CLIP is
        more likely to align with image regions.
        """
        return {
            # Shared by HumanBasis-16 and HumanBasis-24.
            "id_full_body_silhouette": [
                "visible full body silhouette and overall body outline"
            ],
            "id_head_shoulder_contour": [
                "head shoulder contour neck region and upper body boundary"
            ],
            "id_arm_hand_region": [
                "arms hands side body contour sleeve region and hand appearance"
            ],
            "id_body_proportion": [
                "height width ratio limb proportion and overall body build"
            ],
            "id_upper_lower_layout": [
                "spatial layout between upper body lower body arms and legs"
            ],
            "id_posture_body_pose": [
                "walking posture body pose gait-related silhouette and body orientation"
            ],
            "id_upper_body_appearance": [
                "upper body appearance upper garment color texture shape and sleeve area"
            ],
            "id_lower_body_appearance": [
                "lower body appearance lower garment color texture shape and leg area"
            ],
            "id_clothing_color_layout": [
                "color arrangement across upper clothing lower clothing and body regions"
            ],
            "id_clothing_pattern_texture": [
                "stripes logos fabric texture clothing pattern and local visual details"
            ],
            "id_footwear_appearance": [
                "shoe boot footwear color shape and lower foot appearance"
            ],
            "id_carried_object": [
                "hand-carried item object close to the body or object held near the hands"
            ],

            # HumanBasis-16 broader groups.
            "id_torso_upper_body": [
                "torso width upper body shape shoulder area and body trunk structure"
            ],
            "id_leg_foot_region": [
                "legs feet lower limb contour stride region and foot placement"
            ],
            "id_backpack_bag": [
                "backpack bag rear-carried or side-carried object appearance"
            ],
            "id_hair_head_accessory": [
                "hair shape visible head detail hat headwear or head accessory appearance"
            ],

            # HumanBasis-24 finer groups.
            "id_body_outline": [
                "outer body boundary body contour and visible person outline"
            ],
            "id_head_detail": [
                "hair shape visible head detail hat hood helmet and head boundary"
            ],
            "id_torso_structure": [
                "torso width body trunk shape and upper body structural outline"
            ],
            "id_shoulder_width": [
                "shoulder width upper body mass distribution and shoulder line"
            ],
            "id_leg_region": [
                "legs lower limb contour stride-region shape and lower body structure"
            ],
            "id_foot_region": [
                "feet foot placement lower foot boundary and shoe-adjacent region"
            ],
            "id_sleeve_appearance": [
                "sleeve length sleeve color arm clothing and arm-region appearance"
            ],
            "id_pants_skirt_shape": [
                "pants skirt dress lower garment silhouette and lower clothing shape"
            ],
            "id_outerwear_appearance": [
                "coat jacket hoodie outer layer shape color and texture appearance"
            ],
            "id_backpack_rear_object": [
                "backpack rear-carried bag or object attached to the back"
            ],
            "id_handbag_side_bag": [
                "handbag shoulder bag crossbody bag side-carried bag and strap appearance"
            ],
            "id_visible_accessories": [
                "glasses scarf belt small wearable accessories and persistent visible details"
            ],
        }

    @staticmethod
    def _ctx3_prompts():
        """Broad nuisance anchors for aerial-ground observation conditions."""
        return {
            "ctx_view_platform_scale": [
                "aerial or ground camera viewpoint elevation viewing direction target scale and capture platform"
            ],
            "ctx_imaging_quality": [
                "image blur noise compression exposure illumination weather visibility and sensor quality"
            ],
            "ctx_scene_interference": [
                "background clutter surrounding objects occlusion crowding truncation and scene interference"
            ],
        }

    def _build_prompt_bank(self):
        prompts = {}
        prompts.update(self._human_basis_prompts())
        prompts.update(self._ctx3_prompts())
        return prompts

    def get_group_prompts(self, view="non_view"):
        if view != "non_view":
            raise ValueError(f"Unsupported prompt type: {view}")
        return [self.prompt_bank["non_view"][k] for k in self.groups_keys]

    def get_group_names(self):
        return list(self.groups_keys)

    def get_group_roles(self):
        return ["context" if k.startswith("ctx_") else "id" for k in self.groups_keys]

    def get_id_group_names(self):
        return [k for k in self.groups_keys if k.startswith("id_")]

    def get_context_group_names(self):
        return [k for k in self.groups_keys if k.startswith("ctx_")]


def build_semantic_dictionary(
    num_descriptions_per_group_non_view=1,
    granularity=None,
    include_context=True,
    profile=AnatomicalPromptGenerator.DEFAULT_PROFILE,
    depersonalize_axis_prompt=True,
    **kwargs,
):
    return AnatomicalPromptGenerator(
        num_descriptions_per_group_non_view=num_descriptions_per_group_non_view,
        granularity=granularity,
        include_context=include_context,
        profile=profile,
        depersonalize_axis_prompt=depersonalize_axis_prompt,
        **kwargs,
    )
