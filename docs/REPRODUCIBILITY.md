# Reproducibility notes

## Canonical release configuration

The primary experiment is `configs/star_cvi_ag_vpreid.yaml`:

- CLIP ViT-B/16, input `192×96`, patch stride `12×12`
- Human-Basis-24 identity axes plus 3 context axes
- 4 identities × 4 tracklets, 10 ordered frames per tracklet
- AdamW, base LR `1e-5`, weight decay `5e-4`, betas `(0.9, 0.999)`
- LR milestones at 15k and 30k; 40k total iterations
- image encoder / prompt / new modules use `1×` / `2×` / `3×` LR
- evaluator checkpoint and trainer schedule both target iteration 40000

The released OpenGait integration target is commit
`0efafd4779f127fbce34f22aff301bd82e923da5`.

## Engineering corrections made during packaging

The supplied research files were normalized for a public repository without
changing the semantic router, temporal interaction, R-CVI, or loss definitions:

1. Removed workstation-specific absolute dataset paths.
2. Matched evaluator checkpoint iteration and save name to the 40k trainer.
3. Made parameter initialization idempotent so OpenGait's post-build
   initialization does not overwrite pretrained CLIP weights.
4. Connected the existing three-way parameter grouping to `get_optimizer`;
   previously `get_param_groups` was defined but not called by stock OpenGait.
5. Made RGB input handling accept both channel-first and channel-last tracklets.
6. Added the missing OpenAI CLIP tokenizer and BPE vocabulary with its license.
7. Added an additive component registry so custom transforms and the evaluator
   are available without overwriting OpenGait core modules.
8. Preserved the supplied CE/Triplet variants as optional, non-default files.

## Important provenance limitation

The supplied attachments referenced `SequenceAware*` transforms and
`evaluate_UAV_Ground`, but their original source files were not included. This
repository supplies compatible, documented implementations inferred from the
configuration and the official AG-VPReID camera protocol:

- photometric/geometric augmentation parameters are shared across each sequence;
- `ClipRgbTransform` emits `[T,3,192,96]` with official CLIP normalization;
- `C0`-`C3` are ground cameras and `C4`-`C5` are aerial cameras;
- within-platform evaluation excludes same-identity/same-camera matches.

If the experiments used different local implementations, replace these two
files with the original versions before claiming bit-for-bit numerical
reproduction:

- `opengait/data/star_cvi_transform.py`
- `opengait/evaluation/star_cvi_evaluator.py`

## Assets not included

- AG-VPReID images or videos
- the experiment-specific `AG_VPReID.json` partition
- CLIP ViT-B/16 weights
- STAR-CVI checkpoints
- training logs and random-state snapshots

The dataset converter generates an OpenGait partition from the supplied train
and test roots. Confirm that this partition matches the paper split before
reporting results.

## Validation levels

### Included in this repository

```bash
python scripts/validate_repo.py
pytest -q
```

These checks validate syntax, configuration consistency, semantic-axis counts,
tokenizer resources, private paths, and documentation links without a GPU.

### Required for a reproducibility release

1. Record Python, CUDA, cuDNN, PyTorch, torchvision, GPU, and driver versions.
2. Save the exact OpenGait and STAR-CVI commits.
3. Record the dataset split checksum and identity/tracklet counts.
4. Train the 40k schedule from iteration 0 with the published config.
5. Evaluate the iteration-40000 checkpoint under all four protocols.
6. Compare results against the manuscript table and archive raw logs.

The current packaging pass did not perform full GPU training or numerical
evaluation.

## Baseline note

The supplied DeepGaitV2 config targets RGB `192×96` inputs. Stock OpenGait's
DeepGaitV2 is silhouette-oriented and, in the checked upstream revision, asserts
an input width of 44 or 88. The baseline config is retained for experiment
provenance, but its corresponding RGB-adapted model implementation was not among
the supplied files.

