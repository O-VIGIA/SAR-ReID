#!/usr/bin/env python3
"""Run lightweight, GPU-free release checks for the STAR-CVI repository."""

import ast
import gzip
import re
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
MAIN_CONFIG = ROOT / "configs" / "star_cvi_ag_vpreid.yaml"
CLIP_CONFIG = ROOT / "opengait" / "modeling" / "model_clip" / "config_clip" / "cfg.yaml"


def require(condition, message, errors):
    if not condition:
        errors.append(message)


def check_required_files(errors):
    required = [
        "README.md",
        "LICENSE",
        "CITATION.cff",
        "requirements.txt",
        "configs/star_cvi_ag_vpreid.yaml",
        "opengait/modeling/models/star_cvi.py",
        "opengait/modeling/models/star_cvi_texts.py",
        "opengait/modeling/losses/star_cvi_losses.py",
        "opengait/modeling/model_clip/clip/simple_tokenizer.py",
        "opengait/modeling/model_clip/clip/bpe_simple_vocab_16e6.txt.gz",
        "opengait/data/star_cvi_transform.py",
        "opengait/evaluation/star_cvi_evaluator.py",
    ]
    for relative in required:
        require((ROOT / relative).is_file(), f"Missing required file: {relative}", errors)


def check_python(errors):
    for path in sorted(ROOT.rglob("*.py")):
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError) as exc:
            errors.append(f"Python parse failure in {path.relative_to(ROOT)}: {exc}")


def check_configs(errors):
    try:
        main_cfg = yaml.safe_load(MAIN_CONFIG.read_text(encoding="utf-8"))
        clip_cfg = yaml.safe_load(CLIP_CONFIG.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"YAML parse failure: {exc}")
        return

    require(main_cfg["model_cfg"]["model"] == "STARCVI", "Main model must be STARCVI", errors)
    require(
        main_cfg["trainer_cfg"]["save_name"] == main_cfg["evaluator_cfg"]["save_name"],
        "Trainer/evaluator save_name mismatch",
        errors,
    )
    require(
        main_cfg["trainer_cfg"]["total_iter"] == main_cfg["evaluator_cfg"]["restore_hint"],
        "Evaluator restore_hint must match total_iter in the release config",
        errors,
    )
    require(
        main_cfg["model_cfg"]["SeparateFCs"]["parts_num"] == 25,
        "Human-Basis-24 requires 25 retrieval parts (global + 24 axes)",
        errors,
    )
    require(
        main_cfg["model_cfg"]["SeparateBNNecks"]["parts_num"] == 25,
        "SeparateBNNecks parts_num must match SeparateFCs",
        errors,
    )
    require(
        clip_cfg["INPUT"]["SIZE_TRAIN"] == [192, 96],
        "CLIP input size must be 192x96 for the released config",
        errors,
    )
    require(
        clip_cfg["MODEL"]["STRIDE_SIZE"] == [12, 12],
        "CLIP stride must be 12x12 for the released config",
        errors,
    )


def check_resources(errors):
    bpe_path = ROOT / "opengait/modeling/model_clip/clip/bpe_simple_vocab_16e6.txt.gz"
    if bpe_path.is_file():
        try:
            with gzip.open(bpe_path, "rt", encoding="utf-8") as handle:
                header = handle.readline().strip()
            require("#version" in header, "CLIP BPE vocabulary has an invalid header", errors)
        except OSError as exc:
            errors.append(f"Cannot read CLIP BPE vocabulary: {exc}")


def check_private_paths(errors):
    forbidden = re.compile(r"(?:/home/[^/]+|/Users/[^/]+|[A-Za-z]:\\\\Users\\\\[^\\]+)")
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".py", ".yaml", ".yml", ".md", ".txt", ".sh"}:
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        text = path.read_text(encoding="utf-8")
        if forbidden.search(text):
            errors.append(f"Potential private absolute path in {path.relative_to(ROOT)}")


def check_readme_links(errors):
    readme = ROOT / "README.md"
    if not readme.is_file():
        return
    text = readme.read_text(encoding="utf-8")
    for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
        target = target.strip().split("#", 1)[0]
        if not target or re.match(r"^(?:https?://|mailto:)", target):
            continue
        require((ROOT / target).exists(), f"Broken README link: {target}", errors)


def main():
    errors = []
    check_required_files(errors)
    check_python(errors)
    check_configs(errors)
    check_resources(errors)
    check_private_paths(errors)
    check_readme_links(errors)
    if errors:
        print("Repository validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    python_count = len(list(ROOT.rglob("*.py")))
    print(f"Repository validation passed ({python_count} Python files checked).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
