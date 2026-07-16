from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_yaml(relative):
    return yaml.safe_load((ROOT / relative).read_text(encoding="utf-8"))


def test_release_schedule_and_checkpoint_match():
    config = load_yaml("configs/star_cvi_ag_vpreid.yaml")
    assert config["trainer_cfg"]["total_iter"] == 40000
    assert config["evaluator_cfg"]["restore_hint"] == 40000
    assert config["trainer_cfg"]["save_name"] == config["evaluator_cfg"]["save_name"]


def test_release_camera_protocol_and_parts():
    config = load_yaml("configs/star_cvi_ag_vpreid.yaml")
    assert config["evaluator_cfg"]["aerial_platforms"] == ["C4", "C5"]
    assert config["evaluator_cfg"]["ground_platforms"] == ["C0", "C1", "C2", "C3"]
    assert config["model_cfg"]["SeparateFCs"]["parts_num"] == 25
    assert config["model_cfg"]["SeparateBNNecks"]["parts_num"] == 25


def test_clip_geometry_matches_model_input():
    config = load_yaml("opengait/modeling/model_clip/config_clip/cfg.yaml")
    assert config["INPUT"]["SIZE_TRAIN"] == [192, 96]
    assert config["MODEL"]["STRIDE_SIZE"] == [12, 12]

