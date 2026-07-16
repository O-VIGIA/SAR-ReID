import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "opengait/modeling/models/star_cvi_texts.py"


def load_module():
    spec = importlib.util.spec_from_file_location("star_cvi_texts", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_human_basis_24_plus_context_axes():
    module = load_module()
    dictionary = module.build_semantic_dictionary(
        profile="ag_vpreid_human_basis24_ctx3_v1",
        include_context=True,
    )
    names = dictionary.get_group_names()
    prompts = dictionary.get_group_prompts("non_view")

    assert len(names) == 27
    assert len([name for name in names if name.startswith("id_")]) == 24
    assert len([name for name in names if name.startswith("ctx_")]) == 3
    assert len(prompts) == len(names)
    assert all(len(group) == 1 and group[0].strip() for group in prompts)


def test_dictionary_rejects_multiple_descriptions():
    module = load_module()
    try:
        module.build_semantic_dictionary(num_descriptions_per_group_non_view=2)
    except ValueError as exc:
        assert "exactly one description" in str(exc)
    else:
        raise AssertionError("Expected a ValueError for multiple descriptions")

