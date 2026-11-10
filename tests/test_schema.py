import json
from pathlib import Path

import pytest
from runpod.serverless.utils.rp_validator import validate

from sprite_inpaint.schema import FIELDS, json_schema, runpod_schema, validate_input

ROOT = Path(__file__).resolve().parent.parent


def test_committed_json_schema_matches_python_source():
    committed = json.loads((ROOT / "schemas" / "input.schema.json").read_text())
    assert committed == json_schema(), "run: PYTHONPATH=src python -m sprite_inpaint.schema > schemas/input.schema.json"


def test_minimal_input_gets_defaults():
    out = validate({"image": "x", "mask": "y", "prompt": "fix arm"}, runpod_schema())
    assert "errors" not in out, out
    v = out["validated_input"]
    for f in FIELDS:
        if not f.required:
            assert v[f.name] == f.default
    assert isinstance(v["cfg"], float)


def test_int_is_accepted_for_float_fields():
    values, _ = validate_input({"image": "x", "mask": "y", "prompt": "p", "cfg": 2, "denoise": 1})
    assert values["cfg"] == 2.0 and values["denoise"] == 1.0


@pytest.mark.parametrize("extra", [
    {"bogus": 1},
    {"steps": 0},
    {"steps": 8.5},
    {"seed": -2},
    {"seed": True},
    {"alpha_mode": "magic"},
    {"matte_color": "black"},
    {"prompt": "   "},
    {"sampler_name": "nope"},
    {"work_megapixels": 9.0},
    {"return_debug": "yes"},
])
def test_rejects_bad_values(extra):
    raw = {"image": "x", "mask": "y", "prompt": "p"}
    raw.update(extra)
    values, errors = validate_input(raw)
    assert errors and values is None, extra


def test_missing_required():
    out = validate({"image": "x"}, runpod_schema())
    assert any("mask" in e for e in out["errors"])
    assert any("prompt" in e for e in out["errors"])


def test_errors_carry_hint():
    _, msgs = validate_input({"image": "x", "mask": "y", "prompt": "p", "steps": 99})
    assert any("range [1, 50]" in m for m in msgs), msgs


def test_rp_validator_alone_misses_ranges_on_defaulted_fields():
    # Documents why validate_input re-checks constraints.
    out = validate({"image": "x", "mask": "y", "prompt": "p", "steps": 99}, runpod_schema())
    assert "errors" not in out


def test_validate_input_ok():
    values, errors = validate_input({"image": "x", "mask": "y", "prompt": "p", "cfg": 2})
    assert not errors and values["cfg"] == 2.0 and values["steps"] == 8
