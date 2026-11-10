import re
from pathlib import Path

import pytest

from sprite_inpaint import workflow
from sprite_inpaint.workflow import MODEL_FILES, WorkflowParams, build_workflow

ROOT = Path(__file__).resolve().parent.parent

# Every class_type used must be a core node in ComfyUI 0.34.0 (checked by
# hand against comfy nodes.py / comfy_extras at tag v0.34.0).
CORE_NODES = {
    "UNETLoader", "CLIPLoader", "VAELoader", "ModelPatchLoader", "LoadImage",
    "LoadImageMask", "Canny", "ZImageFunControlnet", "ModelSamplingAuraFlow",
    "CLIPTextEncode", "ConditioningZeroOut", "VAEEncode", "SetLatentNoiseMask",
    "KSampler", "VAEDecode", "SaveImage",
}


def params(**kw):
    base = dict(source_name="s.png", mask_name="m.png", guide_name=None, guide_preprocess="none",
                prompt="fix", negative_prompt="", seed=1, steps=8, cfg=1.0, sampler_name="res_multistep",
                scheduler="simple", denoise=1.0, shift=3.0, control_strength=0.9, canny_low=0.1,
                canny_high=0.32, filename_prefix="sprite_inpaint/x")
    base.update(kw)
    return WorkflowParams(**base)


def test_template_uses_only_core_nodes():
    tpl = workflow.load_template()
    assert {n["class_type"] for n in tpl.values()} <= CORE_NODES


def test_inpaint_only_prunes_guide():
    wf = build_workflow(params())
    assert "12" not in wf and "13" not in wf
    assert "image" not in wf["20"]["inputs"]
    assert wf["20"]["inputs"]["inpaint_image"] == ["10", 0]
    assert wf["20"]["inputs"]["mask"] == ["11", 0]
    assert wf["50"]["inputs"]["negative"] == ["31", 0] and "32" not in wf
    assert wf["10"]["inputs"]["image"] == "s.png" and wf["11"]["inputs"]["image"] == "m.png"


def test_guide_direct_and_canny():
    wf = build_workflow(params(guide_name="g.png"))
    assert wf["20"]["inputs"]["image"] == ["12", 0] and "13" not in wf
    wf = build_workflow(params(guide_name="g.png", guide_preprocess="canny", canny_low=0.2, canny_high=0.5))
    assert wf["20"]["inputs"]["image"] == ["13", 0]
    assert wf["13"]["inputs"]["low_threshold"] == 0.2


def test_cfg_above_one_uses_negative_text():
    wf = build_workflow(params(cfg=3.0, negative_prompt="blurry"))
    assert wf["50"]["inputs"]["negative"] == ["32", 0]
    assert wf["32"]["inputs"]["text"] == "blurry" and "31" not in wf


def test_sampler_values_applied():
    wf = build_workflow(params(seed=42, steps=12, denoise=0.7, shift=2.5, control_strength=1.1))
    s = wf["50"]["inputs"]
    assert (s["seed"], s["steps"], s["denoise"]) == (42, 12, 0.7)
    assert wf["21"]["inputs"]["shift"] == 2.5
    assert wf["20"]["inputs"]["strength"] == 1.1


def test_dockerfile_patches_start_and_sets_memory_args():
    df = (ROOT / "Dockerfile").read_text()
    assert "patch-start /start.sh" in df
    assert 'ENV COMFY_EXTRA_ARGS="--reserve-vram 1.0"' in df


def test_crop_not_whole_sprite_for_part_repair():
    """A limb-sized mask on a 1254x1254 sprite generates only a crop."""
    import numpy as np
    from sprite_inpaint import imaging
    region = np.zeros((1254, 1254), bool)
    region[560:750, 690:940] = True
    p = imaging.plan_crop(region, pad_px=96, pad_ratio=0.35, work_megapixels=1.0, max_upscale=4.0)
    assert p.w * p.h < 0.25 * 1254 * 1254


def test_dockerfile_downloads_exactly_the_template_models():
    df = (ROOT / "Dockerfile").read_text()
    dests = re.findall(r"/comfyui/models/([a-z_]+)/(\S+\.safetensors)", df)
    assert sorted(dests) == sorted(MODEL_FILES.items())
    shas = re.findall(r"^\s+([0-9a-f]{64})\s*$", df, re.M)
    assert len(shas) == len(MODEL_FILES)
    assert "@sha256:" in df.splitlines()[[i for i, l in enumerate(df.splitlines()) if l.startswith("FROM")][0]]


@pytest.mark.parametrize("url", re.findall(r"https://huggingface\.co/\S+", (ROOT / "Dockerfile").read_text()))
def test_model_urls_are_pinned_to_a_commit(url):
    assert re.search(r"/resolve/[0-9a-f]{40}/", url), url
