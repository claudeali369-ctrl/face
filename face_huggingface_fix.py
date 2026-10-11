#!/usr/bin/env python3
"""Safely patch the FACE repo's broken Hugging Face image provider integration.

Run this from the root of your cloned/Codespaces `face` repository:
    python face_huggingface_fix.py

The script creates timestamped backups before changing any files. It does not
upload anything to GitHub or touch repository secrets.
"""
from __future__ import annotations

import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path.cwd()
VISUAL_PATH = ROOT / "agents" / "visual_director.py"
IMAGE_UTILS_PATH = ROOT / "utils" / "image_utils.py"
README_PATH = ROOT / "README.md"
REQUIREMENTS_PATH = ROOT / "requirements.txt"


def fail(message: str) -> "NoReturn":
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def replace_exact_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        fail(f"Expected exactly one match for {label}; found {count}. No files have been changed.")
    return text.replace(old, new, 1)


def main() -> None:
    required = [VISUAL_PATH, IMAGE_UTILS_PATH, README_PATH, REQUIREMENTS_PATH]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.is_file()]
    if missing:
        fail(
            "Run this script from the repository root. Missing: " + ", ".join(missing)
        )

    original: dict[Path, str] = {p: p.read_text(encoding="utf-8") for p in required}
    changed = dict(original)

    # Update imports without duplicating them on a second run.
    visual = changed[VISUAL_PATH]
    if "import io\n" not in visual:
        visual = replace_exact_once(visual, "import json\n", "import json\nimport io\n", "visual_director import io")
    if "from huggingface_hub import InferenceClient" not in visual:
        visual = replace_exact_once(
            visual,
            "import requests\n",
            "import requests\nfrom huggingface_hub import InferenceClient\n",
            "visual_director InferenceClient import",
        )

    old_constants = '''HF_FLUX_URL = (
    "https://huggingface.co"
)
HF_SDXL_URL = (
    "https://huggingface.co"
    "stabilityai/stable-diffusion-xl-base-1.0"
)'''
    new_constants = '''# Model IDs are routed through Hugging Face Inference Providers.
# The provider is selected automatically by huggingface_hub's InferenceClient.
HF_FLUX_MODEL = "black-forest-labs/FLUX.1-schnell"
HF_SDXL_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"'''
    if old_constants in visual:
        visual = replace_exact_once(visual, old_constants, new_constants, "Hugging Face model constants")
    elif not all(x in visual for x in ("HF_FLUX_MODEL", "HF_SDXL_MODEL")):
        fail("Could not find the expected broken Hugging Face URL constants. No files have been changed.")

    # Replace the raw POST-to-website implementation with the supported client API.
    provider_pattern = re.compile(
        r"def _provider_hf\(.*?(?=\ndef _provider_pexels\()",
        flags=re.DOTALL,
    )
    provider_replacement = '''def _provider_hf(
    model_id: str,
    prompt: str,
    width: int,
    height: int,
    post_id: str,
    attempt: int,
    provider_tag: str,
) -> Path | None:
    """Generate an image through Hugging Face Inference Providers.

    InferenceClient handles provider routing; the token must have Inference
    Providers permission and the account must have any required credits.
    Returns a local JPEG path on success, or None if all retries fail.
    """
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if not hf_token:
        log.warning("[%s] HF_TOKEN not set — skipping", provider_tag)
        return None

    try:
        client = InferenceClient(api_key=hf_token, timeout=120)
    except Exception as exc:
        log.warning("[%s] Could not initialize Hugging Face client: %s", provider_tag, exc)
        return None

    # Keep requests within a conservative 1024-pixel longest side and use
    # dimensions divisible by 8, which is safer across hosted image models.
    target_width, target_height = int(width), int(height)
    scale = min(1.0, 1024.0 / max(target_width, target_height))
    request_width = max(64, (int(target_width * scale) // 8) * 8)
    request_height = max(64, (int(target_height * scale) // 8) * 8)

    for retry in range(1, 4):
        HF_LIMITER.acquire()
        log.info(
            "[%s] Requesting model %s for post %s (retry %d/3)",
            provider_tag, model_id, post_id, retry,
        )
        try:
            image = client.text_to_image(
                prompt,
                model=model_id,
                width=request_width,
                height=request_height,
            )
            # InferenceClient returns a PIL image, not a raw requests.Response.
            # Restore the requested social-media dimensions after generation.
            if image.size != (target_width, target_height):
                from PIL import Image
                resampling = getattr(Image, "Resampling", Image)
                image = image.resize((target_width, target_height), resampling.LANCZOS)

            # Encode explicitly as JPEG so the cached file extension matches.
            buffer = io.BytesIO()
            image.convert("RGB").save(buffer, format="JPEG", quality=95)
            content = buffer.getvalue()
            buffer.close()
            if not content or len(content) < 1000:
                log.warning("[%s] Image response too small for post %s", provider_tag, post_id)
                return None
            return _save_raw(content, post_id, attempt)
        except Exception as exc:
            response = getattr(exc, "response", None)
            status_code = getattr(response, "status_code", None)
            if status_code == 503 and retry < 3:
                log.warning("[%s] Provider/model temporarily unavailable (503); retrying in 30 s", provider_tag)
                time.sleep(30)
                continue
            log.warning("[%s] Attempt %d failed: %s", provider_tag, retry, exc)
            if retry < 3:
                time.sleep(10)
    return None
'''
    provider_match = provider_pattern.search(visual)
    if provider_match:
        visual = provider_pattern.sub(lambda _m: provider_replacement, visual, count=1)
    elif "InferenceClient(api_key=hf_token" not in visual:
        fail("Could not locate _provider_hf function boundaries. No files have been changed.")

    # Calls keep the same waterfall order while passing real model IDs.
    visual = visual.replace("HF_FLUX_URL", "HF_FLUX_MODEL")
    visual = visual.replace("HF_SDXL_URL", "HF_SDXL_MODEL")

    # The colour helper uses None to trigger auto-contrast. Keep 'auto' compatible too.
    if 'color="auto",' in visual:
        visual = replace_exact_once(visual, 'color="auto",', "color=None,", "automatic overlay colour argument")
    changed[VISUAL_PATH] = visual

    image_utils = changed[IMAGE_UTILS_PATH]
    if "if color is None or (isinstance(color, str) and color.strip().lower() == \"auto\")" not in image_utils:
        image_utils = replace_exact_once(
            image_utils,
            "if color is None:",
            'if color is None or (isinstance(color, str) and color.strip().lower() == "auto"):',
            "automatic colour handling",
        )
    changed[IMAGE_UTILS_PATH] = image_utils

    requirements = changed[REQUIREMENTS_PATH]
    if not re.search(r"(?im)^\s*huggingface[-_]hub(?:\s|[<=>~!]|$)", requirements):
        requirements = requirements.rstrip() + "\nhuggingface_hub>=0.34.0\n"
    changed[REQUIREMENTS_PATH] = requirements

    readme = changed[README_PATH]
    old_token_text = "4. Name it `face-token`, select Read permission → Generate token"
    new_token_text = (
        "4. Create a token with **Inference Providers** permission (a token with only "
        "repository-read permission is not sufficient) → Generate token"
    )
    if old_token_text in readme:
        readme = readme.replace(old_token_text, new_token_text, 1)
    old_free_tier = (
        "Free tier: ~1,000 inference API requests/day. Used for FLUX.1-schnell and "
        "Stable Diffusion XL image generation (second and third fallbacks)."
    )
    new_free_tier = (
        "Availability and cost depend on the selected Inference Provider and your "
        "account credits. Check your Hugging Face Inference Providers usage before "
        "relying on high-volume daily generation. Used for FLUX.1-schnell and Stable "
        "Diffusion XL image generation (second and third fallbacks)."
    )
    if old_free_tier in readme:
        readme = readme.replace(old_free_tier, new_free_tier, 1)
    changed[README_PATH] = readme

    # Sanity checks before writing anything.
    if "https://huggingface.co\"" in changed[VISUAL_PATH] and "HF_FLUX_URL" in changed[VISUAL_PATH]:
        fail("The old Hugging Face URL constants are still present. No files have been changed.")
    if "HF_FLUX_URL" in changed[VISUAL_PATH] or "HF_SDXL_URL" in changed[VISUAL_PATH]:
        fail("Old URL identifiers remain in visual_director.py. No files have been changed.")
    if 'color="auto",' in changed[VISUAL_PATH]:
        fail('The visual director still passes color="auto". No files have been changed.')

    changed_files = [p for p in required if changed[p] != original[p]]
    if not changed_files:
        print("No changes needed; the expected fixes already appear to be applied.")
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = ROOT.parent / f"{ROOT.name}_hf_fix_backup_{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    for path in changed_files:
        relative = path.relative_to(ROOT)
        backup_path = backup_dir / relative
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup_path)

    for path in changed_files:
        path.write_text(changed[path], encoding="utf-8", newline="\n")

    print("Applied Hugging Face and overlay fixes to:")
    for path in changed_files:
        print(f"  - {path.relative_to(ROOT)}")
    print(f"Backups saved in: {backup_dir.name}/")
    print("Next: verify HF_TOKEN has Inference Providers permission, install requirements, and run the VisualDirector workflow.")


if __name__ == "__main__":
    main()
