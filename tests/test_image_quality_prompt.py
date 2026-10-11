
from pathlib import Path


def test_image_quality_prompt_formats_all_placeholders():
    prompt_path = Path("prompts/image_quality_evaluator.txt")
    template = prompt_path.read_text(encoding="utf-8")

    rendered = template.format(
        niche="technology",
        target_region="Bangladesh",
        audience_age_range="18-34",
        image_path="assets/images/raw/test-image.jpg",
    )

    assert "{image_path}" not in rendered
    assert "assets/images/raw/test-image.jpg" in rendered
    assert "technology" in rendered
    assert "Bangladesh" in rendered
