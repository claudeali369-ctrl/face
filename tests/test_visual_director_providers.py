"""VisualDirector image-provider tests. These tests never call the network."""
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import requests
from PIL import Image

from agents import reel_producer, visual_director


class FakeImageResponse:
    status_code = 200
    content = b"x" * 2048
    headers = {"Content-Type": "image/jpeg"}

    def raise_for_status(self):
        return None


class FakeErrorResponse:
    def __init__(self, status_code):
        self.status_code = status_code


class PollinationsProviderTests(unittest.TestCase):
    def setUp(self):
        visual_director.POLLINATIONS_DISABLED_THIS_RUN = False

    def tearDown(self):
        visual_director.POLLINATIONS_DISABLED_THIS_RUN = False

    def test_missing_key_skips_request_without_network_call(self):
        with (
            patch.dict(os.environ, {"POLLINATIONS_API_KEY": "", "POLLINATIONS_API_DISABLED": "", "POLLINATIONS_IMAGE_MODEL": "", "GITHUB_ENV": ""}, clear=False),
            patch.object(visual_director.requests, "get") as get,
        ):
            result = visual_director._provider_pollinations("a test prompt", 256, 256, "post", 1)
        self.assertIsNone(result)
        get.assert_not_called()
        self.assertTrue(visual_director.POLLINATIONS_DISABLED_THIS_RUN)

    def test_uses_current_authenticated_endpoint_and_image_parameters(self):
        with (
            patch.dict(os.environ, {"POLLINATIONS_API_KEY": "sk_test_only", "POLLINATIONS_API_DISABLED": "", "POLLINATIONS_IMAGE_MODEL": "", "GITHUB_ENV": ""}, clear=False),
            patch.object(visual_director.requests, "get", return_value=FakeImageResponse()) as get,
            patch.object(visual_director, "_save_raw", return_value=Path("fake.jpg")) as save,
            patch.object(visual_director.random, "randint", return_value=12345),
        ):
            result = visual_director._provider_pollinations("a cat / in space", 512, 768, "post", 2)
        self.assertEqual(result, Path("fake.jpg"))
        args, kwargs = get.call_args
        self.assertTrue(args[0].startswith("https://gen.pollinations.ai/image/"))
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer sk_test_only")
        self.assertEqual(kwargs["params"]["model"], "flux")
        self.assertEqual(kwargs["params"]["width"], 512)
        self.assertEqual(kwargs["params"]["height"], 768)
        self.assertEqual(kwargs["params"]["seed"], 12345)
        self.assertEqual(kwargs["params"]["safe"], "true")
        self.assertEqual(kwargs["timeout"], (15, 120))
        save.assert_called_once_with(FakeImageResponse.content, "post", 2)

    def test_persistent_auth_or_budget_error_disables_further_requests(self):
        error = requests.HTTPError("payment required")
        error.response = FakeErrorResponse(402)
        with (
            patch.dict(os.environ, {"POLLINATIONS_API_KEY": "sk_test_only", "POLLINATIONS_API_DISABLED": "", "POLLINATIONS_IMAGE_MODEL": "", "GITHUB_ENV": ""}, clear=False),
            patch.object(visual_director.requests, "get", side_effect=error) as get,
        ):
            first = visual_director._provider_pollinations("prompt one", 256, 256, "p1", 1)
            second = visual_director._provider_pollinations("prompt two", 256, 256, "p2", 1)
        self.assertIsNone(first)
        self.assertIsNone(second)
        self.assertEqual(get.call_count, 1)
        self.assertTrue(visual_director.POLLINATIONS_DISABLED_THIS_RUN)


class ReelProviderTests(unittest.TestCase):
    def test_reel_frame_uses_shared_provider_and_outputs_vertical_jpeg(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "provider.jpg"
            output_dir = root / "reel_frames"
            output_dir.mkdir()
            Image.new("RGB", (160, 90), color=(40, 80, 120)).save(source, format="JPEG")

            with (
                patch.object(reel_producer, "IMAGES_REEL", output_dir),
                patch.object(reel_producer, "_provider_pollinations", return_value=source) as pollinations,
                patch.object(reel_producer, "_provider_hf") as hf,
                patch.object(reel_producer, "_provider_pixabay") as pixabay,
                patch.dict(os.environ, {"POLLINATIONS_API_DISABLED": "", "POLLINATIONS_IMAGE_MODEL": "", "PIXABAY_API_KEY": ""}, clear=False),
            ):
                result = reel_producer._generate_segment_image(
                    "test prompt", "test-reel", 1,
                    config={"api": {"pollinations_image_model": "flux"}},
                )

            self.assertIsNotNone(result)
            self.assertTrue(Path(result).is_file())
            with Image.open(result) as output:
                self.assertEqual(output.size, (1080, 1920))
                self.assertEqual(output.format, "JPEG")
            pollinations.assert_called_once()
            hf.assert_not_called()
            pixabay.assert_not_called()


if __name__ == "__main__":
    unittest.main()
