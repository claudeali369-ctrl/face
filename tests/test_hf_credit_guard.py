"""Regression tests for Hugging Face free-credit exhaustion handling."""
import os
import unittest
from unittest.mock import MagicMock, patch

from agents import visual_director as vd


class _Response:
    status_code = 402


class _QuotaExhausted(Exception):
    response = _Response()

    def __str__(self):
        return "402 Payment Required: You have no remaining credits"


class HuggingFaceCreditGuardTests(unittest.TestCase):
    def setUp(self):
        self.old_flag = vd.HF_CREDITS_EXHAUSTED_THIS_RUN
        vd.HF_CREDITS_EXHAUSTED_THIS_RUN = False
        self.addCleanup(setattr, vd, "HF_CREDITS_EXHAUSTED_THIS_RUN", self.old_flag)

    def test_402_persists_month_marker_and_stops_second_model_call(self):
        client = MagicMock()
        client.text_to_image.side_effect = _QuotaExhausted()
        with (
            patch.dict(os.environ, {"HF_TOKEN": "test-token"}),
            patch.object(vd, "InferenceClient", return_value=client),
            patch.object(vd.HF_LIMITER, "acquire"),
            patch.object(vd, "read_state", return_value={}),
            patch.object(vd, "write_state") as write_state,
        ):
            first = vd._provider_hf("model-a", "prompt", 512, 512, "post-1", 1, "test-HF")
            second = vd._provider_hf("model-b", "prompt", 512, 512, "post-2", 2, "test-HF-2")

        self.assertIsNone(first)
        self.assertIsNone(second)
        client.text_to_image.assert_called_once()
        write_state.assert_called_once()
        self.assertTrue(write_state.call_args.args[0] == vd.HF_CREDIT_STATUS_FILE)
        self.assertTrue(write_state.call_args.args[1]["credits_exhausted"])

    def test_current_month_marker_skips_huggingface_without_request(self):
        from datetime import datetime, timezone
        month = datetime.now(timezone.utc).strftime("%Y-%m")
        with (
            patch.dict(os.environ, {"HF_TOKEN": "test-token"}),
            patch.object(vd, "read_state", return_value={"month": month, "credits_exhausted": True}),
            patch.object(vd, "InferenceClient") as client_factory,
        ):
            result = vd._provider_hf("model-a", "prompt", 512, 512, "post-1", 1, "test-HF")

        self.assertIsNone(result)
        client_factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
