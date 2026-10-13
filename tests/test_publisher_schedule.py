"""Regression tests for publisher schedule logic; never call Facebook."""
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from agents.publisher import _is_within_window, _validate_item_assets, run_publishing_loop


class PublisherScheduleTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 10, 12, 12, 0, tzinfo=timezone.utc)

    def scheduled(self, minutes_from_now):
        return (self.now + timedelta(minutes=minutes_from_now)).isoformat()

    def test_publisher_allows_bounded_late_execution(self):
        self.assertTrue(_is_within_window(self.scheduled(-30), now=self.now))
        self.assertFalse(_is_within_window(self.scheduled(-61), now=self.now))

    def test_publisher_allows_only_small_early_tolerance(self):
        self.assertTrue(_is_within_window(self.scheduled(4), now=self.now))
        self.assertFalse(_is_within_window(self.scheduled(6), now=self.now))

    def test_explicit_legacy_window_remains_symmetric(self):
        self.assertTrue(_is_within_window(self.scheduled(-8), 10, now=self.now))
        self.assertFalse(_is_within_window(self.scheduled(-11), 10, now=self.now))

    def test_manual_override_publishes_ready_content_outside_window(self):
        content = [{
            "post_id": "test-post", "format": "text", "status": "ready",
            "caption": "Test caption", "hashtags": [],
        }]
        schedule = {"test-post": {"scheduled_time_utc": "2020-01-01T00:00:00+00:00"}}
        with (
            patch("agents.publisher._now_utc", return_value=self.now),
            patch("agents.publisher._publish_with_retry", return_value="fb-post-123"),
            patch("agents.publisher._append_post_history"),
        ):
            published, failed, skipped = run_publishing_loop(
                fb=object(), content=content, schedule_map=schedule, publish_now=True
            )
        self.assertEqual(len(published), 1)
        self.assertEqual(published[0]["post_id"], "fb-post-123")
        self.assertEqual(failed, [])
        self.assertEqual(skipped, [])
        self.assertEqual(content[0]["status"], "published")

    def test_incomplete_single_image_is_failed_before_publish_retry(self):
        content = [{
            "post_id": "broken-image", "format": "single_image", "status": "ready",
            "image_ready": False, "image_path": "", "caption": "Test", "hashtags": [],
        }]
        with (
            patch("agents.publisher._write_error"),
            patch("agents.publisher._publish_with_retry") as publish,
        ):
            published, failed, skipped = run_publishing_loop(
                fb=object(), content=content, schedule_map={"broken-image": {"scheduled_time_utc": "2026-10-12T12:00:00+00:00"}}, publish_now=True
            )
        publish.assert_not_called()
        self.assertEqual(published, [])
        self.assertEqual(len(failed), 1)
        self.assertEqual(skipped, [])
        self.assertEqual(content[0]["status"], "failed")
        self.assertIn("image_ready", failed[0]["error"])

    def test_incomplete_story_is_failed_before_publish_retry(self):
        content = [{
            "post_id": "broken-story", "format": "story", "status": "ready",
            "image_ready": False,
            "story_slides": [{"image_path": ""}, {"image_path": ""}, {"image_path": ""}],
            "caption": "Test", "hashtags": [],
        }]
        with (
            patch("agents.publisher._write_error"),
            patch("agents.publisher._publish_with_retry") as publish,
        ):
            published, failed, skipped = run_publishing_loop(
                fb=object(), content=content, schedule_map={"broken-story": {"scheduled_time_utc": "2026-10-12T12:00:00+00:00"}}, publish_now=True
            )
        publish.assert_not_called()
        self.assertEqual(published, [])
        self.assertEqual(len(failed), 1)
        self.assertEqual(skipped, [])
        self.assertEqual(content[0]["status"], "failed")
        self.assertIn("story image_ready", failed[0]["error"])

    def test_text_post_does_not_require_an_image(self):
        self.assertIsNone(_validate_item_assets({"format": "text"}))

    def test_future_item_with_missing_asset_is_deferred_not_failed(self):
        content = [{
            "post_id": "future-broken-image", "format": "single_image", "status": "ready",
            "image_ready": False, "image_path": "", "caption": "Test", "hashtags": [],
        }]
        schedule = {
            "future-broken-image": {
                "scheduled_time_utc": (self.now + timedelta(minutes=20)).isoformat()
            }
        }
        with (
            patch("agents.publisher._now_utc", return_value=self.now),
            patch("agents.publisher._write_error") as write_error,
            patch("agents.publisher._publish_with_retry") as publish,
        ):
            published, failed, skipped = run_publishing_loop(
                fb=object(), content=content, schedule_map=schedule, publish_now=False
            )
        write_error.assert_not_called()
        publish.assert_not_called()
        self.assertEqual(published, [])
        self.assertEqual(failed, [])
        self.assertEqual(skipped, [])
        self.assertEqual(content[0]["status"], "ready")

    def test_default_mode_defers_old_content_without_calling_facebook(self):
        content = [{
            "post_id": "test-post", "format": "text", "status": "ready",
            "caption": "Test caption", "hashtags": [],
        }]
        schedule = {"test-post": {"scheduled_time_utc": "2020-01-01T00:00:00+00:00"}}
        with (
            patch("agents.publisher._now_utc", return_value=self.now),
            patch("agents.publisher._publish_with_retry") as publish,
        ):
            published, failed, skipped = run_publishing_loop(
                fb=object(), content=content, schedule_map=schedule
            )
        publish.assert_not_called()
        self.assertEqual(published, [])
        self.assertEqual(failed, [])
        self.assertEqual(skipped, [])
        self.assertEqual(content[0]["status"], "ready")


if __name__ == "__main__":
    unittest.main()
