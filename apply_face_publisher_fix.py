"""Safely patch FACE publisher scheduling in a local repository checkout.

Run this script from inside the repository root after copying it there.
It validates every expected source block before writing any files.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"{label}: expected one matching block, found {count}; no files have been written."
        )
    return source.replace(old, new, 1)


publisher_path = ROOT / "agents" / "publisher.py"
workflow_path = ROOT / ".github" / "workflows" / "daily_pipeline.yml"
tests_path = ROOT / "tests" / "test_publisher_schedule.py"

publisher = publisher_path.read_text(encoding="utf-8")
workflow = workflow_path.read_text(encoding="utf-8")

publisher = replace_once(
    publisher,
    "SCHEDULE_WINDOW_MINUTES = 10          # publish if within \u00b110 min of scheduled time",
    """# The image/text pipeline takes time; tolerate a bounded delay after the slot.
SCHEDULE_EARLY_TOLERANCE_MINUTES = 5
SCHEDULE_LATE_TOLERANCE_MINUTES = 60""",
    "publisher schedule constants",
)

publisher = replace_once(
    publisher,
    '''def _is_within_window(scheduled_iso: str, window_minutes: int = SCHEDULE_WINDOW_MINUTES) -> bool:
    """Return True if the current UTC time is within \u00b1window_minutes of scheduled_iso."""
    try:
        scheduled = datetime.fromisoformat(scheduled_iso)
        if scheduled.tzinfo is None:
            scheduled = scheduled.replace(tzinfo=timezone.utc)
        delta = abs((_now_utc() - scheduled).total_seconds())
        return delta <= window_minutes * 60
    except Exception as exc:
        log.warning(f"Could not parse scheduled_time '{scheduled_iso}': {exc}")
        return False''',
    '''def _is_within_window(
    scheduled_iso: str,
    window_minutes: int | None = None,
    *,
    now: datetime | None = None,
) -> bool:
    """Return whether a slot is due, with bounded late tolerance.

    The default allows up to 5 minutes early and 60 minutes late because the
    workflow generates content and images before it reaches the publisher.
    Passing window_minutes preserves the old symmetric-window behavior for
    callers and tests that explicitly need it.
    """
    try:
        scheduled = datetime.fromisoformat(scheduled_iso.replace("Z", "+00:00"))
        if scheduled.tzinfo is None:
            scheduled = scheduled.replace(tzinfo=timezone.utc)

        current = now or _now_utc()
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)

        seconds_until_scheduled = (scheduled - current).total_seconds()
        if window_minutes is not None:
            return abs(seconds_until_scheduled) <= window_minutes * 60

        return (
            -SCHEDULE_LATE_TOLERANCE_MINUTES * 60
            <= seconds_until_scheduled
            <= SCHEDULE_EARLY_TOLERANCE_MINUTES * 60
        )
    except (TypeError, ValueError, OverflowError) as exc:
        log.warning("Could not parse scheduled_time %r: %s", scheduled_iso, exc)
        return False''',
    "publisher schedule-window helper",
)

publisher = replace_once(
    publisher,
    '''def run_publishing_loop(
    fb: FacebookClient,
    content: list[dict],
    schedule_map: dict[str, dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Iterate over ready content items and publish those in the schedule window.

    Returns: (published_items, failed_items, skipped_items)
    """
    published, failed, skipped = [], [], []
    now = _now_utc()''',
    '''def run_publishing_loop(
    fb: FacebookClient,
    content: list[dict],
    schedule_map: dict[str, dict],
    publish_now: bool = False,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Publish ready content whose slot is due. For deliberate manual testing,
    publish_now=True bypasses the schedule check for every ready item.

    Returns: (published_items, failed_items, skipped_items)
    """
    published, failed, skipped = [], [], []
    deferred_count = 0
    now = _now_utc()

    if publish_now:
        log.warning(
            "Immediate publish override enabled: schedule checks are bypassed "
            "for all ready content items."
        )''',
    "publisher loop signature",
)

publisher = replace_once(
    publisher,
    '''        if not _is_within_window(scheduled_time):
            log.info(
                "Post outside publishing window \u2014 "
                "scheduled_time=%s, current_utc=%s, window=\u00b1%s minutes",
                scheduled_time,
                _now_utc().isoformat(),
                SCHEDULE_WINDOW_MINUTES,
            )
            continue''',
    '''        if not publish_now and not _is_within_window(scheduled_time, now=now):
            deferred_count += 1
            log.info(
                "Post deferred outside publishing window \u2014 "
                "scheduled_time=%s, current_utc=%s, allowed_early=%s min, allowed_late=%s min",
                scheduled_time,
                now.isoformat(),
                SCHEDULE_EARLY_TOLERANCE_MINUTES,
                SCHEDULE_LATE_TOLERANCE_MINUTES,
            )
            continue''',
    "publisher schedule gate",
)

publisher = replace_once(
    publisher,
    '''        published.append({"post_id": fb_post_id, "format": fmt, "status": "published"})

    return published, failed, skipped''',
    '''        published.append({"post_id": fb_post_id, "format": fmt, "status": "published"})

    if deferred_count:
        log.info(
            "Deferred %d ready item(s): no Facebook request was made for those items.",
            deferred_count,
        )

    return published, failed, skipped''',
    "publisher deferred summary",
)

publisher = replace_once(
    publisher,
    '''    log.info("\u2550\u2550 PublisherAgent starting \u2550\u2550")
    run_at = _now_utc()



    # \u2500\u2500 Pre-flight: token validity check''',
    '''    log.info("\u2550\u2550 PublisherAgent starting \u2550\u2550")
    run_at = _now_utc()
    publish_now = os.environ.get("FACE_PUBLISH_NOW", "").strip().lower() in {
        "1", "true", "yes", "on"
    }

    # \u2500\u2500 Pre-flight: token validity check''',
    "publisher manual override flag",
)

publisher = replace_once(
    publisher,
    "    published, failed, skipped = run_publishing_loop(fb, content_items, schedule_map)",
    """    published, failed, skipped = run_publishing_loop(
        fb, content_items, schedule_map, publish_now=publish_now
    )""",
    "publisher manual override call",
)

workflow = replace_once(
    workflow,
    "  workflow_dispatch:    # Manual trigger for testing",
    '''  workflow_dispatch:
    inputs:
      publish_now:
        description: "Publish all ready items immediately; may publish several items at once."
        required: false
        type: boolean
        default: false''',
    "workflow_dispatch manual input",
)

workflow = replace_once(
    workflow,
    '''      GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      GITHUB_REPOSITORY: ${{ github.repository }}''',
    '''      GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      GITHUB_REPOSITORY: ${{ github.repository }}
      FACE_PUBLISH_NOW: ${{ github.event_name == 'workflow_dispatch' && inputs.publish_now || false }}''',
    "workflow publish-now environment flag",
)

test_source = '''"""Regression tests for publisher schedule logic; never call Facebook."""
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from agents.publisher import _is_within_window, run_publishing_loop


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
'''

if tests_path.exists():
    raise FileExistsError(
        f"{tests_path} already exists; inspect it before applying this script."
    )

# Write only after all source-block checks have passed.
publisher_path.write_text(publisher, encoding="utf-8")
workflow_path.write_text(workflow, encoding="utf-8")
tests_path.write_text(test_source, encoding="utf-8")

print("Patched agents/publisher.py")
print("Patched .github/workflows/daily_pipeline.yml")
print("Created tests/test_publisher_schedule.py")
print("Next: run python -m unittest discover -s tests -p 'test_publisher_schedule.py'")
