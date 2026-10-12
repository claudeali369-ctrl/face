"""Regression tests for technology trend sourcing; no network calls are made."""
import unittest
from unittest.mock import patch

from agents import trend_hunter


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class TrendHunterSourceTests(unittest.TestCase):
    def test_technology_niche_uses_technology_subreddits(self):
        subs = trend_hunter._get_subreddits({"niche": "technology"})
        self.assertEqual(subs, trend_hunter.NICHE_SUBREDDITS["technology"])
        self.assertNotEqual(subs, trend_hunter.NICHE_SUBREDDITS["default"])

    def test_hacker_news_fetch_normalizes_and_sorts_stories(self):
        stories = {
            1: {
                "id": 1, "type": "story", "title": "A useful AI workflow",
                "score": 85, "descendants": 12, "url": "https://example.com/1",
                "time": 1791820000,
            },
            2: {
                "id": 2, "type": "story", "title": "A new programming tool",
                "score": 240, "descendants": 54, "url": "https://example.com/2",
                "time": 1791820100,
            },
            3: {"id": 3, "type": "story", "title": "Deleted story", "score": 999, "deleted": True},
            4: {"id": 4, "type": "comment", "title": "Not a story", "score": 999},
        }
        with (
            patch.object(
                trend_hunter.requests, "get",
                return_value=FakeResponse([1, 2, 3, 4]),
            ),
            patch.object(
                trend_hunter, "_fetch_hacker_news_item",
                side_effect=lambda story_id: stories[story_id],
            ),
        ):
            result = trend_hunter.fetch_hacker_news(limit=4)

        self.assertEqual([item["topic"] for item in result], [
            "A new programming tool", "A useful AI workflow"
        ])
        self.assertEqual(result[0]["source"], "hacker_news")
        self.assertEqual(result[0]["raw_score"], 2.4)
        self.assertEqual(result[0]["raw_data"]["story_id"], 2)
        self.assertTrue(result[0]["raw_data"]["published_at"])

    def test_hacker_news_handles_failed_top_story_request(self):
        with patch.object(
            trend_hunter.requests, "get",
            side_effect=trend_hunter.requests.RequestException("offline"),
        ):
            self.assertEqual(trend_hunter.fetch_hacker_news(limit=3), [])


if __name__ == "__main__":
    unittest.main()
