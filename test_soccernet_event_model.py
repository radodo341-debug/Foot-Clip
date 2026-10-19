import unittest
from app.soccernet_event_model import convert_predictions


class SoccerNetAdapterTests(unittest.TestCase):
    def test_extracts_goal_and_converts_milliseconds(self):
        payload = {"data": [{"events": [
            {"label": "GOAL", "position_ms": 120500, "confidence": 0.91},
            {"label": "SHOT", "position_ms": 121000, "confidence": 0.99},
            {"label": "GOAL", "position_ms": 200000, "confidence": 0.9},
        ]}]}
        got = convert_predictions(payload, duration_seconds=180, threshold=0.35)
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0]["label"], "goal")
        self.assertAlmostEqual(got[0]["timestamp"], 120.5)
        self.assertEqual(got[0]["source"], "opensportslib-soccernet-pretrained")

    def test_threshold_and_invalid_times(self):
        payload = {"data": [{"events": [
            {"label": "GOAL", "position_ms": 1000, "confidence": 0.2},
            {"label": "GOAL", "position_ms": -1000, "confidence": 0.9},
            {"label": "GOAL", "position_ms": 5000, "confidence": 0.9},
        ]}]}
        self.assertEqual(convert_predictions(payload, duration_seconds=4, threshold=0.35), [])

    def test_never_fabricates_reaction_classes(self):
        payload = {"data": [{"events": [
            {"label": "GOAL", "position_ms": 1000, "confidence": 0.9},
            {"label": "CELEBRATION", "position_ms": 2000, "confidence": 0.99},
        ]}]}
        got = convert_predictions(payload, duration_seconds=4)
        self.assertEqual([p["label"] for p in got], ["goal"])


if __name__ == "__main__":
    unittest.main()
