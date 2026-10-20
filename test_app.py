
import unittest
from unittest.mock import patch
import numpy as np

from app import app, analyse_audio, generate_suggestions


class TestMusicAnalysisApp(unittest.TestCase):

    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_homepage_loads(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Music Analysis", response.data)

    def test_rejects_unsupported_file(self):
        import io

        response = self.client.post(
            "/",
            data={
                "audio": (io.BytesIO(b"test"), "song.txt")
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(
            b"Only MP3 and WAV files are supported",
            response.data,
        )

    def test_suggestions_are_generated(self):
        suggestions = generate_suggestions(
            bpm=120,
            beat_count=240,
            duration=120,
        )

        self.assertGreater(len(suggestions), 0)
        self.assertTrue(
            any("rule-based" in item for item in suggestions)
        )

    @patch("app.librosa.beat.beat_track")
    @patch("app.librosa.load")
    def test_audio_analysis_results(
        self, mock_load, mock_beat_track
    ):
        mock_load.return_value = (
            np.zeros(22050 * 10),
            22050,
        )

        mock_beat_track.return_value = (
            np.array([120.0]),
            np.array([0, 10, 20, 30]),
        )

        with patch(
            "app.librosa.frames_to_time",
            return_value=np.array([0.0, 0.5, 1.0, 1.5]),
        ):
            results = analyse_audio("fake_song.wav")

        self.assertEqual(results["bpm"], 120.0)
        self.assertEqual(results["duration"], 10.0)
        self.assertEqual(results["beat_count"], 4)
        self.assertEqual(
            results["beat_times"],
            [0.0, 0.5, 1.0, 1.5],
        )
        self.assertGreater(len(results["suggestions"]), 0)


if __name__ == "__main__":
    unittest.main()
