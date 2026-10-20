
import unittest
from app import app, generate_suggestions


class TestMusicAnalysisApp(unittest.TestCase):

    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_homepage_loads(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(
            b"Music Analysis",
            response.data
        )

    def test_rejects_unsupported_file(self):
        response = self.client.post(
            "/",
            data={"audio": ( __import__("io").BytesIO(b"test"), "song.txt")},
            content_type="multipart/form-data"
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(
            b"Only MP3 and WAV files are supported",
            response.data
        )

    def test_suggestions_are_generated(self):
        suggestions = generate_suggestions(
            bpm=120,
            beat_count=240,
            duration=120
        )
        self.assertGreater(len(suggestions), 0)
        self.assertTrue(
            any("rule-based" in item for item in suggestions)
        )


if __name__ == "__main__":
    unittest.main()
