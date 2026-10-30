import base64
import io
import json
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

import app


def photo():
    output = io.BytesIO()
    Image.new("RGB", (64, 48), "white").save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode()


class LocalDiceTests(unittest.TestCase):
    def setUp(self):
        app.HITS.clear()
        self.client = TestClient(app.app)

    def test_page_and_local_privacy(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.get("/", headers={"Host": "example.com"}).status_code, 403)
        self.assertEqual(self.client.post("/api/recognize", headers={"Origin": "https://example.com"}, json={}).status_code, 403)

    def test_byok_page_allows_direct_model_calls(self):
        page = self.client.get("/static/byok.html")
        self.assertEqual(page.status_code, 200)
        csp = page.headers["content-security-policy"]
        self.assertIn("connect-src 'self' https:", csp)
        self.assertIn("http://127.0.0.1:*", csp)
        self.assertIn("byok.js", page.text)
        main = self.client.get("/")
        self.assertNotIn("https:", main.headers["content-security-policy"])

    def test_image_validation_and_resize(self):
        output = base64.b64decode(app.prepare_image(photo()))
        with Image.open(io.BytesIO(output)) as im:
            self.assertEqual(im.size, (64, 48))
            self.assertEqual(im.format, "JPEG")
        r = self.client.post("/api/recognize", json={"image": "not an image"})
        self.assertEqual(r.status_code, 400)

    def test_invalid_requests(self):
        for body in [{}, {"image": 23}, {"image": photo(), "d4": "bad"}, []]:
            self.assertEqual(self.client.post("/api/recognize", json=body).status_code, 400)

    def test_model_result_and_ambiguity(self):
        result = {"dice": [
            {"kind": "d20", "value": 17, "bbox": [20, 30, 200, 250], "note": ""},
            {"kind": "d4", "value": None, "bbox": [300, 30, 500, 250], "note": "读数不清"},
            {"kind": "d00", "value": 0, "bbox": [600, 30, 900, 250], "note": ""},
        ], "warning": "请核对"}
        fake = {"message": {"content": json.dumps(result)}, "done_reason": "stop"}
        with patch.object(app, "ollama_json", return_value=fake) as mocked:
            response = self.client.post("/api/recognize", json={"image": photo(), "d4": "bottom"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["dice"], result["dice"])
        self.assertIn("bottom edges", mocked.call_args.args[1]["messages"][0]["content"])

    def test_out_of_range_is_not_silently_counted(self):
        for kind, value in [("d20", 21), ("d4", 0), ("d00", 15), ("d6", True)]:
            result = {"dice": [{"kind": kind, "value": value, "bbox": [0, 0, 100, 100], "note": ""}], "warning": ""}
            with patch.object(app, "ollama_json", return_value={"message": {"content": json.dumps(result)}}):
                response = self.client.post("/api/recognize", json={"image": photo()})
            self.assertEqual(response.status_code, 502)

    def test_no_dice_and_busy(self):
        with patch.object(app, "ollama_json", return_value={"message": {"content": '{"dice":[],"warning":""}'}}):
            self.assertEqual(self.client.post("/api/recognize", json={"image": photo()}).json()["dice"], [])
        with app.LOCK:
            response = self.client.post("/api/recognize", json={"image": photo()})
            self.assertEqual(response.status_code, 409)

    def test_backend_unavailable(self):
        with patch.object(app, "ollama_json", side_effect=OSError("offline")):
            self.assertFalse(self.client.get("/api/status").json()["ready"])
            self.assertEqual(self.client.post("/api/recognize", json={"image": photo()}).status_code, 503)
        self.assertFalse(app.LOCK.locked())


if __name__ == "__main__":
    unittest.main()
