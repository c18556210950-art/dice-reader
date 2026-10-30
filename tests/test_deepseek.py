import io
import json
import os
import unittest
import urllib.error
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

import app
import deepseek_client


class DeepSeekTests(unittest.TestCase):
    def test_status_has_no_secret_and_does_not_call_local_engine(self):
        with patch.object(app, 'PROVIDER', 'deepseek'), patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'unit-secret-only'}), patch.object(app, 'ollama_json') as local:
            response = TestClient(app.app).get('/api/status')
            self.assertTrue(response.json()['ready'])
            self.assertEqual(response.json()['provider'], 'deepseek')
            self.assertNotIn('unit-secret-only', response.text)
            self.assertNotIn('privacy', response.json())
            self.assertNotIn('DeepSeek', response.text)
            local.assert_not_called()

    def test_api_shape_and_image_are_sent_to_fixed_official_endpoint(self):
        upstream = {'choices': [{'finish_reason': 'stop', 'message': {'content': '{"dice":[],"warning":""}'}}], 'usage': {'total_tokens': 100}}
        with patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'unit-secret-only'}), patch.object(deepseek_client.HTTP, 'open', return_value=io.BytesIO(json.dumps(upstream).encode())) as opened:
            content, usage = deepseek_client.recognize('image-bytes', 'Read dice', {})
        request = opened.call_args.args[0]
        self.assertEqual(request.full_url, 'https://api.deepseek.com/chat/completions')
        payload = json.loads(request.data)
        self.assertEqual(payload['model'], 'deepseek-flash')
        self.assertEqual(payload['thinking']['type'], 'disabled')
        self.assertEqual(payload['temperature'], 0)
        self.assertEqual(payload['top_p'], 1)
        self.assertEqual(payload['messages'][0]['content'][1]['image_url']['url'], 'data:image/jpeg;base64,image-bytes')
        self.assertNotIn('unit-secret-only', content)
        self.assertEqual(usage['total_tokens'], 100)

    def test_auth_errors_are_sanitized(self):
        error = urllib.error.HTTPError(deepseek_client.ENDPOINT, 401, 'unit-secret-only', {}, None)
        with patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'unit-secret-only'}), patch.object(deepseek_client.HTTP, 'open', side_effect=error):
            with self.assertRaises(HTTPException) as raised:
                deepseek_client.recognize('image-bytes', 'Read dice', {})
        self.assertEqual(raised.exception.status_code, 503)
        self.assertNotIn('unit-secret-only', raised.exception.detail)

    def test_rate_limit_is_per_client_ip(self):
        app.HITS.clear()
        client = TestClient(app.app)
        with patch.object(app, 'RATE_MAX', 2), patch.object(app, 'RATE_WINDOW', 60):
            headers = {'X-Real-IP': '198.51.100.7'}
            for _ in range(2):
                response = client.post('/api/recognize', headers=headers, content=b'{}')
                self.assertEqual(response.status_code, 400)
            response = client.post('/api/recognize', headers=headers, content=b'{}')
            self.assertEqual(response.status_code, 429)
            self.assertIn('Retry-After', response.headers)
            other = client.post('/api/recognize', headers={'X-Real-IP': '198.51.100.8'}, content=b'{}')
            self.assertEqual(other.status_code, 400)
        app.HITS.clear()

    def test_bad_die_is_repaired_instead_of_failing_the_whole_roll(self):
        payload = {
            'dice': [
                {'kind': 'd6', 'value': 18, 'bbox': [0, 0, 100, 100], 'note': ''},
                {'kind': 'd10', 'value': 0, 'bbox': [100, 100, 200, 200], 'note': ''},
                {'kind': 'd8', 'value': 3, 'bbox': [200, 200, 300, 300],
                 'note': '', 'extra': 'ignored'},
                {'kind': 'd12', 'value': 5, 'bbox': [300, 300, 200, 200], 'note': ''},
            ],
            'warning': 'x' * 600,
        }
        with patch.object(app, 'PROVIDER', 'deepseek'), patch.object(app, 'REFINE', False), patch.object(deepseek_client, 'recognize', return_value=(json.dumps(payload), {})):
            result = app.infer('image-bytes', 'auto')
        self.assertFalse(app.LOCK.locked())
        self.assertEqual([(d['kind'], d['value']) for d in result['dice']],
                         [('d6', None), ('d10', 10), ('d8', 3)])
        self.assertLessEqual(len(result['warning']), 500)


if __name__ == '__main__':
    unittest.main()
