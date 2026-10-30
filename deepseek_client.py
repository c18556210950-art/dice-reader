"""Server-only DeepSeek vision adapter. Credentials never leave request headers."""
import json
import os
import urllib.error
import urllib.request

from fastapi import HTTPException

MODEL = 'deepseek-flash'
ENDPOINT = 'https://api.deepseek.com/chat/completions'


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())


def configured():
    return bool(os.environ.get('DEEPSEEK_API_KEY', '').strip())


def recognize(image, prompt, schema, timeout=180):
    key = os.environ.get('DEEPSEEK_API_KEY', '').strip()
    if not key:
        raise HTTPException(503, '服务器尚未配置 DeepSeek 密钥')
    thinking = os.environ.get('DICE_DEEPSEEK_THINKING', 'disabled') == 'enabled'
    payload = {
        'model': MODEL, 'stream': False,
        'thinking': {'type': 'enabled' if thinking else 'disabled'},
        # DeepSeek defaults to 1.0; identical photos then read differently every call.
        'temperature': 0, 'top_p': 1,
        'max_tokens': 8192 if thinking else 4096,
        'response_format': {'type': 'json_object'},
        'messages': [{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': prompt + '\nReturn only a JSON object. JSON schema:\n' + json.dumps(schema, ensure_ascii=False)},
                {'type': 'image_url', 'image_url': {
                    'url': 'data:image/jpeg;base64,' + image}},
            ],
        }],
    }
    if thinking:
        payload['reasoning_effort'] = 'low'
    request = urllib.request.Request(ENDPOINT,
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key})
    try:
        with HTTP.open(request, timeout=timeout) as response:
            data = json.load(response)
    except urllib.error.HTTPError as exc:
        messages = {
            400: 'DeepSeek 拒绝了识别请求，请检查图片模型及接口配置',
            401: 'DeepSeek 密钥验证失败，请检查服务器配置',
            402: 'DeepSeek 账户余额不足，请充值后重试',
            429: 'DeepSeek 请求过于频繁，请稍后重试',
        }
        # Do not echo upstream responses or request headers to the browser/logs.
        raise HTTPException(503, messages.get(exc.code, 'DeepSeek 服务暂不可用，请稍后重试')) from None
    except (urllib.error.URLError, OSError):
        raise HTTPException(503, '连接 DeepSeek 失败或超时，请稍后重试') from None
    try:
        choice = data['choices'][0]
        if choice.get('finish_reason') == 'length':
            raise HTTPException(502, 'DeepSeek 达到输出上限，未能完成识别，请手动核对')
        content = choice['message']['content']
        if not isinstance(content, str) or not content.strip():
            raise ValueError('empty output')
        return content, data.get('usage', {})
    except (KeyError, IndexError, TypeError, ValueError):
        raise HTTPException(502, 'DeepSeek 未返回有效识别结果，请重试或手动录入') from None
