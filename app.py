"""D&D dice reader with local and explicitly configured cloud inference."""
import base64
import binascii
import io
import json
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

ROOT = Path(__file__).resolve().parent
MODEL = "qwen3-vl:4b"
PROVIDER = os.environ.get("DICE_PROVIDER", "local")
if PROVIDER not in {"local", "deepseek"}:
    raise RuntimeError("Unsupported DICE_PROVIDER")
# A dedicated loopback service; no configurable external endpoint or cloud models.
OLLAMA = "http://127.0.0.1:11435"
LOCAL_HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))
LOCK = threading.Lock()
PUBLIC_ORIGIN = os.environ.get("DICE_PUBLIC_ORIGIN", "")
SERVER_MODE = os.environ.get("DICE_SERVER_MODE") == "1"
Image.MAX_IMAGE_PIXELS = 32_000_000
KINDS = {"d4": (1, 4), "d6": (1, 6), "d8": (1, 8), "d10": (1, 10),
         "d12": (1, 12), "d20": (1, 20), "d00": (0, 90), "unknown": (0, 100)}
# Refine each detected die on a magnified crop; one global image loses too much detail.
REFINE = os.environ.get("DICE_DEEPSEEK_REFINE", "1") != "0"
try:
    WORKERS = max(1, min(12, int(os.environ.get("DICE_DEEPSEEK_WORKERS", "6"))))
except ValueError:
    WORKERS = 6
# One client cannot start recognitions faster than this, whatever the page does.
RATE_WINDOW = max(10, int(os.environ.get("DICE_RATE_WINDOW", "60")))
RATE_MAX = max(1, int(os.environ.get("DICE_RATE_MAX", "10")))
HITS = {}
HITS_LOCK = threading.Lock()


def client_ip(request):
    # nginx overwrites both headers, so they cannot be spoofed through the web.
    real = request.headers.get("x-real-ip", "").strip()
    if real:
        return real
    forwarded = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    if forwarded:
        return forwarded
    return request.client.host if request.client else "unknown"


def allow_request(ip):
    now = time.monotonic()
    cutoff = now - RATE_WINDOW
    with HITS_LOCK:
        recent = [stamp for stamp in HITS.get(ip, ()) if stamp > cutoff]
        if len(recent) >= RATE_MAX:
            HITS[ip] = recent
            return False
        recent.append(now)
        HITS[ip] = recent
        if len(HITS) > 5000:
            for key in [key for key, stamps in HITS.items() if not stamps or stamps[-1] <= cutoff]:
                del HITS[key]
        return True


class Die(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    kind: str
    value: int | None
    bbox: list[int] = Field(min_length=4, max_length=4)
    note: str = Field(max_length=240)

    @model_validator(mode="after")
    def validate_die(self):
        if self.kind not in KINDS:
            raise ValueError("不支持的骰子类型")
        x1, y1, x2, y2 = self.bbox
        if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
            raise ValueError("骰子位置无效")
        if self.value is not None:
            lo, hi = KINDS[self.kind]
            if not lo <= self.value <= hi or (self.kind == "d00" and self.value % 10):
                raise ValueError("点数超出骰子范围")
        return self


class DiceResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    dice: list[Die] = Field(max_length=80)
    warning: str = Field(max_length=500)


PROMPT = """Inspect this overhead photograph of a Dungeons & Dragons dice tray.
Find each physical die exactly once, ignoring tray decorations and text elsewhere.
Read the actual roll result, not every visible number. For d6/d8/d10/d12/d20 read
the upper face. Do not sum side faces. A normal d10 marked 0 has roll value 10.
Values must stay inside the die range: d4 1-4, d6 1-6, d8 1-8, d10 1-10,
d12 1-12, d20 1-20.
A percentile tens die is kind d00, with value 0,10,20,...90 (00 is 0).
Do not pair percentile dice or invent a d100 value. d4 reading convention: {d4}.
For d4 never assume the largest visible digit is the result. If the result is
ambiguous, occluded, blurred, cocked, or has unresolved 6/9 orientation, use null.
For each die return kind from d4,d6,d8,d10,d12,d20,d00,unknown, value integer or
null, bbox [left,top,right,bottom] on 0..1000 normalized image coordinates, and
a short Chinese note explaining ambiguity. If unsure of kind use unknown.
No dice means an empty dice list. Return JSON matching the schema, with a short
Chinese warning for limitations. Never fabricate hidden dice or numbers.
"""

DIE_PROMPT = """This image is a magnified crop from an overhead dice-tray photo and
shows one die, centered. Identify its type and read the roll result from the face
pointing up at the camera. Kinds and ranges: d4 1-4, d6 1-6, d8 1-8, d10 1-10
(a face marked 0 counts as 10), d12 1-12, d20 1-20, d00 0,10,...90. A d10 must
never be reported as d20 or d00. Read only the upper face, never sum side faces.
d4 reading convention: {d4}. If the upper face is hidden, blurred, cocked or the
6/9 orientation is unresolved, use value null. Ignore other dice at the edges.
Return only JSON: {{"kind": one of d4,d6,d8,d10,d12,d20,d00,unknown,
"value": integer or null, "note": "short Chinese note"}}.
"""

DIE_SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"type": "string",
                 "enum": ["d4", "d6", "d8", "d10", "d12", "d20", "d00", "unknown"]},
        "value": {"type": ["integer", "null"]},
        "note": {"type": "string"},
    },
    "required": ["kind", "value", "note"],
    "additionalProperties": False,
}


def _text(value, limit):
    return value[:limit] if isinstance(value, str) else ""


def _int(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().lstrip("+-").isdigit():
        return int(value.strip())
    return value if isinstance(value, int) else None


def _bbox(value):
    if not isinstance(value, list) or len(value) != 4:
        return None
    coords = [_int(item) for item in value]
    if any(item is None for item in coords):
        return None
    x1, y1, x2, y2 = (max(0, min(1000, item)) for item in coords)
    return [x1, y1, x2, y2] if x1 < x2 and y1 < y2 else None


def normalize_die(raw, bbox=None):
    """Repair one model-produced die instead of failing the whole roll."""
    if not isinstance(raw, dict):
        return None
    kind = raw.get("kind")
    kind = kind if isinstance(kind, str) and kind in KINDS else "unknown"
    value = _int(raw.get("value"))
    if kind == "d10" and value == 0:
        value = 10
    if value is not None:
        low, high = KINDS[kind]
        if not low <= value <= high or (kind == "d00" and value % 10):
            value = None
    box = bbox or _bbox(raw.get("bbox"))
    if box is None:
        return None
    return {"kind": kind, "value": value, "bbox": box,
            "note": _text(raw.get("note"), 240)}


def parse_result(content):
    data = json.loads(content)
    raw = data.get("dice") if isinstance(data, dict) else None
    raw = raw[:80] if isinstance(raw, list) else []
    dice = [die for die in (normalize_die(item) for item in raw) if die]
    result = DiceResult.model_validate({
        "dice": dice,
        "warning": _text(data.get("warning") if isinstance(data, dict) else "", 500)})
    dropped = len(raw) - len(dice)
    if dropped:
        result.warning = (f"有 {dropped} 颗骰子格式不完整，已忽略；" + result.warning)[:500]
    return result


def _die_crop(source, bbox):
    width, height = source.size
    x1, y1, x2, y2 = bbox
    pad_x, pad_y = (x2 - x1) * 0.18, (y2 - y1) * 0.18
    left = max(0, int((x1 - pad_x) * width / 1000))
    top = max(0, int((y1 - pad_y) * height / 1000))
    right = min(width, int((x2 + pad_x) * width / 1000))
    bottom = min(height, int((y2 + pad_y) * height / 1000))
    if right - left < 24 or bottom - top < 24:
        return None
    try:
        crop = source.crop((left, top, right, bottom))
        # Tiny crops are upscaled so the die still fills the vision encoder.
        if max(crop.size) < 360:
            scale = 360 / max(crop.size)
            crop = crop.resize((round(crop.width * scale), round(crop.height * scale)),
                               Image.LANCZOS)
        buffer = io.BytesIO()
        crop.save(buffer, format="JPEG", quality=95)
        return base64.b64encode(buffer.getvalue()).decode()
    except (OSError, ValueError):
        return None


def refine(image, result, rule, started):
    """Re-read every die on its own magnified crop at temperature 0.

    The global image is compressed by the API to a small token budget (about
    800 image tokens for 1600x1206), so each die gets too few pixels. A crop
    makes one die fill the frame, which measured 6/6 correct on the test tray.
    Crops are independent, so they run concurrently; a 10-die roll takes two
    waves instead of ten sequential round trips.
    """
    import deepseek_client
    try:
        with Image.open(io.BytesIO(base64.b64decode(image))) as source:
            source = ImageOps.exif_transpose(source)
            source.load()
            source = source.convert("RGB")
    except (binascii.Error, ValueError, OSError, UnidentifiedImageError):
        return result, {}
    jobs = []
    for index, die in enumerate(result.dice):
        encoded = _die_crop(source, die.bbox)
        if encoded:
            jobs.append((index, die, encoded))
    if not jobs:
        return result, {}

    def read_one(job):
        _, die, encoded = job
        try:
            content, used = deepseek_client.recognize(
                encoded, DIE_PROMPT.format(d4=rule), DIE_SCHEMA, timeout=45)
        except HTTPException as exc:
            return None, None, exc.status_code
        except Exception:
            return None, None, None
        try:
            return normalize_die(json.loads(content), die.bbox), used, None
        except ValueError:
            return None, None, None

    workers = min(WORKERS, len(jobs))
    outcomes = {}
    stopped = False
    pool = ThreadPoolExecutor(max_workers=workers)
    futures = {pool.submit(read_one, job): job for job in jobs}
    try:
        for future in as_completed(futures, timeout=max(5, 110 - (time.monotonic() - started))):
            outcomes[futures[future][0]] = future.result()
    except TimeoutError:
        stopped = True
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    refined = 0
    extra = {}
    for index, die, _ in jobs:
        answer, used, status = outcomes.get(index, (None, None, "timeout"))
        if status is not None or answer is None:
            stopped = True
            continue
        kind = answer["kind"] if answer["kind"] != "unknown" else die.kind
        value = answer["value"]
        if answer["kind"] == "unknown" and value is None:
            value = die.value
        merged = normalize_die({"kind": kind, "value": value,
                                "note": answer["note"] or die.note}, die.bbox)
        if merged is None:
            continue
        die.kind, die.value, die.note = merged["kind"], merged["value"], merged["note"]
        refined += 1
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            if isinstance(used.get(key), int):
                extra[key] = extra.get(key, 0) + used[key]
    if refined:
        unread = sum(1 for die in result.dice if die.value is None)
        note = f"已逐颗放大复核 {refined} 颗骰子"
        if stopped or refined < len(result.dice):
            note += f"（共 {len(result.dice)} 颗，部分未复核）"
        if unread:
            note += f"，{unread} 颗读数留空"
        # The first-pass warning may describe dice this pass has since read correctly,
        # but a dropped-die notice is still relevant.
        if result.warning.startswith("有 "):
            note = result.warning + note
        result.warning = (note + "；请核对骰子数量与每颗读数。")[:500]
    return result, extra


def ollama_json(path, payload=None, timeout=5):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(OLLAMA + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with LOCAL_HTTP.open(req, timeout=timeout) as response:
        return json.load(response)


def prepare_image(encoded):
    try:
        raw = base64.b64decode(encoded, validate=True)
        if len(raw) > 15 * 1024 * 1024:
            raise HTTPException(413, "照片不能超过 15MB")
        with Image.open(io.BytesIO(raw)) as image:
            if image.width * image.height > 32_000_000:
                raise ValueError("image too large")
            image = ImageOps.exif_transpose(image)
            image.load()
            image = image.convert("RGB")
            image.thumbnail((1600, 1600))
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=94)
            return base64.b64encode(output.getvalue()).decode()
    except HTTPException:
        raise
    except (binascii.Error, ValueError, OSError, UnidentifiedImageError,
            Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise HTTPException(400, "无法读取照片，请使用 JPG、PNG 或 WebP 图片")


app = FastAPI(title="骰盘 · DND 骰子识别", docs_url=None, redoc_url=None)


@app.middleware("http")
async def local_requests_only(request: Request, call_next):
    # Reject cross-site browser access to this unauthenticated local service.
    host = request.headers.get("host", "")
    origin = request.headers.get("origin")
    if host not in {"127.0.0.1:8765", "localhost:8765", "testserver"} or (
        origin and origin not in {"http://127.0.0.1:8765", "http://localhost:8765", PUBLIC_ORIGIN}
    ):
        from fastapi.responses import JSONResponse
        return JSONResponse({"detail": "仅允许本机页面访问"}, status_code=403)
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    # The BYOK page calls the user's own model provider directly from the browser.
    connect_src = ("'self' https: http://localhost:* http://127.0.0.1:*"
                   if request.url.path.endswith("/byok.html") else "'self'")
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; img-src 'self' blob: data:; "
        "script-src 'self'; style-src 'self' 'unsafe-inline'; "
        f"connect-src {connect_src}; frame-ancestors 'none'"
    )
    return response


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/status")
def status():
    if PROVIDER == "deepseek":
        import deepseek_client
        ready = deepseek_client.configured()
        return {"ready": ready, "provider": "deepseek", "model": deepseek_client.MODEL,
                "message": "识别服务已配置（测试版，请逐颗核对）" if ready
                else "识别服务尚未配置"}
    try:
        models = ollama_json("/api/tags").get("models", [])
        ready = any(m.get("name") == MODEL for m in models)
        if not ready:
            try:
                state = json.loads((ROOT / ".runtime/hf-download-status.json").read_text(encoding="utf-8"))
                total = state["total"]
                received = state["downloaded"]
                return {"ready": False, "model": MODEL,
                        "message": f"官方 4B 模型准备中：文件已下载 {received / total:.1%}；完成后可自动识别"}
            except (OSError, KeyError, ValueError, ZeroDivisionError):
                pass
            try:
                manifest = json.loads((ROOT / ".runtime" / "manifest.json").read_text(encoding="utf-8"))
                entries = [manifest["config"], *manifest["layers"]]
                total = sum(e["size"] for e in entries)
                downloaded = 0
                for entry in entries:
                    path = ROOT / ".runtime" / "models" / "blobs" / entry["digest"].replace(":", "-")
                    candidate = path if path.exists() else path.with_suffix(".download")
                    if candidate.exists():
                        downloaded += min(candidate.stat().st_size, entry["size"])
                return {"ready": False, "model": MODEL,
                        "message": f"本地模型准备中：文件已下载 {downloaded / total:.1%}；完成后可自动识别"}
            except (OSError, KeyError, ValueError, ZeroDivisionError):
                pass
        return {"ready": ready, "model": MODEL, "message": "电脑上的识别模型已就绪" if ready
                else ("电脑上的视觉模型尚未准备完成，自动识别暂不可用" if SERVER_MODE
                      else "模型尚未下载，请先运行 setup.ps1")}
    except (urllib.error.URLError, OSError, ValueError):
        return {"ready": False, "model": MODEL,
                "message": "计算电脑尚未连接，请在电脑上运行“连接服务器”" if SERVER_MODE
                else "本地识别服务未启动，请先运行 setup.ps1，再运行 start.ps1"}


@app.post("/api/recognize")
async def recognize(request: Request):
    ip = client_ip(request)
    if not allow_request(ip):
        raise HTTPException(429, "请求过于频繁，请稍后再试",
                            headers={"Retry-After": str(RATE_WINDOW)})
    # Bound the body while streaming, including requests without Content-Length.
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > 21 * 1024 * 1024:
            raise HTTPException(413, "照片不能超过 15MB")
    try:
        data = json.loads(body)
        encoded = data["image"]
        convention = data.get("d4", "auto")
        if not isinstance(encoded, str) or convention not in {"auto", "top", "bottom"}:
            raise ValueError()
    except (ValueError, KeyError, TypeError):
        raise HTTPException(400, "请求格式不正确")
    image = prepare_image(encoded)
    # Inference is blocking; run it outside the event loop so status/UI stay responsive.
    from starlette.concurrency import run_in_threadpool
    return await run_in_threadpool(infer, image, convention)


def infer(image, convention):
    if not LOCK.acquire(blocking=False):
        raise HTTPException(409, "正在识别另一张照片，请稍候")
    started = time.monotonic()
    try:
        rule = {"auto": "infer top-reading or bottom-reading; if uncertain use null",
                "top": "read the common digit at the top vertex",
                "bottom": "read the common digit along the bottom edges"}[convention]
        usage = {}
        if PROVIDER == "deepseek":
            import deepseek_client
            content, usage = deepseek_client.recognize(image, PROMPT.format(d4=rule), DiceResult.model_json_schema())
        else:
            response = ollama_json("/api/chat", {
                "model": MODEL, "stream": False, "think": False,
                "messages": [{"role": "user", "content": PROMPT.format(d4=rule),
                              "images": [image]}],
                "format": DiceResult.model_json_schema(),
                "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 4096},
                "keep_alive": "5m",
            }, timeout=300)
            if response.get("done_reason") == "length":
                raise HTTPException(502, "模型达到输出上限，未能完成识别，请稍后重试或手动录入")
            content = response["message"]["content"]
        if PROVIDER == "deepseek":
            result = parse_result(content)
            if REFINE and result.dice:
                result, extra = refine(image, result, rule, started)
                for key, value in extra.items():
                    usage[key] = usage.get(key, 0) + value
            result.warning = ("测试版自动读数，请核对骰子数量、类型、结果面及每颗读数。"
                              + result.warning)[:500]
        else:
            result = DiceResult.model_validate_json(content)
        # All predictions require human review; no invented model confidence score.
        return {**result.model_dump(), "seconds": round(time.monotonic() - started, 1),
                "provider": PROVIDER, "usage": usage}
    except HTTPException:
        raise
    except urllib.error.HTTPError:
        raise HTTPException(503, "本地模型无法运行，请检查模型下载情况及本地服务日志")
    except (urllib.error.URLError, OSError):
        raise HTTPException(503, "本地模型连接失败或超时，请检查服务；可稍后重试")
    except (KeyError, ValueError, ValidationError):
        raise HTTPException(502, "模型返回的结果无法校验，请重拍或手动录入")
    finally:
        LOCK.release()


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8765, access_log=False)
