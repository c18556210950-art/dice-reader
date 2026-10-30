"""Download Qwen's official matched GGUF pair; verify hashes before registration."""
import concurrent.futures
import copy
import json
import subprocess
import time
from pathlib import Path

from download_model import ROOT, REGISTRY, checksum, download_proxy, fetch

FILES = [
    {"name": "Qwen3VL-4B-Instruct-Q4_K_M.gguf", "size": 2497281664,
     "sha256": "66358cb18bb6b3b1b6675aa412c7a88ef01d228f481184d13668e5201c730a0a",
     "mediaType": "application/vnd.ollama.image.model"},
    {"name": "mmproj-Qwen3VL-4B-Instruct-F16.gguf", "size": 836180256,
     "sha256": "256f3a43bd4205ffef48d6b92715e1e70b5b0e9aef06522584967513a9985331",
     "mediaType": "application/vnd.ollama.image.projector"},
]
SOURCE = 'https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF/resolve/main/'
STATUS = ROOT / 'hf-download-status.json'


def paths(entry):
    target = ROOT / 'models' / 'blobs' / ('sha256-' + entry['sha256'])
    return target, target.with_suffix('.download')


def download(entry):
    target, partial = paths(entry)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == entry['size'] and checksum(target) == entry['sha256']:
        return
    for attempt in range(1, 31):
        if partial.exists() and partial.stat().st_size == entry['size']:
            break
        command = ['curl.exe', '--fail', '--location', '--silent', '--show-error',
                   '--connect-timeout', '20', '--max-time', '600', '--speed-limit',
                   '4096', '--speed-time', '60', '--continue-at', '-', '--output', str(partial)]
        proxy = download_proxy()
        if proxy:
            command += ['--proxy', proxy]
        # A fresh resolve request avoids reusing an expired temporary storage URL.
        result = subprocess.run(command + [SOURCE + entry['name'] + '?download=true&attempt=' + str(int(time.time()))])
        if result.returncode == 0:
            break
        print(f"Retrying {entry['name']}, attempt {attempt}; keeping received bytes.", flush=True)
        time.sleep(3)
    if not partial.exists() or partial.stat().st_size != entry['size']:
        raise RuntimeError('Download incomplete: ' + entry['name'])
    print('Verifying SHA256: ' + entry['name'], flush=True)
    if checksum(partial) != entry['sha256']:
        partial.replace(partial.with_suffix('.bad'))
        raise RuntimeError('Checksum mismatch: ' + entry['name'])
    partial.replace(target)
    print('Verified: ' + entry['name'], flush=True)


def progress(started, initial):
    received = 0
    for entry in FILES:
        target, partial = paths(entry)
        candidate = target if target.exists() else partial
        received += min(candidate.stat().st_size, entry['size']) if candidate.exists() else 0
    total = sum(e['size'] for e in FILES)
    speed = max(0, received - initial) / max(time.monotonic() - started, 1)
    state = {'source': 'Qwen official Hugging Face', 'downloaded': received,
             'total': total, 'bytes_per_second': round(speed), 'updated': time.time(),
             'phase': 'downloading'}
    temp = STATUS.with_suffix('.tmp')
    temp.write_text(json.dumps(state), encoding='utf-8')
    temp.replace(STATUS)
    print(f"Official 4B: {received/total:.1%}; {received/1e9:.2f}/{total/1e9:.2f} GB; {speed/1e6:.2f} MB/s", flush=True)
    return received


def register_and_test():
    # Retain the existing runner/parser metadata for this same Qwen3-VL 4B family.
    original = json.loads((ROOT / 'instruct-manifest.json').read_text(encoding='utf-8'))
    manifest = copy.deepcopy(original)
    for entry in FILES:
        for i, layer in enumerate(manifest['layers']):
            if layer['mediaType'] == entry['mediaType']:
                manifest['layers'][i] = {'mediaType': entry['mediaType'],
                    'digest': 'sha256:' + entry['sha256'], 'size': entry['size']}
    blobs = ROOT / 'models' / 'blobs'
    for entry in [manifest['config'], *manifest['layers']]:
        path = blobs / entry['digest'].replace(':', '-')
        if not path.exists() or checksum(path) != entry['digest'][7:]:
            raise RuntimeError('A required model component is missing or invalid')
    candidate = ROOT / 'models/manifests/registry.ollama.ai/library/dice-qwen3-vl/4b-instruct'
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_text(json.dumps(manifest), encoding='utf-8')
    from app import ollama_json, prepare_image
    import base64
    import io
    from PIL import Image
    stream = io.BytesIO()
    Image.new('RGB', (128, 128), 'white').save(stream, format='PNG')
    image = base64.b64encode(stream.getvalue()).decode()
    print('Testing local GPU vision inference; this is not a dice accuracy test.', flush=True)
    schema = {'type': 'object', 'properties': {'has_dice': {'type': 'boolean'}}, 'required': ['has_dice']}
    response = ollama_json('/api/chat', {
        'model': 'dice-qwen3-vl:4b-instruct', 'stream': False, 'think': False,
        'messages': [{'role': 'user', 'content': 'Does this plain image contain any dice? Return has_dice as a boolean.', 'images': [image]}],
        'format': schema, 'options': {'num_ctx': 4096, 'num_predict': 128, 'temperature': 0},
        'keep_alive': '5m',
    }, timeout=300)
    content = json.loads(response['message']['content'])
    if content.get('has_dice') is not False:
        raise RuntimeError('Vision smoke test did not correctly reject the blank image')
    destination = ROOT / 'models/manifests/registry.ollama.ai/library/qwen3-vl/4b'
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix('.tmp')
    temp.write_text(json.dumps(manifest), encoding='utf-8')
    temp.replace(destination)
    state = json.loads(STATUS.read_text(encoding='utf-8'))
    state.update(phase='ready', updated=time.time())
    STATUS.write_text(json.dumps(state), encoding='utf-8')
    print('Official 4B model installed; local vision smoke test passed.', flush=True)


def main():
    ROOT.mkdir(exist_ok=True)
    manifest_path = ROOT / 'instruct-manifest.json'
    if not manifest_path.exists():
        fetch(REGISTRY + '/manifests/4b-instruct', manifest_path)
    original = json.loads(manifest_path.read_text(encoding='utf-8'))
    for entry in [original['config'], *original['layers']]:
        if entry['mediaType'] in {e['mediaType'] for e in FILES}:
            continue
        target = ROOT / 'models/blobs' / entry['digest'].replace(':', '-')
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or checksum(target) != entry['digest'][7:]:
            fetch(REGISTRY + '/blobs/' + entry['digest'], target)
            if checksum(target) != entry['digest'][7:]:
                raise RuntimeError('Metadata checksum mismatch')
    started = time.monotonic()
    initial = sum((p.stat().st_size if p.exists() else 0) for e in FILES
                  for p in [paths(e)[0] if paths(e)[0].exists() else paths(e)[1]])
    print('Downloading official 4B + matched F16 visual encoder from Hugging Face via existing proxy.', flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(download, entry) for entry in FILES]
        while not all(f.done() for f in futures):
            progress(started, initial)
            time.sleep(5)
        for future in futures:
            future.result()
    progress(started, initial)
    register_and_test()


if __name__ == '__main__':
    # Keep two setup windows from writing the same partial model files.
    import msvcrt
    ROOT.mkdir(exist_ok=True)
    with (ROOT / 'hf-download.lock').open('a+b') as lock:
        lock.seek(0)
        lock.write(b'0')
        lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            raise SystemExit('The official 4B download is already running.')
        try:
            main()
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
