"""Fetch official Ollama blobs with Windows curl TLS; verify every SHA256."""
import concurrent.futures
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent / ".runtime"
REGISTRY = "https://registry.ollama.ai/v2/library/qwen3-vl"


def download_proxy():
    """Use the user's existing proxy without changing system or inference settings."""
    if 'DICE_DOWNLOAD_PROXY' in os.environ:
        return os.environ['DICE_DOWNLOAD_PROXY'].strip()
    if os.name == 'nt':
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                    r'Software\Microsoft\Windows\CurrentVersion\Internet Settings') as key:
                if not winreg.QueryValueEx(key, 'ProxyEnable')[0]:
                    return ''
                value = winreg.QueryValueEx(key, 'ProxyServer')[0]
                if '=' in value:
                    schemes = dict(item.split('=', 1) for item in value.split(';') if '=' in item)
                    value = schemes.get('https', schemes.get('http', ''))
                if value:
                    return value if '://' in value else 'http://' + value
        except OSError:
            pass
    return os.environ.get('HTTPS_PROXY', os.environ.get('https_proxy', ''))


def fetch(url, target, resume=False):
    command = ["curl.exe", "--fail", "--location", "--silent", "--show-error",
               "--retry", "3", "--retry-all-errors", "--connect-timeout", "30", "--speed-limit", "1024",
               "--speed-time", "90"]
    proxy = download_proxy()
    if proxy:
        command += ["--proxy", proxy]
    if resume:
        command += ["--continue-at", "-"]
    subprocess.run(command + ["--output", str(target), url], check=True)


def checksum(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    ROOT.mkdir(exist_ok=True)
    print('Download route: existing proxy' if download_proxy() else 'Download route: direct', flush=True)
    manifest_path = ROOT / "manifest.json"
    fetch(REGISTRY + "/manifests/4b", manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    blobs = ROOT / "models" / "blobs"
    blobs.mkdir(parents=True, exist_ok=True)
    entries = [manifest["config"], *manifest["layers"]]
    for entry in entries:
        if not re.fullmatch(r"sha256:[a-f0-9]{64}", entry["digest"]):
            raise ValueError("Unexpected official manifest digest")

    def download(entry):
        digest = entry["digest"]
        target = blobs / digest.replace(":", "-")
        if target.exists() and target.stat().st_size == entry["size"] and checksum(target) == digest[7:]:
            return
        partial = target.with_suffix(".download")
        if not partial.exists() or partial.stat().st_size != entry["size"]:
            fetch(REGISTRY + "/blobs/" + digest, partial, resume=True)
        if partial.stat().st_size != entry["size"] or checksum(partial) != digest[7:]:
            # Retain the failed file for inspection, but do not install it.
            partial.replace(partial.with_suffix(".bad"))
            raise ValueError("Model integrity check failed; run setup again")
        partial.replace(target)

    total = sum(e["size"] for e in entries)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(download, entry) for entry in entries]
        while not all(f.done() for f in futures):
            sizes = []
            for entry in entries:
                target = blobs / entry["digest"].replace(":", "-")
                candidate = target if target.exists() else target.with_suffix(".download")
                sizes.append(min(candidate.stat().st_size, entry["size"]) if candidate.exists() else 0)
            print(f"Model download: {sum(sizes)/total:.1%} ({sum(sizes)/1e9:.2f}/{total/1e9:.2f} GB)", flush=True)
            time.sleep(5)
        for future in futures:
            future.result()
    destination = ROOT / "models" / "manifests" / "registry.ollama.ai" / "library" / "qwen3-vl" / "4b"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest), encoding="utf-8")
    temporary.replace(destination)
    print("Official model downloaded; all file hashes verified.", flush=True)


if __name__ == "__main__":
    main()
