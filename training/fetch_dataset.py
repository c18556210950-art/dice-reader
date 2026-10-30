"""Download and pin an attributable public dataset; never trust teacher face labels."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import PurePosixPath
from urllib.parse import quote

from download_model import fetch
from training.common import DATA, KINDS, safe_image, write_records, read_records, validate_die, corpus_lock

REPO = 'G-G-Games/diecamera-frames'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=0, help='0 downloads all frames')
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    source = DATA / 'sources' / 'diecamera'
    source.mkdir(parents=True, exist_ok=True)
    lock = source / 'revision.json'
    if not lock.exists():
        fetch(f'https://huggingface.co/api/datasets/{REPO}', source / 'repository.json')
        revision = json.loads((source / 'repository.json').read_text(encoding='utf-8'))['sha']
        lock.write_text(json.dumps({'repo': REPO, 'revision': revision}), encoding='utf-8')
    revision = json.loads(lock.read_text())['revision']
    base = f'https://huggingface.co/datasets/{REPO}/resolve/{revision}/'
    for name in ('README.md', 'metadata.jsonl'):
        if not (source / name).exists():
            fetch(base + name, source / name)
    raw = read_records(source / 'metadata.jsonl')
    if args.limit:
        raw = raw[:args.limit]

    def one(r):
        rel = PurePosixPath(r['file_name'])
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('Unsafe source path')
        # The actual metadata already includes data/, despite an older card saying otherwise.
        relative = 'diecamera/' + str(rel)
        image = safe_image(relative)
        image.parent.mkdir(parents=True, exist_ok=True)
        if not image.exists():
            temp = image.with_suffix('.download')
            fetch(base + quote(str(rel), safe='/'), temp)
            from PIL import Image
            with Image.open(temp) as im:
                im.verify()
            temp.replace(image)
        dice = []
        for d in r['dice']:
            d = dict(d)
            if d['type'] == 'd100':
                d['type'] = 'd00'
            if d['type'] not in KINDS:
                raise ValueError(f"Unknown source kind {d['type']}")
            # Store printed d10 zero as zero, normalize to ten only when summing.
            if d['type'] == 'd10' and d.get('value') == 10:
                d['value'] = 0
            if d['type'] == 'd00' and d.get('value') == 100:
                d['value'] = 0
            d['value_trusted'] = r.get('values_trusted') is True
            validate_die(d)
            dice.append(d)
        return dict(id='dc-' + r['id'], image=relative, group='diecamera/' + str(rel.parent),
                    source=REPO, revision=revision, license='CC-BY-SA-4.0',
                    box_reviewed=True, human_reviewed=False, dice=dice,
                    sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
                    label_source=r.get('label_source'), captured_at=r.get('captured_at'))

    old = {}
    def persist():
        with corpus_lock():
            latest={r['id']:r for r in read_records()}
            for key,row in old.items():
                if not latest.get(key,{}).get('human_reviewed'):
                    latest[key]=row
            write_records(sorted(latest.values(),key=lambda r:r['id']))
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(one, r): r for r in raw}
        for i, job in enumerate(concurrent.futures.as_completed(jobs), 1):
            try:
                r = job.result()
                if not old.get(r['id'], {}).get('human_reviewed'):
                    old[r['id']] = r
            except Exception as exc:
                failures.append({'id': jobs[job]['id'], 'error': str(exc)})
            if i % 20 == 0:
                persist()
                print(f'{i}/{len(raw)} frames; failures={len(failures)}', flush=True)
    persist()
    (source/'failures.json').write_text(json.dumps(failures,indent=2),encoding='utf-8')
    print(f'Imported {len(old)} records; failures={len(failures)}', flush=True)
    if failures:
        raise SystemExit('Some files failed; rerun to resume. See sources/diecamera/failures.json')


if __name__ == '__main__':
    main()
