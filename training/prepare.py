"""Export YOLO boxes and trusted face crops with shared, group-isolated splits."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import shutil
from pathlib import Path

from PIL import Image, ImageOps
from training.common import DATA, KINDS, CLASSES, read_records, safe_image, assign_splits, square_crop, validate_die


def prepare(output=None, seed=42):
    out = Path(output or DATA / 'prepared')
    if out.exists():
        raise ValueError('Output exists. Choose a new --output; old exports and training runs are preserved.')
    rows = read_records()
    if not rows:
        raise ValueError('No records. Download or annotate images first.')
    counts = {s:Counter() for s in ('train','val','test')}
    frames = Counter()
    groups = defaultdict(set)
    splits = {}
    hashes = {}
    assigned=assign_splits(rows,seed)
    # Equal images from distinct groups must not leak across a partition.
    for r in rows:
        path = safe_image(r['image'])
        if not path.exists():
            raise ValueError(f'Missing image: {path}')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        split = assigned[r['group']]
        hashes.setdefault(digest, set()).add(split)
        splits[r['id']] = (split, digest)
        for d in r['dice']:
            validate_die(d)
    conflicts = {h for h, s in hashes.items() if len(s)>1}
    seen = set()
    excluded = []
    for s in ('train','val','test'):
        for part in ('images','labels'):
            (out/'detector'/part/s).mkdir(parents=True,exist_ok=True)
        for label in CLASSES:
            (out/'classifier'/s/label).mkdir(parents=True,exist_ok=True)
    for r in rows:
        split, digest = splits[r['id']]
        if digest in conflicts or digest in seen or not r.get('box_reviewed'):
            excluded.append(r['id'])
            continue
        seen.add(digest)
        # Filenames are content hashes, never untrusted record IDs.
        stem = digest[:24]
        with Image.open(safe_image(r['image'])) as im:
            image = ImageOps.exif_transpose(im).convert('RGB')
        image.save(out/'detector'/'images'/split/(stem+'.jpg'), quality=95)
        labels = []
        for i,d in enumerate(r['dice']):
            b=d['box']; x,y,w,h=(b[k] for k in ('x','y','w','h'))
            labels.append(f'{KINDS.index(d["type"])} {x+w/2:.8f} {y+h/2:.8f} {w:.8f} {h:.8f}')
            if d.get('value_trusted') is True and not d.get('unreadable',False):
                label = f'{d["type"]}_{d["value"]}'
                square_crop(image,b).save(out/'classifier'/split/label/f'{stem}-{i}.jpg',quality=95)
                counts[split][label]+=1
        (out/'detector'/'labels'/split/(stem+'.txt')).write_text('\n'.join(labels),encoding='utf-8')
        frames[split]+=1
        groups[split].add(r['group'])
    # JSON is valid YAML, so a YAML dependency is unnecessary for this export.
    (out/'detector'/'data.yaml').write_text(json.dumps(dict(path=(out/'detector').resolve().as_posix(),
        train='images/train',val='images/val',test='images/test',names=dict(enumerate(KINDS))),indent=2),encoding='utf-8')
    report=dict(seed=seed, frames=frames, groups={s:sorted(g) for s,g in groups.items()},
        faces={s:dict(c) for s,c in counts.items()}, excluded=excluded,
        absent_training_classes=[c for c in CLASSES if not counts['train'][c]],
        split_by_record={r['id']:splits[r['id']][0] for r in rows},
        sources=sorted({r.get('source','personal') for r in rows}))
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    shutil.copyfile(DATA/'records.jsonl',out/'records.snapshot.jsonl')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return report


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output'); p.add_argument('--seed',type=int,default=42)
    a=p.parse_args(); prepare(a.output,a.seed)
