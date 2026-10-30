import hashlib
import json
import math
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / '.runtime' / 'dice-training'
KINDS = ['d4', 'd6', 'd8', 'd10', 'd12', 'd20', 'd00']
VALUES = {k: list(range(1, int(k[1:]) + 1)) for k in KINDS if k not in ('d10', 'd00')}
VALUES.update(d10=list(range(10)), d00=list(range(0, 100, 10)))
CLASSES = [f'{kind}_{v}' for kind in KINDS for v in VALUES[kind]]


@contextmanager
def corpus_lock():
    """Serialize read/modify/write across downloader and annotation processes."""
    DATA.mkdir(parents=True,exist_ok=True)
    with (DATA/'records.lock').open('a+b') as stream:
        stream.seek(0,2)
        if not stream.tell():stream.write(b'0');stream.flush()
        stream.seek(0)
        if __import__('os').name=='nt':
            import msvcrt
            msvcrt.locking(stream.fileno(),msvcrt.LK_LOCK,1)
            try:yield
            finally:
                stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
        else:
            import fcntl
            fcntl.flock(stream,fcntl.LOCK_EX)
            try:yield
            finally:fcntl.flock(stream,fcntl.LOCK_UN)


def read_records(path=None):
    path = Path(path or DATA / 'records.jsonl')
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()] if path.exists() else []


def write_records(rows, path=None):
    path = Path(path or DATA / 'records.jsonl')
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    temp.replace(path)


def safe_image(relative):
    base = (DATA / 'images').resolve()
    path = (base / relative).resolve()
    if not path.is_relative_to(base):
        raise ValueError('Invalid image path')
    return path


def validate_die(d):
    kind = d['type']
    if kind not in KINDS:
        raise ValueError('Unknown dice type')
    b = d['box']
    vals = [b[k] for k in ('x', 'y', 'w', 'h')]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in vals):
        raise ValueError('Box coordinates must be finite numbers')
    x, y, w, h = vals
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x+w > 1.000001 or y+h > 1.000001:
        raise ValueError('Box must lie inside the image')
    value = d.get('value')
    if value is not None and (type(value) is not int or value not in VALUES[kind]):
        raise ValueError('Invalid face value: d10 uses printed 0–9; d00 uses 00–90')
    if d.get('value_trusted') is True and (value is None or d.get('unreadable', False)):
        raise ValueError('Unreadable or missing values cannot be trusted')
    return d


def group_split(group, seed=42):
    n = int(hashlib.sha256(f'{seed}:{group}'.encode()).hexdigest()[:8], 16) % 100
    return 'test' if n < 15 else 'val' if n < 30 else 'train'


def assign_splits(rows, seed=42):
    """Greedy group allocation by frame count, without consulting face labels."""
    from collections import Counter
    counts=Counter(r['group'] for r in rows)
    target={'train':len(rows)*.7,'val':len(rows)*.15,'test':len(rows)*.15}
    used={s:0 for s in target}; result={}
    order=sorted(counts,key=lambda g:(-counts[g],hashlib.sha256(f'{seed}:{g}'.encode()).hexdigest()))
    for group in order:
        split=max(target,key=lambda s:target[s]-used[s])
        result[group]=split;used[split]+=counts[group]
    return result


def square_crop(image, box, margin=.12):
    from PIL import Image
    x,y,w,h = (box[k] for k in ('x','y','w','h'))
    cx, cy = (x+w/2)*image.width, (y+h/2)*image.height
    side = max(w*image.width, h*image.height)*(1+2*margin)
    left=max(0,min(image.width-1,int(cx-side/2)))
    top=max(0,min(image.height-1,int(cy-side/2)))
    right=max(left+1,min(image.width,int(cx+side/2+.5)))
    bottom=max(top+1,min(image.height,int(cy+side/2+.5)))
    crop = image.crop((left,top,right,bottom))
    out = Image.new('RGB', (max(crop.size),)*2, (114,114,114))
    out.paste(crop, ((out.width-crop.width)//2,(out.height-crop.height)//2))
    return out
