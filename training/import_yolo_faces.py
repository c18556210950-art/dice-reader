"""Import an exported Roboflow YOLO directory into a separate train-only face manifest."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from PIL import Image, ImageOps
import yaml
from training.common import DATA, VALUES, square_crop, validate_die


def parse_class(name):
    match=re.fullmatch(r'(d4|d6|d8|d10|d12|d20|d100|d00)[_-](\d+)',str(name).lower())
    if not match:return None
    kind,value=match.groups();value=int(value)
    if kind=='d100':kind='d00'
    if kind=='d10' and value==10:value=0
    return (kind,value) if value in VALUES[kind] else None


def main():
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--name',required=True)
    p.add_argument('--source-url',required=True);p.add_argument('--prepared',default=str(DATA/'prepared'))
    a=p.parse_args()
    if not re.fullmatch('[a-zA-Z0-9_-]+',a.name):p.error('Use a simple source name')
    root=Path(a.directory).resolve();config=yaml.safe_load((root/'data.yaml').read_text(encoding='utf-8'))
    names=config['names'];names=dict(enumerate(names)) if isinstance(names,list) else {int(k):v for k,v in names.items()}
    destination=DATA/'sources'/a.name
    if destination.exists():p.error('Output exists; choose a new name')
    destination.mkdir(parents=True);(destination/'images').mkdir()
    # External validation/test folders are never silently folded into training.
    images=root/'train'/'images';labels=root/'train'/'labels'
    if not images.exists():images=root/'images'/'train';labels=root/'labels'/'train'
    if not images.exists():raise ValueError('Expected train/images or images/train')
    rows=[];skipped=Counter();seen=set()
    for file in sorted(images.iterdir()):
        if file.suffix.lower() not in ('.jpg','.jpeg','.png'):continue
        label=labels/(file.stem+'.txt')
        if not label.exists():skipped['missing_annotation']+=1;continue
        digest=hashlib.sha256(file.read_bytes()).hexdigest()
        if digest in seen:skipped['duplicate_image']+=1;continue
        seen.add(digest)
        with Image.open(file) as original:image=ImageOps.exif_transpose(original).convert('RGB')
        for index,line in enumerate(label.read_text().splitlines()):
            parts=line.split()
            if len(parts)!=5:raise ValueError(f'Expected detection boxes: {label}')
            cls=int(parts[0]);parsed=parse_class(names[cls])
            if parsed is None:skipped[str(names[cls])]+=1;continue
            kind,value=parsed;cx,cy,w,h=map(float,parts[1:])
            box=dict(x=cx-w/2,y=cy-h/2,w=w,h=h)
            validate_die(dict(type=kind,value=value,box=box,value_trusted=True))
            crop=square_crop(image,box);path=destination/'images'/f'{digest}-{index}.jpg';crop.save(path,quality=95)
            rows.append(dict(image=str(path.resolve()),type=kind,value=value,split='train',holdout=False,
                value_trusted=True,source=a.source_url,source_frame_sha256=digest,
                label_origin='external annotation; not locally human reviewed',
                sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    if not rows:raise ValueError('No supported face labels found; nothing can be trained')
    (destination/'train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    report=Path(a.prepared)/'report.json'
    audit=dict(source=a.source_url,selected=len(rows),excluded=dict(skipped),
        prepared_report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
        by_kind=dict(Counter(r['type'] for r in rows)),note='External annotations require quality review; source train split only')
    (destination/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
