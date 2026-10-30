"""Additional face crops, train only; block original holdout frames AND capture dates."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import quote

from PIL import Image
from download_model import fetch
from training.common import DATA, VALUES, read_records

REPO='G-G-Games/diecamera-crops'


def canonical(frame):
    return re.sub(r'^[a-zA-Z]+','',frame.removeprefix('dc-'))


def select(rows, originals, report):
    frames={canonical(r['id']):r for r in originals}
    forbidden_dates={g.rsplit('/',1)[1] for split in ('val','test') for g in report['groups'][split]}
    selected=[];reasons=Counter()
    for row in rows:
        r=dict(row)
        if r.get('holdout') is not False:reasons['source_holdout']+=1;continue
        if r.get('type') not in VALUES or type(r.get('value')) is not int:
            reasons['unsupported_label']+=1;continue
        if r['type']=='d10' and r['value']==10:r['value']=0
        if r['value'] not in VALUES[r['type']]:reasons['invalid_value']+=1;continue
        if r['source']=='rig':
            if r['date'] in forbidden_dates:
                reasons['heldout_capture_date']+=1;continue
            name=Path(r['file_name']).stem
            try:frame,index=name.removeprefix('rig_').rsplit('_',1);index=int(index)
            except ValueError:reasons['untraceable_rig']+=1;continue
            original=frames.get(canonical(frame))
            if original:
                if report['split_by_record'][original['id']]!='train':
                    reasons['heldout_frame']+=1;continue
                if index<len(original['dice']) and original['dice'][index].get('value_trusted'):
                    reasons['already_trusted_training_crop']+=1;continue
            r['license']='CC-BY-NC-4.0'
        elif r['source'] in ('roboflow:200_dataset','roboflow:d4-turbo-rad-v4'):
            r['license']='Image CC-BY-4.0; labels CC-BY-NC-4.0; see source README'
        else:reasons['unknown_provenance']+=1;continue
        r['split']='train';r['value_trusted']=True
        selected.append(r)
    return selected,dict(reasons)


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepared',default=str(DATA/'prepared'));p.add_argument('--workers',type=int,default=8)
    a=p.parse_args();prepared=Path(a.prepared)
    directory=DATA/'sources'/'face-supplement';directory.mkdir(parents=True,exist_ok=True)
    lock=directory/'revision.json'
    if not lock.exists():
        fetch(f'https://huggingface.co/api/datasets/{REPO}',directory/'repo.json')
        revision=json.loads((directory/'repo.json').read_text(encoding='utf-8'))['sha']
        lock.write_text(json.dumps(dict(repo=REPO,revision=revision)),encoding='utf-8')
    revision=json.loads(lock.read_text())['revision']
    base=f'https://huggingface.co/datasets/{REPO}/resolve/{revision}/'
    for name,path in [('README.md',directory/'README.md'),('data/metadata.jsonl',directory/'metadata.jsonl')]:
        if not path.exists():fetch(base+name,path)
    selected,reasons=select(read_records(directory/'metadata.jsonl'),
        read_records(prepared/'records.snapshot.jsonl'),json.loads((prepared/'report.json').read_text()))
    image_dir=directory/'images';image_dir.mkdir(exist_ok=True)
    def download(r):
        name=r['file_name']
        if Path(name).name!=name or '/' in name or '\\' in name:raise ValueError('Unsafe filename')
        target=image_dir/name
        if not target.exists():
            temp=target.with_suffix('.download');fetch(base+'data/'+quote(name),temp)
            with Image.open(temp) as im:im.verify()
            temp.replace(target)
        return dict(r,image=str(target.resolve()),revision=revision,repo=REPO,
                    sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    done=[]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        jobs=[pool.submit(download,r) for r in selected]
        for n,job in enumerate(as_completed(jobs),1):
            done.append(job.result())
            if n%50==0:print(f'{n}/{len(selected)} supplementary crops',flush=True)
    done.sort(key=lambda r:r['file_name'])
    manifest=directory/'train.jsonl'
    manifest.write_text(''.join(json.dumps(r)+'\n' for r in done),encoding='utf-8')
    audit=dict(revision=revision,prepared=str(prepared.resolve()),
        prepared_report_sha256=hashlib.sha256((prepared/'report.json').read_bytes()).hexdigest(),
        selected=len(done),excluded=reasons,by_kind=dict(Counter(r['type'] for r in done)),
        source=dict(Counter(r['source'] for r in done)))
    (directory/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':main()
