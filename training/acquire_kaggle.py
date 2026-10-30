"""Archive a public Kaggle dataset with version, source card and safe extraction."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

from download_model import fetch
from training.common import DATA


def download_ranges(url,partial,workers=8):
    """Resume a large public archive with verified parallel HTTP byte ranges."""
    import httpx
    from download_model import download_proxy
    partial=Path(partial);parts=partial.with_suffix('.ranges');parts.mkdir(exist_ok=True)
    with httpx.Client(proxy=download_proxy() or None,trust_env=False,follow_redirects=True,
                      timeout=httpx.Timeout(90,connect=30)) as client:
        with client.stream('GET',url,headers={'Range':'bytes=0-0','Accept-Encoding':'identity'}) as response:
            if response.status_code!=206:raise ValueError('Server does not support verified range download')
            total=int(response.headers['content-range'].rsplit('/',1)[1])
        prefix=partial.stat().st_size if partial.exists() else 0
        if prefix>total:raise ValueError('Partial archive larger than source')
        chunk=16*1024*1024
        ranges=[(start,min(total-1,start+chunk-1)) for start in range(prefix,total,chunk)]
        def one(bounds):
            start,end=bounds;path=parts/f'{start}-{end}.part';expected=end-start+1
            have=path.stat().st_size if path.exists() else 0
            if have>expected:raise ValueError('Invalid saved range')
            for attempt in range(4):
                if have==expected:return path
                try:
                    with client.stream('GET',url,headers={'Range':f'bytes={start+have}-{end}','Accept-Encoding':'identity'}) as r:
                        if r.status_code!=206 or r.headers.get('content-range')!=f'bytes {start+have}-{end}/{total}':
                            raise ValueError('Source byte range mismatch')
                        with path.open('ab') as output:
                            received=have
                            for block in r.iter_raw(256*1024):
                                if received+len(block)>expected:raise ValueError('Source exceeded requested byte range')
                                output.write(block);received+=len(block)
                                if received==expected:break
                    have=path.stat().st_size
                except (httpx.HTTPError,OSError):
                    have=path.stat().st_size if path.exists() else 0
                    if attempt==3:raise
            if have!=expected:raise ValueError('Incomplete range')
            return path
        print(f'Archive {total/1e6:.1f} MB; resuming {prefix/1e6:.1f} MB with {workers} connections',flush=True)
        completed={}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            jobs={pool.submit(one,b):b for b in ranges}
            for i,job in enumerate(as_completed(jobs),1):
                b=jobs[job];completed[b]=job.result()
                print(f'Archive ranges: {i}/{len(ranges)}',flush=True)
        merged=partial.with_suffix('.assembled')
        with merged.open('wb') as output:
            if prefix:
                with partial.open('rb') as source:shutil.copyfileobj(source,output)
            for bounds in ranges:
                with completed[bounds].open('rb') as source:shutil.copyfileobj(source,output)
        if merged.stat().st_size!=total:raise ValueError('Archive length mismatch')
        merged.replace(partial)

SOURCES={
    'polyhedral-types':'ucffool/dice-d4-d6-d8-d10-d12-d20-images',
    'd6-synthetic':'ronanpickell/dice-roll-recognition',
    'd6-photos':'koryakinp/d6-dices-images',
}


def extract(archive, root):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        if sum(f.file_size for f in z.infolist())>8*1024**3:raise ValueError('Archive exceeds extraction limit')
        for member in z.infolist():
            name=PurePosixPath(member.filename)
            if name.is_absolute() or '..' in name.parts or '\\' in member.filename:
                raise ValueError('Unsafe archive member')
            path=(root/str(name)).resolve()
            if not path.is_relative_to(root):raise ValueError('Archive path escaped output')
            if member.is_dir():path.mkdir(parents=True,exist_ok=True);continue
            path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists() and path.stat().st_size==member.file_size:continue
            temporary=path.with_suffix(path.suffix+'.partial')
            with z.open(member) as source,temporary.open('wb') as output:shutil.copyfileobj(source,output)
            temporary.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('source',choices=SOURCES);a=p.parse_args()
    ref=SOURCES[a.source];root=DATA/'sources'/a.source;root.mkdir(parents=True,exist_ok=True)
    card=root/'kaggle-metadata.json'
    if not card.exists():fetch('https://www.kaggle.com/api/v1/datasets/view/'+ref,card)
    meta=json.loads(card.read_text(encoding='utf-8'));version=meta['currentVersionNumber']
    archive=root/f'version-{version}.zip'
    if not archive.exists():
        part=archive.with_suffix('.download')
        print(f'Downloading {ref}, version {version}',flush=True)
        download_ranges('https://www.kaggle.com/api/v1/datasets/download/'+ref+f'?datasetVersionNumber={version}',part)
        if not zipfile.is_zipfile(part):raise ValueError('Dataset download is not a ZIP archive; no files imported')
        part.replace(archive)
    print('Extracting without executing any dataset files',flush=True)
    extract(archive,root/'files')
    with archive.open('rb') as stream:checksum=hashlib.file_digest(stream,'sha256').hexdigest()
    files=list((root/'files').rglob('*'))
    images=[f for f in files if f.suffix.lower() in ('.png','.jpg','.jpeg')]
    labels=[f for f in files if f.is_file() and f.suffix.lower() not in ('.png','.jpg','.jpeg')]
    audit=dict(ref=ref,url='https://www.kaggle.com/datasets/'+ref,version=version,
        license=meta.get('licenseName'),archive_sha256=checksum,archive_bytes=archive.stat().st_size,
        images=len(images),label_files=[str(f.relative_to(root/'files')) for f in labels],
        training_imported=False,note='Stored separately; labels and split provenance must be checked before training')
    (root/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':main()
