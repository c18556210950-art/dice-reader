import argparse
import json
from datetime import datetime
from pathlib import Path

from training.common import DATA


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--prepared',default=str(DATA/'prepared'))
    p.add_argument('--stage',choices=['detector','classifier','both'],default='both')
    p.add_argument('--epochs',type=int,default=50)
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--device',default='0')
    p.add_argument('--imgsz',type=int,default=960)
    p.add_argument('--name',default=datetime.now().strftime('%Y%m%d-%H%M%S'))
    p.add_argument('--smoke',action='store_true',help='1 epoch, reduced size/data; wiring test only')
    p.add_argument('--scratch',action='store_true',help='Explicitly disable pretrained initialization')
    p.add_argument('--reader',choices=['baseline','spatial-v2'],default='baseline')
    p.add_argument('--supplement',help='Audited training-only supplementary face manifest (spatial-v2 only)')
    a=p.parse_args(); prepared=Path(a.prepared).resolve()
    if a.epochs<1 or a.batch<1: p.error('epochs and batch must be positive')
    if a.supplement and a.reader!='spatial-v2':p.error('--supplement requires --reader spatial-v2')
    if a.reader=='spatial-v2' and a.scratch:p.error('spatial-v2 currently requires pretrained initialization')
    if a.reader=='spatial-v2' and a.smoke:p.error('Use the baseline reader for --smoke wiring tests')
    if not (prepared/'report.json').exists(): p.error('Run training.prepare first')
    report=json.loads((prepared/'report.json').read_text())
    if not report['frames'].get('train') or not report['frames'].get('val'):
        p.error('Training and validation need different capture groups')
    run=DATA/'runs'/a.name
    if run.exists(): p.error('Run name already exists; choose another name')
    run.mkdir(parents=True)
    (run/'config.json').write_text(json.dumps(vars(a),indent=2),encoding='utf-8')
    epochs=1 if a.smoke else a.epochs
    if a.stage in ('detector','both'):
        # pandas 3's optional Arrow string backend is incompatible with the bundled
        # Arrow/NumPy combination on this PC; CSV training metrics need no Arrow.
        import pandas as pd
        pd.options.future.infer_string=False
        from training.runtime import local_yolo
        YOLO=local_yolo()
        from download_model import fetch
        weights=DATA/'weights'/'yolo11n.pt'; weights.parent.mkdir(parents=True,exist_ok=True)
        if not a.scratch and not weights.exists():
            fetch('https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt',weights)
        model=YOLO('yolo11n.yaml' if a.scratch else str(weights))
        model.train(data=str(prepared/'detector'/'data.yaml'),epochs=epochs,imgsz=320 if a.smoke else a.imgsz,
            batch=a.batch,device=a.device,workers=0,project=str(run),name='detector',
            seed=42,deterministic=True,fliplr=0.,flipud=0.,degrees=180.,
            mosaic=0. if a.smoke else .3,close_mosaic=0 if a.smoke else 10,
            fraction=.15 if a.smoke else 1.,amp=False,plots=True,save=True)
    if a.stage in ('classifier','both'):
        from training.classifier import train_classifier
        device='cpu' if a.device=='cpu' else 'cuda:'+a.device
        if a.reader=='spatial-v2':
            from training.value_reader import train as train_value_reader
            train_value_reader(prepared,run/'classifier',a.supplement,epochs,a.batch,device)
        else:
            train_classifier(prepared,run/'classifier',epochs,a.batch,device,a.smoke,not a.scratch)
    print(f'Run complete: {run}',flush=True)


if __name__=='__main__': main()
