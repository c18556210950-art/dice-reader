"""Shared numeral learning with spatial features, frozen BN and rotation ensembles."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from training.common import DATA, CLASSES, KINDS, VALUES, read_records
from training.classifier import build as old_build

NUMERALS=sorted({v for values in VALUES.values() for v in values})


def pad(image):
    side=max(image.size)
    return ImageOps.pad(image,(side,side),color=(114,114,114))


def transforms(train=False):
    ops=[T.Lambda(pad)]
    if train:
        ops += [T.RandomRotation(180,interpolation=InterpolationMode.BILINEAR,expand=True,fill=(114,114,114))]
    ops += [T.Resize((256,256),antialias=True)]
    if train:
        ops += [T.ColorJitter(brightness=.25,contrast=.25,saturation=.15),T.RandomGrayscale(.25),
                T.RandomApply([T.GaussianBlur(3,sigma=(.1,1.0))],p=.1)]
    return T.Compose(ops+[T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])


def views(image, transform):
    return torch.stack([transform(image.rotate(angle,expand=True)) for angle in (0,90,180,270)])


class ValueFaces(Dataset):
    def __init__(self, prepared, split, supplement=None):
        self.train=split=='train';self.transform=transforms(self.train)
        self.entries=[]
        for path in sorted((Path(prepared)/'classifier'/split).glob('*/*.jpg')):
            kind,value=path.parent.name.split('_')
            self.entries.append((path,kind,int(value),'original'))
        if supplement and self.train:
            for r in read_records(supplement):
                if r.get('split')!='train' or r.get('holdout') is not False or r.get('value_trusted') is not True:
                    raise ValueError('Supplement must contain only trusted training crops')
                self.entries.append((Path(r['image']),r['type'],r['value'],r['source']))

    def __len__(self):return len(self.entries)

    def __getitem__(self,i):
        path,kind,value,_=self.entries[i]
        with Image.open(path) as im:
            image=im.convert('RGB')
            x=self.transform(image) if self.train else views(image,self.transform)
        return x,KINDS.index(kind),NUMERALS.index(value)


def build(pretrained=False):
    model=old_build(pretrained)
    # Preserve where glyphs appear instead of globally averaging all visible faces.
    model.avgpool=nn.AdaptiveAvgPool2d((3,3))
    model.fc=nn.Sequential(nn.Dropout(.35),nn.Linear(512*3*3,len(NUMERALS)))
    return model


def legal_mask(device):
    return torch.tensor([[v in VALUES[k] for v in NUMERALS] for k in KINDS],device=device)


def masked_logits(scores,kinds,mask):
    return scores.masked_fill(~mask[kinds],-1e4)


@torch.inference_mode()
def collect(model,loader,device,mask):
    model.eval();all_scores=[];all_targets=[];all_kinds=[]
    for x,k,y in loader:
        b,n,c,h,w=x.shape
        scores=model(x.reshape(b*n,c,h,w).to(device)).reshape(b,n,-1).mean(1)
        k=k.to(device);scores=masked_logits(scores,k,mask)
        all_scores.append(scores.cpu());all_targets.append(y);all_kinds.append(k.cpu())
    return torch.cat(all_scores),torch.cat(all_targets),torch.cat(all_kinds)


def metrics(scores,targets,kinds,temperature=1.):
    predictions=scores.argmax(1);correct=predictions==targets;per={};confusion=Counter()
    for kind,y,p in zip(kinds.tolist(),targets.tolist(),predictions.tolist()):
        label=f'{KINDS[kind]}_{NUMERALS[y]}'
        stat=per.setdefault(label,dict(n=0,correct=0));stat['n']+=1;stat['correct']+=int(y==p)
        confusion[f'{label} -> {KINDS[kind]}_{NUMERALS[p]}']+=1
    return dict(n=len(targets),top1=correct.float().mean().item(),
                nll=nn.functional.cross_entropy(scores/temperature,targets).item(),
                per_class=per,confusion=dict(confusion))


def train(prepared,run,supplement=None,epochs=40,batch=16,device='cuda:0'):
    torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42);random.seed(42)
    prepared=Path(prepared);run=Path(run);run.mkdir(parents=True,exist_ok=False)
    if supplement:
        audit=json.loads(Path(supplement).with_name('audit.json').read_text())
        actual=hashlib.sha256((prepared/'report.json').read_bytes()).hexdigest()
        if audit['prepared_report_sha256']!=actual:raise ValueError('Supplement split audit does not match prepared data')
        (run/'supplement.snapshot.jsonl').write_bytes(Path(supplement).read_bytes())
    train_data=ValueFaces(prepared,'train',supplement);val=ValueFaces(prepared,'val')
    counts=Counter(f'{k}_{v}' for _,k,v,_ in train_data.entries)
    weights=[1/math.sqrt(counts[f'{k}_{v}']) for _,k,v,_ in train_data.entries]
    sampler=WeightedRandomSampler(weights,len(train_data),replacement=True)
    dl=DataLoader(train_data,batch_size=batch,sampler=sampler,num_workers=0)
    vl=DataLoader(val,batch_size=batch,num_workers=0)
    model=build(True).to(device);mask=legal_mask(device)
    for parameter in model.parameters():parameter.requires_grad=False
    for parameter in model.fc.parameters():parameter.requires_grad=True
    optimizer=torch.optim.AdamW([{'params':model.fc.parameters(),'lr':.001},
                                 {'params':model.layer4.parameters(),'lr':.00003}],weight_decay=.02)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs,eta_min=1e-6)
    history=[];best=-1.
    for epoch in range(1,epochs+1):
        if epoch==6:
            for parameter in model.layer4.parameters():parameter.requires_grad=True
        model.train()
        for module in model.modules():
            if isinstance(module,nn.BatchNorm2d):module.eval()
        total=0.;n=0
        for x,k,y in dl:
            x,k,y=x.to(device),k.to(device),y.to(device)
            optimizer.zero_grad(set_to_none=True)
            scores=masked_logits(model(x),k,mask)
            loss=nn.functional.cross_entropy(scores,y,label_smoothing=0.)
            loss.backward();optimizer.step();total+=loss.item()*len(y);n+=len(y)
        scheduler.step()
        scores,targets,kinds=collect(model,vl,device,mask)
        metric=metrics(scores,targets,kinds)
        history.append(dict(epoch=epoch,train_loss=total/n,val=metric))
        print(json.dumps(dict(epoch=epoch,train_n=n,train_loss=total/n,val_top1=metric['top1'])),flush=True)
        state=dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
            architecture='spatial-value-v2',classes=CLASSES,numerals=NUMERALS,
            train_counts={c:counts[c] for c in CLASSES},epoch=epoch,smoke=False,
            temperature=1.,rotation_tta=True,transform='padded-256-imagenet-v2')
        if metric['top1']>best:
            best=metric['top1'];torch.save(state,run/'best.pt')
        torch.save(state,run/'last.pt')
        (run/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
    state=torch.load(run/'best.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(state['state_dict']);model.to(device)
    scores,targets,kinds=collect(model,vl,device,mask)
    grid=[.5,.75,1.,1.5,2.,3.,4.]
    temperature=min(grid,key=lambda t:nn.functional.cross_entropy(scores/t,targets).item())
    state['temperature']=temperature;torch.save(state,run/'best.pt')
    (run/'validation.json').write_text(json.dumps(metrics(scores,targets,kinds,temperature),indent=2),encoding='utf-8')
    config=dict(architecture=state['architecture'],epochs=epochs,batch=batch,
        pretrained=True,original_train_n=len(ValueFaces(prepared,'train')),train_n=len(train_data),
        prepared=str(prepared.resolve()),supplement=str(supplement) if supplement else None,
        train_counts=dict(counts),temperature=temperature,
        selection='validation top1 only; calibration uses validation NLL; no test loading during training')
    (run/'run.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print(f'Complete: {run}; best validation={best:.4f}',flush=True)


class SpatialFaceReader:
    def __init__(self, state, device='cpu'):
        self.checkpoint=state;self.device=device
        self.model=build();self.model.load_state_dict(state['state_dict']);self.model.to(device).eval()
        self.transform=transforms(False);self.mask=legal_mask(device)

    @torch.inference_mode()
    def read(self,crop,kind):
        if not any(self.checkpoint['train_counts'].get(f'{kind}_{v}',0) for v in VALUES[kind]):
            return dict(type=kind,value=VALUES[kind][0],confidence=0.,trained=False)
        x=views(crop,self.transform).to(self.device)
        scores=self.model(x).mean(0,keepdim=True)
        k=torch.tensor([KINDS.index(kind)],device=self.device)
        probs=(masked_logits(scores,k,self.mask)/self.checkpoint.get('temperature',1.)).softmax(1)[0]
        i=probs.argmax().item();value=NUMERALS[i]
        return dict(type=kind,value=value,confidence=probs[i].item(),
                    trained=self.checkpoint['train_counts'].get(f'{kind}_{value}',0)>0)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepared',default=str(DATA/'prepared'))
    p.add_argument('--run',required=True);p.add_argument('--supplement');p.add_argument('--epochs',type=int,default=40)
    p.add_argument('--batch',type=int,default=16);p.add_argument('--device',default='cuda:0')
    a=p.parse_args();train(a.prepared,a.run,a.supplement,a.epochs,a.batch,a.device)
