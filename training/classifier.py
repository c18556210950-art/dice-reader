"""ResNet18 face classifier with whole-crop transforms and type-constrained output."""
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms as T

from training.common import CLASSES, DATA


def transform(train=False):
    # Never mirror numerals. A square crop already preserves the complete die.
    ops=[T.Resize((224,224))]
    if train:
        ops += [T.RandomRotation(180,fill=(114,114,114)),
                T.ColorJitter(brightness=.2,contrast=.2,saturation=.15)]
    return T.Compose(ops+[T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])


class Faces(Dataset):
    def __init__(self, root, train=False, limit=0):
        self.items=sorted((p,CLASSES.index(p.parent.name)) for p in Path(root).glob('*/*.jpg'))
        if limit:
            rng=random.Random(42); rng.shuffle(self.items); self.items=self.items[:limit]
        self.transform=transform(train)

    def __len__(self): return len(self.items)

    def __getitem__(self,i):
        p,y=self.items[i]
        with Image.open(p) as im: x=self.transform(im.convert('RGB'))
        return x,y


def build(pretrained=False):
    model=models.resnet18(weights=None)
    if pretrained:
        from download_model import fetch, checksum
        path=DATA/'weights'/'resnet18-f37072fd.pth'; path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists() or not checksum(path).startswith('f37072fd'):
            partial=path.with_suffix('.download')
            fetch('https://download.pytorch.org/models/resnet18-f37072fd.pth',partial)
            if not checksum(partial).startswith('f37072fd'): raise ValueError('Pretrained classifier checksum mismatch')
            partial.replace(path)
        model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
    model.fc=nn.Linear(model.fc.in_features,len(CLASSES))
    return model


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval(); correct=0; total=0; loss=0.; per={}; confusion={}
    for x,y in loader:
        x,y=x.to(device),y.to(device); scores=model(x); pred=scores.argmax(1)
        loss += nn.functional.cross_entropy(scores,y,reduction='sum').item()
        correct+=(pred==y).sum().item(); total+=len(y)
        for a,b in zip(y.cpu().tolist(),pred.cpu().tolist()):
            label=CLASSES[a]; stat=per.setdefault(label,{'n':0,'correct':0});stat['n']+=1;stat['correct']+=int(a==b)
            key=label+' -> '+CLASSES[b];confusion[key]=confusion.get(key,0)+1
    return dict(n=total,top1=correct/total if total else None,loss=loss/total if total else None,
                per_class=per,confusion=confusion)


def train_classifier(prepared, run, epochs=40, batch=32, device='cuda:0', smoke=False, pretrained=True):
    torch.manual_seed(42); random.seed(42); np.random.seed(42)
    torch.set_num_threads(4)
    run=Path(run);run.mkdir(parents=True,exist_ok=False)
    prepared=Path(prepared)
    train=Faces(prepared/'classifier'/'train',True,32 if smoke else 0)
    val=Faces(prepared/'classifier'/'val',limit=32 if smoke else 0)
    if not len(train) or not len(val): raise ValueError('Train and validation both need trusted faces; collect more sessions or change split seed.')
    dl=DataLoader(train,batch_size=batch,shuffle=True,num_workers=0)
    vl=DataLoader(val,batch_size=batch,num_workers=0)
    model=build(pretrained).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    counts={c:0 for c in CLASSES}
    for _,y in train.items: counts[CLASSES[y]]+=1
    best=-1.;history=[]
    for epoch in range(1,epochs+1):
        model.train(); total=0.;n=0
        for x,y in dl:
            x,y=x.to(device),y.to(device);optimizer.zero_grad(set_to_none=True)
            loss=nn.functional.cross_entropy(model(x),y);loss.backward();optimizer.step()
            total+=loss.item()*len(y);n+=len(y)
        metrics=evaluate(model,vl,device)
        row=dict(epoch=epoch,train_loss=total/n,val=metrics);history.append(row)
        print(json.dumps(dict(epoch=epoch,train_loss=total/n,val_n=metrics['n'],val_top1=metrics['top1'])),flush=True)
        state=dict(state_dict=model.cpu().state_dict(),classes=CLASSES,train_counts=counts,
                   epoch=epoch,smoke=smoke,pretrained=pretrained,transform='whole-square-imagenet-v1')
        torch.save(state,run/'last.pt')
        if metrics['top1']>best:
            best=metrics['top1'];torch.save(state,run/'best.pt')
        model.to(device)
        (run/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
    checkpoint=torch.load(run/'best.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(checkpoint['state_dict']);model.to(device)
    test=Faces(prepared/'classifier'/'test')
    metrics=evaluate(model,DataLoader(test,batch_size=batch,num_workers=0),device)
    (run/'test.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    (run/'run.json').write_text(json.dumps(dict(smoke=smoke,prepared=str(prepared.resolve()),
        epochs=epochs,pretrained=pretrained,absent_classes=[c for c,n in counts.items() if not n]),indent=2),encoding='utf-8')
    return run/'best.pt'


class FaceReader:
    def __init__(self, checkpoint, device='cpu'):
        self.device=device
        self.checkpoint=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if self.checkpoint['classes']!=CLASSES: raise ValueError('Class mapping mismatch')
        self.delegate=None
        if self.checkpoint.get('architecture')=='spatial-value-v2':
            from training.value_reader import SpatialFaceReader
            self.delegate=SpatialFaceReader(self.checkpoint,device)
            return
        self.model=build();self.model.load_state_dict(self.checkpoint['state_dict']);self.model.to(device).eval()
        self.transform=transform()

    @torch.inference_mode()
    def read(self, crop, kind):
        if self.delegate is not None:return self.delegate.read(crop,kind)
        scores=self.model(self.transform(crop).unsqueeze(0).to(self.device))[0]
        probs=scores.softmax(0)
        allowed=[i for i,c in enumerate(CLASSES) if c.startswith(kind+'_')]
        i=max(allowed,key=lambda i:probs[i].item())
        # Preserve global probability: a type mismatch must lower confidence.
        return dict(type=kind,value=int(CLASSES[i].split('_')[1]),confidence=probs[i].item(),
                    trained=self.checkpoint['train_counts'].get(CLASSES[i],0)>0)
