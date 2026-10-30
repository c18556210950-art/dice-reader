"""Local detector -> square crop -> type-constrained face classifier; abstain on uncertainty."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageOps
from training.common import square_crop


class Pipeline:
    def __init__(self, detector, classifier, device='cpu',threshold=.8,allow_smoke=False):
        from training.runtime import local_yolo
        YOLO=local_yolo()
        from training.classifier import FaceReader
        self.detector=YOLO(str(detector));self.reader=FaceReader(classifier,device)
        self.device=device;self.threshold=threshold
        self.imgsz=int(self.detector.overrides.get('imgsz',960))
        if self.reader.checkpoint.get('smoke') and not allow_smoke:
            raise ValueError('Smoke checkpoint is for wiring tests, not real dice recognition.')

    def predict(self,path):
        with Image.open(path) as im: image=ImageOps.exif_transpose(im).convert('RGB')
        result=self.detector.predict(image,device=self.device,conf=.25,verbose=False,imgsz=self.imgsz)[0]
        dice=[]
        for box in result.boxes:
            x1,y1,x2,y2=box.xyxy[0].cpu().tolist()
            b=dict(x=x1/image.width,y=y1/image.height,w=(x2-x1)/image.width,h=(y2-y1)/image.height)
            kind=result.names[int(box.cls.item())]
            prediction=self.reader.read(square_crop(image,b),kind)
            confidence=min(prediction['confidence'],box.conf.item())
            accepted=prediction['trained'] and confidence>=self.threshold
            dice.append(dict(**prediction,box=b,detector_confidence=box.conf.item(),accepted=accepted))
        complete=bool(dice) and all(d['accepted'] for d in dice)
        total=sum(10 if d['type']=='d10' and d['value']==0 else d['value'] for d in dice) if complete else None
        return dict(dice=dice,proposed_total=total,total=None,needs_review=True,
                    confidence_gate_passed=complete,
                    note='当前为训练原型，所有读数和候选总和均待人工核对；高置信度也可能错误或漏检。D10的0计作10；百分骰独立计数，不自动与D10配对。')


def main():
    p=argparse.ArgumentParser();p.add_argument('image');p.add_argument('--detector',required=True)
    p.add_argument('--classifier',required=True);p.add_argument('--device',default='cpu')
    p.add_argument('--threshold',type=float,default=.8);p.add_argument('--allow-smoke',action='store_true')
    p.add_argument('--output')
    a=p.parse_args()
    if not 0<a.threshold<=1:p.error('threshold must be in (0,1]')
    r=Pipeline(a.detector,a.classifier,a.device,a.threshold,a.allow_smoke).predict(a.image)
    output=json.dumps(r,ensure_ascii=False,indent=2);print(output)
    if a.output:Path(a.output).write_text(output,encoding='utf-8')


if __name__=='__main__':main()
