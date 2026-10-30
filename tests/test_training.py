import copy
import io
import json

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from training import common, prepare, annotate
from training.evaluate import iou


def die(trusted=True):
    return dict(type='d20',value=7,box=dict(x=.1,y=.1,w=.5,h=.5),value_trusted=trusted)


def test_label_rules():
    common.validate_die(die())
    for change in ({'value':21},{'box':dict(x=.8,y=0,w=.3,h=.2)},
                   {'box':dict(x=float('nan'),y=0,w=.2,h=.2)},
                   {'unreadable':True},{'value':None},{'value':True}):
        with pytest.raises(ValueError):common.validate_die(dict(die(),**change))
    d=dict(die(),type='d10',value=0);common.validate_die(d)
    with pytest.raises(ValueError):common.validate_die(dict(d,type='d00',value=9))


def test_group_partition_and_export(tmp_path,monkeypatch):
    monkeypatch.setattr(common,'DATA',tmp_path)
    monkeypatch.setattr(prepare,'DATA',tmp_path)
    records=[]
    for i in range(20):
        p=common.safe_image(f'{i}.png');p.parent.mkdir(parents=True,exist_ok=True)
        Image.new('RGB',(50+i,60),(i*11,20,70)).save(p)
        records.append(dict(id=f'r{i}',image=f'{i}.png',group=f'g{i//2}',box_reviewed=True,
                            dice=[die(i%2==0)],source='test'))
    common.write_records(records)
    out=tmp_path/'export';report=prepare.prepare(out)
    assert sum(report['frames'].values())==20
    assert sum(sum(c.values()) for c in report['faces'].values())==10
    assert len(set.union(*(set(v) for v in report['groups'].values())))==10
    groups={}
    for r in records:
        split=report['split_by_record'][r['id']]
        assert groups.setdefault(r['group'],split)==split
    for split in ('train','val','test'):assert report['frames'][split]>0
    with pytest.raises(ValueError):prepare.prepare(out)


def test_duplicate_cross_group_excluded(tmp_path,monkeypatch):
    monkeypatch.setattr(common,'DATA',tmp_path);monkeypatch.setattr(prepare,'DATA',tmp_path)
    p=common.safe_image('a.jpg');p.parent.mkdir(parents=True);Image.new('RGB',(30,40)).save(p)
    rows=[dict(id=str(i),image='a.jpg',group=f'g{i}',box_reviewed=True,dice=[die()]) for i in range(10)]
    common.write_records(rows);report=prepare.prepare(tmp_path/'export')
    assert len(report['excluded'])==10


def test_annotator_upload_save_and_origin(tmp_path,monkeypatch):
    monkeypatch.setattr(common,'DATA',tmp_path)
    monkeypatch.setattr(annotate,'DATA',tmp_path)
    client=TestClient(annotate.app)
    assert client.post('/api/upload?group=one',content=b'x',headers={'origin':'https://evil.example'}).status_code==403
    out=io.BytesIO();Image.new('RGB',(30,40),'red').save(out,'PNG')
    r=client.post('/api/upload?group=phone-day-set',content=out.getvalue());assert r.status_code==200
    rid=r.json()['id']
    assert client.get('/api/image/'+rid).status_code==200
    assert client.post('/api/records/'+rid,json={'dice':[die()]}).status_code==200
    row=client.get('/api/records').json()[0]
    assert row['human_reviewed'] and row['dice'][0]['value_trusted']
    assert client.post('/api/records/'+rid,json={'dice':[dict(die(),value=21)]}).status_code==400
    assert client.post('/api/records/'+rid,json={'dice':[]}).status_code==200


def test_crop_and_iou():
    im=Image.new('RGB',(300,100));crop=common.square_crop(im,dict(x=.8,y=0,w=.2,h=1))
    assert crop.width==crop.height
    b=die()['box'];assert iou(b,b)==1
    with pytest.raises(ValueError):common.safe_image('../../secret')


def test_prediction_is_always_provisional(tmp_path):
    from types import SimpleNamespace
    from training.predict import Pipeline
    class Number:
        def __init__(self,n):self.n=n
        def item(self):return self.n
    class Coordinates:
        def cpu(self):return self
        def tolist(self):return [10,10,80,80]
    box=SimpleNamespace(xyxy=[Coordinates()],cls=Number(3),conf=Number(.95))
    result=SimpleNamespace(boxes=[box],names={3:'d10'})
    p=Pipeline.__new__(Pipeline)
    p.detector=SimpleNamespace(predict=lambda *args,**kwargs:[result])
    p.reader=SimpleNamespace(read=lambda *args:dict(type='d10',value=0,confidence=.99,trained=True))
    p.device='cpu';p.threshold=.8;p.imgsz=640
    image=tmp_path/'test.png';Image.new('RGB',(100,100)).save(image)
    r=p.predict(image)
    assert r['proposed_total']==10 and r['confidence_gate_passed']
    assert r['total'] is None and r['needs_review']
    p.reader.read=lambda *args:dict(type='d10',value=0,confidence=.4,trained=True)
    assert p.predict(image)['proposed_total'] is None
