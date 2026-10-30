import json
from pathlib import Path
import zipfile

import pytest
import torch

from training.supplement import select, canonical
from training.acquire_kaggle import extract
from training.value_reader import NUMERALS, legal_mask, masked_logits, metrics
from training.common import KINDS


def test_supplement_blocks_source_and_project_holdouts():
    original=[dict(id='dc-c123-0',dice=[{'value_trusted':True}],group='cam/2026-07-10'),
              dict(id='dc-s456-0',dice=[],group='cam/2026-07-11')]
    report=dict(groups={'val':['cam/2026-07-11'],'test':['cam/2026-07-12']},
                split_by_record={'dc-c123-0':'train','dc-s456-0':'val'})
    def rig(name,date='2026-07-10',**kw):
        return dict(file_name=name,date=date,type='d20',value=6,holdout=False,source='rig',**kw)
    rows=[rig('rig_123-0_0.jpg'),rig('rig_999-0_0.jpg','2026-07-12'),
          rig('rig_456-0_0.jpg'),rig('rig_123-0_1.jpg'),
          dict(rig('rf.jpg'),source='roboflow:200_dataset'),
          dict(rig('eval.jpg'),holdout=True)]
    selected,excluded=select(rows,original,report)
    assert [r['file_name'] for r in selected]==['rig_123-0_1.jpg','rf.jpg']
    assert all(r['split']=='train' and r['value_trusted'] for r in selected)
    assert excluded==dict(already_trusted_training_crop=1,heldout_capture_date=1,heldout_frame=1,source_holdout=1)
    assert canonical('dc-s123-0')=='123-0'


def test_shared_values_and_type_constraints():
    assert len(NUMERALS)==28
    mask=legal_mask('cpu')
    scores=torch.zeros((2,len(NUMERALS)))
    scores[:,NUMERALS.index(90)]=100
    scores[0,NUMERALS.index(4)]=5;scores[1,NUMERALS.index(20)]=5
    kinds=torch.tensor([KINDS.index('d4'),KINDS.index('d20')])
    legal=masked_logits(scores,kinds,mask)
    targets=torch.tensor([NUMERALS.index(4),NUMERALS.index(20)])
    assert torch.equal(legal.argmax(1),targets)
    assert metrics(legal,targets,kinds)['top1']==1


def test_archive_extraction_rejects_escape(tmp_path):
    archive=tmp_path/'bad.zip'
    with zipfile.ZipFile(archive,'w') as z:z.writestr('../escape.txt','bad')
    with pytest.raises(ValueError):extract(archive,tmp_path/'output')
    assert not (tmp_path/'escape.txt').exists()
    archive=tmp_path/'good.zip'
    with zipfile.ZipFile(archive,'w') as z:z.writestr('dataset/info.txt','safe')
    extract(archive,tmp_path/'output')
    assert (tmp_path/'output/dataset/info.txt').read_text()=='safe'
