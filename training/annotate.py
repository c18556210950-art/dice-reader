"""Loopback annotation tool. Independent of the deployed dice webpage."""
import hashlib
import io
import json
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from PIL import Image, ImageOps, UnidentifiedImageError

from training.common import DATA, KINDS, read_records, write_records, safe_image, validate_die, corpus_lock

app=FastAPI(docs_url=None,redoc_url=None)
LOCK=threading.RLock()


@app.middleware('http')
async def loopback(request, call_next):
    from fastapi.responses import JSONResponse
    if request.headers.get('host') not in {'127.0.0.1:8767','localhost:8767','testserver'}:
        return JSONResponse({'detail':'Local access only'},status_code=403)
    origin=request.headers.get('origin')
    if origin and origin not in {'http://127.0.0.1:8767','http://localhost:8767'}:
        return JSONResponse({'detail':'Origin rejected'},status_code=403)
    return await call_next(request)


@app.get('/')
def index():
    return FileResponse(Path(__file__).with_name('annotate.html'))


@app.get('/api/health')
def health():
    import os
    return dict(app='dice-local-annotator',pid=os.getpid())


@app.get('/api/records')
def records():
    with LOCK:
        return read_records()


@app.get('/api/image/{record_id}')
def image(record_id:str):
    with LOCK:
        r=next((r for r in read_records() if r['id']==record_id),None)
    if not r: raise HTTPException(404,'Unknown image')
    return FileResponse(safe_image(r['image']))


@app.post('/api/upload')
async def upload(request:Request, group:str):
    group=group.strip()
    if not group or len(group)>160: raise HTTPException(400,'请填写拍摄批次（设备/日期/骰子套装）')
    body=bytearray()
    async for block in request.stream():
        body.extend(block)
        if len(body)>20*1024*1024: raise HTTPException(413,'单张图片不能超过20MB')
    try:
        with Image.open(io.BytesIO(body)) as im:
            if im.width*im.height>32_000_000: raise ValueError('图片超过3200万像素')
            im=ImageOps.exif_transpose(im).convert('RGB')
            im.thumbnail((2400,2400))
            stream=io.BytesIO(); im.save(stream,'JPEG',quality=95)
    except (UnidentifiedImageError,ValueError,OSError,Image.DecompressionBombError) as exc:
        raise HTTPException(400,'无法读取图片') from exc
    data=stream.getvalue(); digest=hashlib.sha256(data).hexdigest()
    row=dict(id='own-'+digest[:24], image='personal/'+digest+'.jpg',group='personal/'+group,
             source='personal',license='private',box_reviewed=False,human_reviewed=False,dice=[],sha256=digest)
    with LOCK, corpus_lock():
        rows=read_records()
        if not any(r['id']==row['id'] for r in rows):
            path=safe_image(row['image']); path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(data)
            rows.append(row); write_records(rows)
    return {'id':row['id']}


@app.post('/api/records/{record_id}')
async def save(record_id:str,request:Request):
    payload=await request.json()
    dice=payload.get('dice')
    if not isinstance(dice,list) or len(dice)>80: raise HTTPException(400,'骰子数量无效')
    clean=[]
    try:
        for d in dice:
            validate_die(d)
            if type(d.get('value_trusted',False)) is not bool or type(d.get('unreadable',False)) is not bool:
                raise ValueError('Invalid checkbox values')
            clean.append({k:d[k] for k in ('type','value','box','value_trusted','unreadable') if k in d})
    except (ValueError,KeyError,TypeError) as exc: raise HTTPException(400,str(exc)) from exc
    with LOCK, corpus_lock():
        rows=read_records(); row=next((r for r in rows if r['id']==record_id),None)
        if row is None: raise HTTPException(404,'Unknown image')
        history=DATA/'annotation-history';history.mkdir(parents=True,exist_ok=True)
        (history/(str(time.time_ns())+'.json')).write_text(json.dumps(row,ensure_ascii=False),encoding='utf-8')
        row.update(dice=clean,box_reviewed=True,human_reviewed=True,label_source='personal-human')
        write_records(rows)
    return {'saved':True}


if __name__=='__main__':
    import uvicorn
    uvicorn.run(app,host='127.0.0.1',port=8767,access_log=False)
