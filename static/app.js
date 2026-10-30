const $ = id => document.getElementById(id);
let dice = [], imageData = null, objectURL = null, ready = false, busy = false, pendingAuto = false;
let generation = 0;
const ranges = {d4:[1,4],d6:[1,6],d8:[1,8],d10:[1,10],d12:[1,12],d20:[1,20],d00:[0,90],unknown:[0,100]};
function valid(d){const [lo,hi]=ranges[d.kind];return Number.isInteger(d.value)&&d.value>=lo&&d.value<=hi&&(d.kind!=='d00'||d.value%10===0);}
function status(text,error=false){$('status').textContent=text;$('status').classList.toggle('error',error);}
function updateButton(){for(const id of ['camera','replace','file','camera-file','d4','add']){const el=$(id);if(el)el.disabled=busy;}}
async function checkStatus(){
  const previousReady=ready;
  try{const r=await fetch('api/status');if(!r.ok)throw Error();const s=await r.json();ready=s.ready;$('model-state').textContent=s.message;if(!busy&&(!imageData||previousReady!==ready))status(s.message);}
  catch{ready=false;status('无法连接识别服务',true);}
  updateButton();
  if(ready&&!previousReady&&pendingAuto&&imageData&&!busy){pendingAuto=false;recognize(true);}
}
function percentilePairs(){
  const {pairs,usedUnits}=pairPercentile(dice);
  const pairedTens=new Set(pairs.map(p=>p.tens));
  return {pairs,usedUnits,pairedTens};
}
function renderSummary(){
  const known=dice.filter(valid), pending=dice.filter(d=>!valid(d)||!d.confirmed).length;
  const {pairs,usedUnits,pairedTens}=percentilePairs();
  let sum=0;
  dice.forEach((d,i)=>{if(valid(d)&&!pairedTens.has(i)&&!usedUnits.has(i))sum+=d.value;});
  pairs.forEach(p=>{sum+=p.total;});
  $('count').textContent=`${dice.length} 颗骰子`;
  $('sum').textContent=known.length?sum:'—';
  $('sum-label').textContent=!dice.length?'等待投掷':pending?'暂计总和 · 待核对':'本次总和 · 已核对';
  $('sum-detail').textContent=!dice.length?'每一场冒险，都从一次投掷开始。':pending?`${known.length} 颗有读数，${pending} 颗待核对；请检查是否漏骰。`:`${dice.length} 颗骰子已核对，请确认照片中没有遗漏。`;
  const box=$('pairing');box.replaceChildren();
  pairs.forEach(p=>{
    const line=document.createElement('div');line.className='pair-line';
    const units=dice[p.units].value%10;
    line.textContent=`百分位配对：第 ${p.tens+1} 颗（${dice[p.tens].value}）+ 第 ${p.units+1} 颗（${units}）= ${p.total}`;
    box.append(line);
  });
  box.hidden=!pairs.length;
  $('empty').hidden=!!dice.length;$('confirm-all').disabled=!dice.length||known.length!==dice.length||busy;
}
function render(){
  renderSummary();
  $('rows').replaceChildren();$('markers').replaceChildren();
  const {pairs}=percentilePairs();
  const pairInfo=new Map();
  pairs.forEach(p=>{pairInfo.set(p.tens,{total:p.total,role:'tens'});pairInfo.set(p.units,{total:p.total,role:'units'});});
  dice.forEach((d,i)=>{
    const row=document.createElement('div');row.className='die-row';row.id=`die-${i}`;
    const num=document.createElement('span');num.className='number';num.textContent=String(i+1).padStart(2,'0');
    const select=document.createElement('select');select.setAttribute('aria-label',`第 ${i+1} 颗骰子类型`);
    Object.keys(ranges).forEach(kind=>{const opt=document.createElement('option');opt.value=kind;opt.textContent=kind==='unknown'?'未确定':kind==='d00'?'百分位':kind;select.append(opt);});select.value=d.kind;select.disabled=busy;
    select.onchange=()=>{d.kind=select.value;d.confirmed=false;render();};
    const input=document.createElement('input');input.type='number';input.value=d.value??'';input.placeholder='?';input.min=ranges[d.kind][0];input.max=ranges[d.kind][1];input.step=d.kind==='d00'?10:1;input.disabled=busy;input.setAttribute('aria-label',`第 ${i+1} 颗骰子读数`);
    input.oninput=()=>{d.value=input.value===''?null:Number(input.value);d.confirmed=false;box.checked=false;box.disabled=!valid(d)||busy;note.textContent=!valid(d)?'请输入符合骰子范围的整数读数。':d.note;note.hidden=!note.textContent;renderSummary();const label=$('markers').querySelector(`[data-index="${i}"] span`);if(label){label.parentElement.classList.remove('confirmed');}if(label)label.textContent=`${i+1} · ${d.value??'?'}`;};
    input.onchange=()=>render();
    const review=document.createElement('label');review.className='review';const box=document.createElement('input');box.type='checkbox';box.checked=d.confirmed&&valid(d);box.disabled=!valid(d)||busy;box.onchange=()=>{d.confirmed=box.checked;render();};review.append(box,document.createTextNode('已核对'));
    const del=document.createElement('button');del.className='delete';del.textContent='×';del.title='移除这颗骰子';del.setAttribute('aria-label',`移除第 ${i+1} 颗骰子`);del.disabled=busy;del.onclick=()=>{dice.splice(i,1);render();};
    const pair=pairInfo.get(i);
    const badge=document.createElement('span');
    if(pair){badge.className='pair-badge';badge.textContent=pair.role==='tens'?`配对 ${pair.total}`:`已并入 ${pair.total}`;badge.title='百分位骰自动配对，不重复计入总和';}
    row.append(num,select,input,badge,review,del);
    const note=document.createElement('div');note.className='die-note';note.textContent=!valid(d)?'请输入符合骰子范围的整数读数。':d.note;note.hidden=!note.textContent;row.append(note);
    $('rows').append(row);
    if(d.bbox){const [x1,y1,x2,y2]=d.bbox;const marker=document.createElement('button');marker.dataset.index=String(i);marker.className=`marker ${d.confirmed?'confirmed':''}`;marker.style.left=`${x1/10}%`;marker.style.top=`${y1/10}%`;marker.style.width=`${(x2-x1)/10}%`;marker.style.height=`${(y2-y1)/10}%`;marker.setAttribute('aria-label',`查看第 ${i+1} 颗骰子`);const label=document.createElement('span');label.textContent=`${i+1} · ${d.value??'?'}`;marker.append(label);marker.onclick=()=>{row.scrollIntoView({behavior:'smooth',block:'nearest'});row.classList.add('highlight');input.focus();setTimeout(()=>row.classList.remove('highlight'),1600);};$('markers').append(marker);}
  });
}
async function loadFile(file){
  if(busy||!file)return;
  if(file.type&&!['image/jpeg','image/png','image/webp'].includes(file.type)){status('请选择 JPG、PNG 或 WebP 图片',true);return;}
  if(file.size>15*1024*1024){status('照片不能超过 15MB',true);return;}
  const version=++generation;
  try{
    // Canvas normalizes EXIF rotation so preview and model boxes use the same pixels.
    const bitmap=await createImageBitmap(file,{imageOrientation:'from-image'});
    if(version!==generation){bitmap.close();return;}
    if(bitmap.width*bitmap.height>32000000){bitmap.close();throw Error('照片分辨率过大，请缩小到 3200 万像素以内');}
    const scale=Math.min(1,1600/Math.max(bitmap.width,bitmap.height));const canvas=document.createElement('canvas');canvas.width=Math.round(bitmap.width*scale);canvas.height=Math.round(bitmap.height*scale);canvas.getContext('2d').drawImage(bitmap,0,0,canvas.width,canvas.height);bitmap.close();
    const blob=await new Promise(resolve=>canvas.toBlob(resolve,'image/jpeg',0.95));if(version!==generation)return;
    if(!blob)throw Error('无法读取照片');
    imageData=canvas.toDataURL('image/jpeg',0.95).split(',')[1];if(objectURL)URL.revokeObjectURL(objectURL);objectURL=URL.createObjectURL(blob);$('photo').src=objectURL;
    dice=[];render();$('warning').hidden=true;$('dropzone').hidden=true;$('preview').hidden=false;$('replace').hidden=false;
    updateButton();
    if(ready){status('照片已就绪，正在识别…');recognize(true);}
    else{pendingAuto=true;status('照片已就绪；识别服务尚未就绪，恢复后会自动识别。');}
  }catch(e){if(version===generation)status(e.message||'无法读取照片',true);}
}
async function recognize(force=false){
  if(busy||!imageData||!ready)return;
  if(!force&&dice.length&&!window.confirm('重新识别会替换当前读数和手动修改，继续吗？'))return;
  pendingAuto=false;busy=true;updateButton();render();status('正在识别，请稍候…');
  try{
    const r=await fetch('api/recognize',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image:imageData,d4:$('d4').value})});
    const data=await r.json();if(!r.ok)throw Error(data.detail||'识别失败');
    dice=data.dice.map(d=>({...d,confirmed:false}));
    $('warning').textContent=data.warning||(!dice.length?'没有识别到骰子，请尝试更清晰的照片，或手动添加。':'模型读数尚未核对，请逐颗检查，特别留意 d4、6/9 和漏骰。');
    $('warning').hidden=false;status(`识别完成，用时 ${data.seconds} 秒。请核对每颗骰子。`);
  }catch(e){status(e.message||'识别失败，可重试或手动录入。',true);}
  finally{busy=false;updateButton();render();}
}
$('file').onchange=e=>{loadFile(e.target.files[0]);e.target.value='';};
$('camera').onclick=()=>$('camera-file').click();
$('camera-file').onchange=e=>{loadFile(e.target.files[0]);e.target.value='';};
$('replace').onclick=()=>$('file').click();
$('d4').onchange=()=>{if(imageData&&ready&&!busy){status('读数方式已更改，正在重新识别…');recognize(true);}};
$('dropzone').onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();$('file').click();}};
for(const ev of ['dragenter','dragover'])$('dropzone').addEventListener(ev,e=>{e.preventDefault();$('dropzone').classList.add('dragover');});
for(const ev of ['dragleave','drop'])$('dropzone').addEventListener(ev,e=>{e.preventDefault();$('dropzone').classList.remove('dragover');if(ev==='drop')loadFile(e.dataTransfer.files[0]);});
$('add').onclick=()=>{dice.push({kind:'unknown',value:null,note:'手动添加，请选择类型并填写读数。',bbox:null,confirmed:false});render();};
$('confirm-all').onclick=()=>{dice.forEach(d=>{d.confirmed=valid(d);});render();};
render();checkStatus();setInterval(checkStatus,30000);
