/* BYOK dice reader: the key and photos never touch this site's server. */
const $ = id => document.getElementById(id);
const KINDS = {d4:[1,4],d6:[1,6],d8:[1,8],d10:[1,10],d12:[1,12],d20:[1,20],d00:[0,90],unknown:[0,100]};
const PRESETS = {
  deepseek:{label:'DeepSeek',base:'https://api.deepseek.com',model:'deepseek-flash',models:['deepseek-flash'],adapter:'openai',extra:{thinking:{type:'disabled'}}},
  openai:{label:'OpenAI',base:'https://api.openai.com/v1',model:'gpt-4o-mini',models:['gpt-4o-mini','gpt-4.1-mini','gpt-4o','gpt-5-mini'],adapter:'openai'},
  anthropic:{label:'Anthropic Claude',base:'https://api.anthropic.com/v1',model:'claude-sonnet-4-5',models:['claude-sonnet-4-5','claude-haiku-4-5','claude-opus-4-1'],adapter:'anthropic'},
  gemini:{label:'Google Gemini',base:'https://generativelanguage.googleapis.com/v1beta/openai',model:'gemini-2.5-flash',models:['gemini-2.5-flash','gemini-2.5-pro'],adapter:'openai'},
  openrouter:{label:'OpenRouter',base:'https://openrouter.ai/api/v1',model:'google/gemini-2.5-flash',models:['google/gemini-2.5-flash','openai/gpt-4o-mini','anthropic/claude-sonnet-4.5','qwen/qwen3-vl-235b-a22b-instruct'],adapter:'openai'},
  moonshot:{label:'Moonshot / Kimi',base:'https://api.moonshot.cn/v1',model:'moonshot-v1-8k-vision-preview',models:['moonshot-v1-8k-vision-preview','kimi-latest'],adapter:'openai'},
  dashscope:{label:'阿里云百炼 Qwen',base:'https://dashscope.aliyuncs.com/compatible-mode/v1',model:'qwen-vl-max',models:['qwen-vl-max','qwen-vl-plus','qwen3-vl-plus'],adapter:'openai'},
  custom:{label:'自定义（OpenAI 兼容）',base:'',model:'',models:[],adapter:'openai'}
};
const DETECT_PROMPT = run => `Inspect this overhead photograph of a Dungeons & Dragons dice tray.
Find each physical die exactly once, ignoring tray decorations and text elsewhere.
Read the roll result from the upper face, never sum side faces. A normal d10 marked 0 has roll value 10.
Values must stay inside: d4 1-4, d6 1-6, d8 1-8, d10 1-10, d12 1-12, d20 1-20.
A percentile tens die is kind d00 with value 0,10,20,...90.
d4 reading convention: ${run}. For d4 never assume the largest visible digit is the result.
If a die is ambiguous, occluded, blurred, cocked, or has unresolved 6/9 orientation, use null.
Return only JSON: {"dice":[{"kind":"d4|d6|d8|d10|d12|d20|d00|unknown","value":integer or null,"bbox":[left,top,right,bottom] normalized to 0..1000,"note":"short Chinese note"}],"warning":"short Chinese warning"}.
No dice means an empty dice list. Never fabricate hidden dice or numbers.`;
const DIE_PROMPT = run => `This image is a magnified crop of one D&D die from a tray photo.
Identify its type and read the face pointing up at the camera. Ignore other dice at the edges.
Kinds and ranges: d4 1-4, d6 1-6, d8 1-8, d10 1-10 (a face marked 0 counts as 10), d12 1-12, d20 1-20, d00 0,10,...90.
A d10 must never be reported as d20 or d00. Read only the upper face, never sum side faces.
d4 reading convention: ${run}. If the face is hidden, blurred, cocked or 6/9 orientation is unresolved, use null.
Return only JSON: {"kind":"d4|d6|d8|d10|d12|d20|d00|unknown","value":integer or null,"note":"short Chinese note"}.`;

let dice = [], imageCanvas = null, imageData = null, objectURL = null, busy = false, generation = 0;
let currentProvider = 'deepseek';

class ApiError extends Error {
  constructor(status, message){super(message); this.status = status;}
}

const ranges = KINDS;
function valid(d){const [lo,hi]=ranges[d.kind];return Number.isInteger(d.value)&&d.value>=lo&&d.value<=hi&&(d.kind!=='d00'||d.value%10===0);}
function status(text,error=false){$('status').textContent=text;$('status').classList.toggle('error',error);}
function provider(){return PRESETS[currentProvider];}
function configured(){return !!($('base-url').value.trim()&&$('api-key').value.trim()&&$('model').value.trim());}
function runRule(){return {auto:'infer top-reading or bottom-reading; if uncertain use null',top:'read the common digit at the top vertex',bottom:'read the common digit along the bottom edges'}[$('d4').value];}

function asInt(value){
  if(typeof value==='boolean')return null;
  if(typeof value==='number'&&Number.isInteger(value))return value;
  if(typeof value==='string'&&/^[+-]?\d+$/.test(value.trim()))return Number(value.trim());
  return null;
}
function asBox(value){
  if(!Array.isArray(value)||value.length!==4)return null;
  const coords=value.map(asInt);
  if(coords.some(v=>v===null))return null;
  const [x1,y1,x2,y2]=coords.map(v=>Math.max(0,Math.min(1000,v)));
  return x1<x2&&y1<y2?[x1,y1,x2,y2]:null;
}
function normalizeDie(raw, fallbackBox){
  if(!raw||typeof raw!=='object'||Array.isArray(raw))return null;
  let kind=typeof raw.kind==='string'&&KINDS[raw.kind]?raw.kind:'unknown';
  let value=asInt(raw.value);
  if(kind==='d10'&&value===0)value=10;
  if(value!==null){
    const [lo,hi]=KINDS[kind];
    if(value<lo||value>hi||(kind==='d00'&&value%10))value=null;
  }
  const bbox=fallbackBox||asBox(raw.bbox);
  if(!bbox)return null;
  return {kind,value,bbox,note:typeof raw.note==='string'?raw.note.slice(0,240):''};
}
function normalizeResult(raw){
  const list=raw&&Array.isArray(raw.dice)?raw.dice.slice(0,40):[];
  const diceOut=list.map(item=>normalizeDie(item)).filter(Boolean);
  return {dice:diceOut,dropped:list.length-diceOut.length,
          warning:raw&&typeof raw.warning==='string'?raw.warning.slice(0,500):''};
}
function parseJson(text){
  if(typeof text!=='string'||!text.trim())throw new Error('服务商返回了空内容');
  let body=text.trim().replace(/^```(?:json)?\s*/i,'').replace(/\s*```$/,'');
  const left=body.indexOf('{'),right=body.lastIndexOf('}');
  if(left<0||right<=left)throw new Error('服务商没有返回 JSON');
  return JSON.parse(body.slice(left,right+1));
}
function mapLimit(items,limit,fn){
  const results=new Array(items.length);
  let next=0;
  const workers=Array.from({length:Math.min(limit,items.length)},async()=>{
    while(true){const index=next++;if(index>=items.length)return;results[index]=await fn(items[index],index);}
  });
  return Promise.all(workers).then(()=>results);
}

function endpoint(base,suffix){
  const clean=base.trim().replace(/\/+$/,'');
  return clean.endsWith(suffix)?clean:clean+suffix;
}
async function postJson(url,payload,headers){
  let response;
  try{
    response=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json',...headers},body:JSON.stringify(payload)});
  }catch{
    throw new ApiError(0,'无法连接服务商：请检查网络、API 地址，或该服务是否允许浏览器直连（CORS）。');
  }
  const text=await response.text();
  let data=null;try{data=JSON.parse(text);}catch{}
  if(!response.ok){
    const detail=data?.error?.message||data?.message||text.slice(0,300)||('HTTP '+response.status);
    throw new ApiError(response.status,detail);
  }
  return data;
}
function without(payload,...keys){
  const copy={...payload};
  keys.forEach(key=>delete copy[key]);
  return copy;
}
async function openaiCall(prompt,imageBase64){
  const url=endpoint($('base-url').value,'/chat/completions');
  const payload={
    model:$('model').value.trim(),
    messages:[{role:'user',content:[
      {type:'text',text:prompt},
      {type:'image_url',image_url:{url:'data:image/jpeg;base64,'+imageBase64}}]}],
    temperature:0,
    response_format:{type:'json_object'},
    ...(provider().extra||{})
  };
  const headers={Authorization:'Bearer '+$('api-key').value.trim()};
  const attempts=[payload,without(payload,'response_format'),
    without(payload,'response_format','temperature'),
    without(payload,'response_format','temperature','thinking')];
  let last=null;
  for(const attempt of attempts){
    try{
      const data=await postJson(url,attempt,headers);
      const content=data?.choices?.[0]?.message?.content;
      if(typeof content==='string')return content;
      if(Array.isArray(content))return content.map(part=>part?.text||'').join('');
      throw new ApiError(502,'服务商没有返回文本结果');
    }catch(error){
      last=error;
      const retryable=error.status===400||error.status===422;
      if(!retryable||!/response_format|temperature|thinking|unsupported|not supported|invalid.*param|unknown.*param/i.test(error.message))throw error;
    }
  }
  throw last;
}
async function anthropicCall(prompt,imageBase64){
  const url=endpoint($('base-url').value,'/messages');
  const payload={
    model:$('model').value.trim(),
    max_tokens:2048,
    temperature:0,
    messages:[{role:'user',content:[
      {type:'text',text:prompt},
      {type:'image',source:{type:'base64',media_type:'image/jpeg',data:imageBase64}}]}]
  };
  const data=await postJson(url,payload,{
    'x-api-key':$('api-key').value.trim(),
    'anthropic-version':'2023-06-01',
    'anthropic-dangerous-direct-browser-access':'true'
  });
  const text=(data?.content||[]).filter(part=>part?.type==='text').map(part=>part.text).join('\n');
  if(!text)throw new ApiError(502,'Claude 没有返回文本结果');
  return text;
}
function askModel(prompt,imageBase64){
  return provider().adapter==='anthropic'?anthropicCall(prompt,imageBase64):openaiCall(prompt,imageBase64);
}

function cropDie(bbox){
  if(!imageCanvas)return null;
  const [x1,y1,x2,y2]=bbox;
  const padX=(x2-x1)*0.18,padY=(y2-y1)*0.18;
  const left=Math.max(0,Math.round((x1-padX)*imageCanvas.width/1000));
  const top=Math.max(0,Math.round((y1-padY)*imageCanvas.height/1000));
  const right=Math.min(imageCanvas.width,Math.round((x2+padX)*imageCanvas.width/1000));
  const bottom=Math.min(imageCanvas.height,Math.round((y2+padY)*imageCanvas.height/1000));
  const width=right-left,height=bottom-top;
  if(width<24||height<24)return null;
  const scale=Math.max(1,360/Math.max(width,height));
  const canvas=document.createElement('canvas');
  canvas.width=Math.round(width*scale);canvas.height=Math.round(height*scale);
  canvas.getContext('2d').drawImage(imageCanvas,left,top,width,height,0,0,canvas.width,canvas.height);
  return canvas.toDataURL('image/jpeg',0.95).split(',')[1];
}
async function refine(result){
  if(!result.dice.length)return {result,refined:0,failed:0};
  let done=0;
  const answers=await mapLimit(result.dice,4,async die=>{
    const crop=cropDie(die.bbox);
    if(!crop)return null;
    try{
      return normalizeDie(parseJson(await askModel(DIE_PROMPT(runRule()),crop)),die.bbox);
    }catch{return null;}
    finally{done++;if(result.dice.length>1)status(`正在逐颗复核 ${done}/${result.dice.length}…`);}
  });
  let refined=0,failed=0;
  result.dice.forEach((die,index)=>{
    const answer=answers[index];
    if(!answer){failed++;return;}
    const kind=answer.kind!=='unknown'?answer.kind:die.kind;
    let value=answer.value;
    if(answer.kind==='unknown'&&value===null)value=die.value;
    const merged=normalizeDie({kind,value,note:answer.note||die.note},die.bbox);
    if(!merged){failed++;return;}
    die.kind=merged.kind;die.value=merged.value;die.note=merged.note;refined++;
  });
  return {result,refined,failed};
}

function percentilePairs(){
  const {pairs,usedUnits}=pairPercentile(dice);
  return {pairs,usedUnits,pairedTens:new Set(pairs.map(p=>p.tens))};
}
function renderSummary(){
  const known=dice.filter(valid),pending=dice.filter(d=>!valid(d)||!d.confirmed).length;
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
    line.textContent=`百分位配对：第 ${p.tens+1} 颗（${dice[p.tens].value}）+ 第 ${p.units+1} 颗（${dice[p.units].value%10}）= ${p.total}`;
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
function updateButtons(){
  for(const id of ['camera','replace','file','camera-file','d4','add','rerun']){const el=$(id);if(el)el.disabled=busy;}
}

async function loadFile(file){
  if(busy||!file)return;
  if(file.type&&!['image/jpeg','image/png','image/webp'].includes(file.type)){status('请选择 JPG、PNG 或 WebP 图片',true);return;}
  if(file.size>15*1024*1024){status('照片不能超过 15MB',true);return;}
  const version=++generation;
  try{
    const bitmap=await createImageBitmap(file,{imageOrientation:'from-image'});
    if(version!==generation){bitmap.close();return;}
    if(bitmap.width*bitmap.height>32000000){bitmap.close();throw Error('照片分辨率过大，请缩小到 3200 万像素以内');}
    const scale=Math.min(1,1600/Math.max(bitmap.width,bitmap.height));
    const canvas=document.createElement('canvas');canvas.width=Math.round(bitmap.width*scale);canvas.height=Math.round(bitmap.height*scale);
    canvas.getContext('2d').drawImage(bitmap,0,0,canvas.width,canvas.height);bitmap.close();
    const blob=await new Promise(resolve=>canvas.toBlob(resolve,'image/jpeg',0.95));if(version!==generation)return;
    if(!blob)throw Error('无法读取照片');
    imageCanvas=canvas;imageData=canvas.toDataURL('image/jpeg',0.95).split(',')[1];
    if(objectURL)URL.revokeObjectURL(objectURL);objectURL=URL.createObjectURL(blob);$('photo').src=objectURL;
    dice=[];render();$('warning').hidden=true;$('dropzone').hidden=true;$('preview').hidden=false;$('replace').hidden=false;$('rerun').hidden=false;
    updateButtons();
    if(configured())recognize(true);
    else status('照片已就绪；填写 API Key、地址和模型后会自动识别。');
  }catch(error){if(version===generation)status(error.message||'无法读取照片',true);}
}
async function recognize(force=false){
  if(busy||!imageData)return;
  if(!configured()){status('请先填写 API Key、API 地址和模型',true);return;}
  if(!force&&dice.length&&!window.confirm('重新识别会替换当前读数和手动修改，继续吗？'))return;
  const started=performance.now();
  busy=true;updateButtons();render();status('正在识别：先定位骰子，再逐颗放大复核…');
  try{
    const detected=normalizeResult(parseJson(await askModel(DETECT_PROMPT(runRule()),imageData)));
    let refined=0,failed=0;
    if(detected.dice.length){
      const output=await refine(detected);
      refined=output.refined;failed=output.failed;
    }
    let warning=detected.warning||'';
    if(!detected.dice.length&&!warning)warning='没有识别到骰子，请换一张更清晰的照片，或手动添加。';
    if(detected.dropped)warning=`有 ${detected.dropped} 颗骰子格式不完整，已忽略；`+warning;
    if(refined)warning=`已逐颗放大复核 ${refined} 颗骰子；`+warning;
    if(failed)warning=`${failed} 颗骰子未能复核，仅用整图读数；`+warning;
    warning=(`BYOK：照片直接发送至 ${provider().label}。`+warning).slice(0,500);
    dice=detected.dice.map(die=>({...die,confirmed:false}));
    $('warning').textContent=warning;
    $('warning').hidden=false;
    status(`识别完成，用时 ${((performance.now()-started)/1000).toFixed(1)} 秒。请核对每颗骰子。`);
  }catch(error){
    let message=error.message||'识别失败';
    if(error.status===401||error.status===403)message='API Key 无效或无权限：'+message;
    else if(error.status===429)message='服务商限流或余额不足：'+message;
    status(message,true);
  }finally{busy=false;updateButtons();render();}
}

/* ---------- provider and key storage ---------- */
function rememberOn(){return $('remember').checked;}
function primaryStore(){return rememberOn()?localStorage:sessionStorage;}
function secondaryStore(){return rememberOn()?sessionStorage:localStorage;}
function keyName(id){return 'dice-byok.key.'+id;}
function loadKey(id){return primaryStore().getItem(keyName(id))||'';}
function saveKey(){
  const id=currentProvider,key=$('api-key').value.trim();
  if(key)primaryStore().setItem(keyName(id),key);else primaryStore().removeItem(keyName(id));
  secondaryStore().removeItem(keyName(id));
}
function saveBaseModel(){
  try{localStorage.setItem('dice-byok.base.'+currentProvider,$('base-url').value.trim());
    localStorage.setItem('dice-byok.model.'+currentProvider,$('model').value.trim());}catch{}
}
function fillModels(){
  const list=$('model-list');list.replaceChildren();
  (provider().models||[]).forEach(name=>{const option=document.createElement('option');option.value=name;list.append(option);});
}
function applyPreset(id){
  currentProvider=id;
  const preset=PRESETS[id];
  $('base-url').value=localStorage.getItem('dice-byok.base.'+id)??preset.base;
  $('model').value=localStorage.getItem('dice-byok.model.'+id)??preset.model;
  $('api-key').value=loadKey(id);
  fillModels();
  try{localStorage.setItem('dice-byok.provider',id);}catch{}
}

function buildProviders(){
  Object.entries(PRESETS).forEach(([id,preset])=>{
    const option=document.createElement('option');option.value=id;option.textContent=preset.label;$('provider').append(option);
  });
}
function initSettings(){
  buildProviders();
  $('remember').checked=localStorage.getItem('dice-byok.remember')==='1';
  currentProvider=localStorage.getItem('dice-byok.provider')||'deepseek';
  if(!PRESETS[currentProvider])currentProvider='deepseek';
  $('provider').value=currentProvider;
  applyPreset(currentProvider);
}

$('file').onchange=event=>{loadFile(event.target.files[0]);event.target.value='';};
$('camera').onclick=()=>$('camera-file').click();
$('camera-file').onchange=event=>{loadFile(event.target.files[0]);event.target.value='';};
$('replace').onclick=()=>$('file').click();
$('rerun').onclick=()=>recognize(true);
$('key-toggle').onclick=()=>{
  const input=$('api-key');const show=input.type==='password';
  input.type=show?'text':'password';$('key-toggle').textContent=show?'隐藏':'显示';
};
$('clear-key').onclick=()=>{
  localStorage.removeItem(keyName(currentProvider));
  sessionStorage.removeItem(keyName(currentProvider));
  $('api-key').value='';status('已清除当前服务商的本地密钥');
};
function migrateKeys(to,from){
  const keys=[];
  for(let index=0;index<from.length;index++){
    const key=from.key(index);
    if(key&&key.startsWith('dice-byok.key.'))keys.push(key);
  }
  keys.forEach(key=>{to.setItem(key,from.getItem(key));from.removeItem(key);});
}
$('remember').onchange=()=>{
  try{localStorage.setItem('dice-byok.remember',$('remember').checked?'1':'0');}catch{}
  if($('remember').checked)migrateKeys(localStorage,sessionStorage);
  else migrateKeys(sessionStorage,localStorage);
  saveKey();
  status($('remember').checked?'密钥将保存在本浏览器 localStorage':'密钥只保存在本标签页，关闭后清除');
};
$('provider').onchange=()=>{
  saveKey();saveBaseModel();
  applyPreset($('provider').value);
  status('已切换服务商，请确认 API 地址与模型');
};
for(const id of ['api-key','base-url','model']){
  $(id).addEventListener('change',()=>{
    saveKey();saveBaseModel();
    if(imageData&&!busy&&!dice.length&&configured())recognize(true);
  });
}
$('d4').onchange=()=>{if(imageData&&configured()&&!busy){status('读数方式已更改，正在重新识别…');recognize(true);}};
$('dropzone').onkeydown=event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();$('file').click();}};
for(const eventName of ['dragenter','dragover'])$('dropzone').addEventListener(eventName,event=>{event.preventDefault();$('dropzone').classList.add('dragover');});
for(const eventName of ['dragleave','drop'])$('dropzone').addEventListener(eventName,event=>{event.preventDefault();$('dropzone').classList.remove('dragover');if(eventName==='drop')loadFile(event.dataTransfer.files[0]);});
$('add').onclick=()=>{dice.push({kind:'unknown',value:null,note:'手动添加，请选择类型并填写读数。',bbox:null,confirmed:false});render();};
$('confirm-all').onclick=()=>{dice.forEach(d=>{d.confirmed=valid(d);});render();};

initSettings();
render();
status(configured()?'配置已保存，拍照或选择照片开始识别':'请填写 API Key、API 地址和模型');
