const $ = id => document.getElementById(id);
const fmt = (v,n=2) => v == null ? '—' : Number(v).toFixed(n);
let index, selected, replay, loadingEngine, loadToken=0;
const views={1:'实体环境是事后查看用的真值几何，不代表在线控制器当时已经观测到整个场景。',2:'原模型的65个机器人球证书；环境仍为真值盒体，不是完整环境代理或安全证明。',3:'蓝线为保存的末端路径，橙点为目标；路径只显示到当前回放帧。',4:'每个发布代次最多2,000个采样点；在线为测量单元中心，已知为静态代理中心。隐藏真值障碍；不表示完整未知空间。'};
async function json(url){const r=await fetch(url);if(!r.ok)throw Error(`${url}: HTTP ${r.status}`);return r.json();}
function metric(name,value){const dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=name;dd.textContent=value;$('case-metrics').append(dt,dd);}
function selectCase(id){const c=index.cases.find(x=>x.id===id);if(!c)return;$('mode').value=c.mode;$('task').value=c.task;$('seed').value=c.seed;changeCase();}
async function changeCase(){
  const token=++loadToken;replay?.pause();selected=index.cases.find(c=>c.mode===$('mode').value&&c.task===$('task').value&&c.seed===Number($('seed').value));
  if(!selected)return;
  const c=selected;history.replaceState(null,'',`#case=${encodeURIComponent(c.id)}`);
  $('case-label').textContent=`${c.mode==='online'?'在线':'已知地图'} / ${c.label} / seed ${c.seed} / ${c.version}`;
  $('case-status').textContent=c.passed?'安全到达 · 通过':'未到达 · 失败保留';$('case-status').className=c.passed?'passed':'failed';
  $('case-metrics').replaceChildren();metric('最终位置误差',`${fmt(c.final_error_mm,3)} mm`);metric('保存时域',`${fmt(c.duration_s)} s`);metric('到达计入耗时（原口径）',c.completion_s==null?'未到达':`${fmt(c.completion_s)} s`);metric('控制器 P99',`${fmt(c.controller_p99_ms)} ms`);metric('最低保守代理间隙',`${fmt(c.minimum_proxy_gap_mm,3)} mm`);metric('停止原因',c.termination);
  $('audit-link').href=c.audit;$('trajectory-link').href=c.trajectory;
  const video=$('recording');video.pause();video.removeAttribute('src');video.removeAttribute('poster');video.load();video.hidden=!c.video;
  if(c.video){video.src=c.video;video.poster=c.video.replace('.mp4','.webp');$('video-note').textContent='保存轨迹渲染 · 6× 倍速 · 不是新实验。三维回放与录像可独立操作。';}
  else $('video-note').textContent='此起点提供完整三维轨迹回放。直接录像选取每组各任务 seed 1 和两项失败，共32段；可选择 seed 1 查看。';
  $('seek').disabled=true;$('play').disabled=true;$('time').textContent='0.00 s';$('frame-metric').textContent='';
  if(replay){try{await replay.load(c,token);if(token===loadToken)enablePlayback();}catch(e){showError(e);}}
}
function enablePlayback(){$('seek').disabled=false;$('play').disabled=false;$('seek').max=selected.frames-1;$('seek').value=0;$('play').textContent='▶ 播放';}
function showError(e){console.error(e);$('loading').hidden=false;$('loading').textContent=`回放加载失败：${e.message}。可使用下方录像或重新点击加载。`;$('load3d').disabled=false;}
async function loadEngine(){
  if(loadingEngine)return loadingEngine;
  $('load3d').disabled=true;$('loading').hidden=false;$('loading').textContent='正在加载 MuJoCo 与原机器人模型…';
  loadingEngine=(async()=>{try{
    if(!replay){const {Replay}=await import('./replay.js');replay=await Replay.create($('viewport'),{
      onTime(frame,error){$('seek').value=frame;$('time').textContent=`${fmt(frame*.02)} s`;$('frame-metric').textContent=`帧 ${frame}/${selected.frames-1} · 误差 ${fmt(error,3)} mm`;},
      onPause(){$('play').textContent='▶ 播放';},isCurrent(token){return token===loadToken;}
    });window.replayApi=replay;}
    await replay.load(selected,loadToken);enablePlayback();$('load3d').textContent='三维回放已启用';$('viewer').focus({preventScroll:true});
  }catch(e){showError(e);}finally{loadingEngine=null;}})();return loadingEngine;
}
function setView(n){if(!replay)return;replay.setView(n);$('view-note').textContent=views[n];document.querySelectorAll('[data-view]').forEach(b=>{const active=Number(b.dataset.view)===n;b.classList.toggle('active',active);b.setAttribute('aria-pressed',active);});}
function play(){if(!replay||$('play').disabled)return;replay.toggle();$('play').textContent=replay.playing?'Ⅱ 暂停':'▶ 播放';}
async function init(){
  document.body.dataset.ready='false';index=await json('data.json');
  for(const t of index.tasks){const o=new Option(t.label,t.id);$('task').add(o);const known=index.cases.filter(c=>c.mode==='known'&&c.task===t.id),online=index.cases.filter(c=>c.mode==='online'&&c.task===t.id);const tr=document.createElement('tr');for(const v of [t.label,`${known.filter(c=>c.passed).length}/20`,`${online.filter(c=>c.passed).length}/20`,online.filter(c=>!c.passed).map(c=>'seed '+c.seed).join('、')||'—']){const td=document.createElement('td');td.textContent=v;tr.append(td);}$('task-results').append(tr);}
  for(let i=1;i<=20;i++)$('seed').add(new Option(`seed ${i}`,i));
  for(const id of ['mode','task','seed'])$(id).addEventListener('change',changeCase);
  $('load3d').onclick=loadEngine;$('play').onclick=play;$('seek').oninput=()=>{replay?.pause();replay?.setFrame(Number($('seek').value));};$('speed').onchange=()=>{if(replay)replay.speed=Number($('speed').value);};$('reset-view').onclick=()=>replay?.resetCamera();
  document.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>setView(Number(b.dataset.view)));
  document.querySelectorAll('[data-open]').forEach(b=>b.onclick=async()=>{selectCase(b.dataset.open);$('replay').scrollIntoView({behavior:'smooth'});await loadEngine();});
  window.addEventListener('keydown',e=>{if(/INPUT|SELECT|TEXTAREA/.test(e.target.tagName)||e.ctrlKey||e.metaKey||e.altKey)return;const r=$('viewer').getBoundingClientRect();if(r.bottom<0||r.top>innerHeight)return;if(['1','2','3','4'].includes(e.key)){e.preventDefault();setView(Number(e.key));}if(e.code==='Space'){e.preventDefault();play();}});
  const hash=new URLSearchParams(location.hash.slice(1)),id=hash.get('case');selectCase(id&&index.cases.some(c=>c.id===id)?id:'online__cage__s08');document.body.dataset.ready='true';
}
init().catch(showError);
