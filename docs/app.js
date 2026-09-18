'use strict';
const $=s=>document.querySelector(s);
const pairs={
 known:{ids:['K01','K02'],note:'Known geometry · matched solid-volume covers · LiuQP'},
 online:{ids:['O01','O02'],note:'Independent online trajectories · maps update at recorded publication times'},
 historical:{ids:['H02','H01'],note:'Historical 4.4 sphere / 4.3 ellipsoid pipelines · fresh grounded-scene runs'},
 neo:{ids:['N04','N02'],note:'Known geometry · local NEO adaptation · 46 mm obstacle influence'},
 ablation:{ids:['A03'],note:'Online NEO ellipsoid, 46 mm · manipulability term removed · a separate ablation'}
};
let data,byId,pair='known',mode='real',seconds=0,playing=false;
const videos=[$('#video-a'),$('#video-b')];
const labels=['a','b'];
const escape=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const activeVideos=()=>videos.slice(0,pairs[pair].ids.length);
const statusClass=c=>c.outcome==='Reached & held'?'good':c.outcome==='Aborted'?'abort':'';
function pause(){playing=false;videos.forEach(v=>v.pause());$('#play').textContent='▶ Play comparison';$('#play').setAttribute('aria-label','Play recorded experiments');}
async function play(){
 if(!data)return;
 if(seconds>=60)seek(0);
 try{await Promise.all(activeVideos().map(v=>v.play()));playing=true;$('#play').textContent='Ⅱ Pause';$('#play').setAttribute('aria-label','Pause recorded experiments');}
 catch(e){pause();$('#pair-note').textContent='Playback could not start. Select an experiment and try again.';}
}
function seek(t){seconds=Math.max(0,Math.min(60,Number(t)));activeVideos().forEach(v=>{if(v.readyState>=2)v.currentTime=Math.min(seconds/6,Math.max(0,v.duration-.025));});updateTime();}
function updateTime(){
 $('#seek').value=seconds;$('#time').textContent=seconds.toFixed(2)+' / 60 s';
 const cursor=$('#chart-cursor');if(cursor){const x=48+seconds/60*930;cursor.setAttribute('x1',x);cursor.setAttribute('x2',x);}
 pairs[pair].ids.forEach((id,i)=>{
  const c=byId[id],el=$('#map-'+labels[i]);
  if(c.proxy_mode==='static'){el.textContent=c.static_count+' proxies · fixed known map';return;}
  if(c.proxy_mode!=='causal'){el.textContent='Proxy snapshots unavailable';return;}
  const shownTime=Math.floor(seconds/6*data.video_fps)*.3;
  const maps=c.maps.filter(m=>m.published<=Math.max(0,shownTime-.02)+1e-7);
  const m=maps.at(-1);
  el.textContent=m?m.count+' proxies · source '+m.source.toFixed(2)+' s · published '+m.published.toFixed(2)+' s':'Waiting for the first published map';
 });
}
function select(){
 pause();$('#pair-note').textContent=pairs[pair].note;
 $('#viewer-grid').classList.toggle('single',pairs[pair].ids.length===1);$('#card-b').hidden=pairs[pair].ids.length===1;
 pairs[pair].ids.forEach((id,i)=>{
  const c=byId[id],k=labels[i],v=videos[i];
  $('#case-'+k).textContent=id;$('#label-'+k).textContent=c.representation.toUpperCase();$('#label-'+k).className='rep-label '+c.representation;
  $('#error-'+k).innerHTML=c.final_error_mm.toFixed(3)+' <small>mm</small>';
  $('#status-'+k).textContent=c.outcome;$('#status-'+k).className='status '+statusClass(c);
  v.poster='assets/'+id+'-'+mode+'.webp';v.src='assets/'+id+'-'+mode+'.mp4';
  v.preload="auto";v.onloadeddata=()=>{v.currentTime=Math.min(seconds/6,Math.max(0,v.duration-.025));};v.load();
 });
 document.querySelectorAll('[data-pair]').forEach(b=>{const on=b.dataset.pair===pair;b.classList.toggle('active',on);b.setAttribute('aria-pressed',String(on));});
 document.querySelectorAll('[data-mode]').forEach(b=>{const on=b.dataset.mode===mode;b.classList.toggle('active',on);b.setAttribute('aria-pressed',String(on));});
 drawChart();updateTime();
}
function drawChart(){
 const y=e=>15+(3-Math.log10(Math.max(.0001,Math.min(1000,e))))/7*185;
 let svg='';
 [1000,100,10,1,.1,.01,.001].forEach(v=>{const yy=y(v);svg+='<line x1="48" x2="978" y1="'+yy+'" y2="'+yy+'" stroke="'+(v===1?'#99a786':'#e1e5d9')+'" '+(v===1?'stroke-dasharray="5 4"':'')+'/>';svg+='<text x="38" y="'+(yy+4)+'" text-anchor="end">'+v+'</text>';});
 [0,10,20,30,40,50,60].forEach(t=>{svg+='<text x="'+(48+t/60*930)+'" y="227" text-anchor="middle">'+t+' s</text>';});
 pairs[pair].ids.forEach(id=>{
  const c=byId[id],color=c.representation==='sphere'?'#bf623d':'#36768e';
  svg+='<path d="'+c.curve.map((p,i)=>(i?'L':'M')+(48+p[0]/60*930).toFixed(2)+','+y(p[1]).toFixed(2)).join(' ')+'" fill="none" stroke="'+color+'" stroke-width="2.4" />';
 });
 svg+='<line id="chart-cursor" x1="48" x2="48" y1="12" y2="203" stroke="#263d2a" stroke-opacity=".35" stroke-width="1"/>';
 $('#error-chart').innerHTML=svg;
}
function results(){
 const filter=$('#filter').value;
 const entries=data.cases.filter(c=>filter==='all'||filter==='known'&&/^[KN]/.test(c.id)||filter==='online'&&/^[HO]/.test(c.id)||filter==='ablation'&&c.id[0]==='A'||filter==='aborted'&&!c.complete);
 $('#results-body').innerHTML=entries.map(c=>'<tr><td>'+c.id+'</td><td>'+escape(c.name)+'</td><td>'+c.final_error_mm.toFixed(3)+' mm</td><td>'+(c.confirmed_time_s===null?'—':c.confirmed_time_s.toFixed(2)+' s')+'</td><td><span class="status '+statusClass(c)+'">'+escape(c.outcome)+'</span></td></tr>').join('');
}
function radius(){
 const r=Number($('#radius').value);let width=15,layer=1;
 while(width<=107.85158438426492+r+.005+.001){width*=2;layer++;}
 $('#radius-value').textContent=r+' mm';$('#radius-inline').textContent=r;$('#layer-result').textContent='Layer '+layer+' · '+width+' mm cells';
}
document.querySelectorAll('[data-pair]').forEach(b=>b.addEventListener('click',()=>{if(!data)return;pair=b.dataset.pair;seconds=0;select();}));
document.querySelectorAll('[data-mode]').forEach(b=>b.addEventListener('click',()=>{if(!data)return;mode=b.dataset.mode;select();}));
$('#play').addEventListener('click',()=>playing?pause():play());
$('#restart').addEventListener('click',()=>{pause();seek(0);});
$('#seek').addEventListener('input',e=>{pause();seek(e.target.value);});
$('#filter').addEventListener('change',()=>data&&results());
$('#radius').addEventListener('input',radius);
videos[0].addEventListener('timeupdate',()=>{if(playing){seconds=Math.min(60,videos[0].currentTime*6);if(activeVideos().length>1&&Math.abs(videos[1].currentTime-videos[0].currentTime)>.12)videos[1].currentTime=videos[0].currentTime;updateTime();}});
videos[0].addEventListener('ended',()=>{seconds=60;pause();updateTime();});
document.addEventListener('visibilitychange',()=>{if(document.hidden)pause();});
fetch('data/experiments.json').then(r=>{if(!r.ok)throw Error(r.status);return r.json();}).then(json=>{
 data=json;byId=Object.fromEntries(data.cases.map(c=>[c.id,c]));select();results();radius();document.body.dataset.ready='true';
}).catch(()=>{$('#load-error').hidden=false;$('#play').disabled=true;});

