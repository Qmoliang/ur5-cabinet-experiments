'use strict';
window.MathJax={tex:{inlineMath:[['\\(','\\)']],displayMath:[['\\[','\\]']],tags:'none',processEscapes:true},svg:{fontCache:'local'},options:{enableMenu:false},startup:{pageReady(){return MathJax.startup.defaultPageReady().then(()=>{document.querySelectorAll('mjx-container[display="true"]').forEach(e=>{if(!e.closest('.math-block')){const w=document.createElement('div');w.className='math-block';e.before(w);w.append(e);}});document.body.dataset.mathReady='true';});}}};
document.addEventListener('DOMContentLoaded',async()=>{
 const search=document.querySelector('#doc-search'),cards=[...document.querySelectorAll('.doc-card')];let group='all';
 function filter(){let count=0;const q=search.value.trim().toLowerCase();cards.forEach(c=>{const show=(group==='all'||c.dataset.group===group)&&c.dataset.search.toLowerCase().includes(q);c.hidden=!show;if(show)count++;});document.querySelector('#doc-count').textContent=count+' / '+cards.length+' 份文档';document.querySelector('#no-documents').hidden=count>0;}
 if(search){search.addEventListener('input',filter);document.querySelectorAll('[data-filter]').forEach(b=>b.addEventListener('click',()=>{group=b.dataset.filter;document.querySelectorAll('[data-filter]').forEach(e=>{e.classList.toggle('active',e===b);e.setAttribute('aria-pressed',String(e===b));});filter();}));}
 document.querySelectorAll('.prose table').forEach(t=>{const wrap=document.createElement('div');wrap.className='table-scroll';t.before(wrap);wrap.append(t);});
 if(innerWidth<900){const toc=document.querySelector('.toc details');if(toc)toc.open=false;}
 if(window.mermaid&&document.querySelector('.mermaid')){
  try{mermaid.initialize({startOnLoad:false,securityLevel:'strict',theme:'neutral',fontFamily:'Arial, Microsoft YaHei, sans-serif',flowchart:{htmlLabels:false,useMaxWidth:true}});await mermaid.run({querySelector:'.mermaid'});document.body.dataset.diagramsReady='true';}
  catch(e){document.body.dataset.diagramError=String(e);console.error('Diagram rendering failed',e);}
 }
 document.body.dataset.libraryReady='true';
});

