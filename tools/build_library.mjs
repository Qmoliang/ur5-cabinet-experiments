import {marked} from '../docs/library/vendor/marked-17.0.5.mjs';
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const ROOT=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..'),LIB=path.join(ROOT,'docs/library');
const manifest=JSON.parse(fs.readFileSync(path.join(LIB,'manifest.json'),'utf8'));
const esc=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const url=x=>x.split('/').map(encodeURIComponent).join('/');
const repo='https://github.com/Qmoliang/ur5-cabinet-experiments';
const bySource=new Map(manifest.documents.map(d=>[d.source.replaceAll('\\','/').toLowerCase(),d]));
let current,headings,mathCount,diagramCount,unresolved,displayFixes;
marked.use({gfm:true,breaks:false,extensions:[
 {name:'displayMath',level:'block',start:s=>s.search(/\\\[|\$\$/),tokenizer(s){
  const m=/^(?:\\\[([\s\S]*?)\\\]|\$\$([\s\S]*?)\$\$)(?:\n|$)/.exec(s);
  if(m)return {type:'displayMath',raw:m[0],text:m[1]??m[2]};
 },renderer(t){if(t.text.includes('不加入 QP}')&&t.text.trim().endsWith('}\n}')){t.text=t.text.replace(/\n\}\s*$/,'');displayFixes++;}mathCount++;return '<div class="math-block">\\['+esc(t.text)+'\\]</div>\n';}},
 {name:'inlineMath',level:'inline',start:s=>s.search(/\\\(|\$/),tokenizer(s){
  const m=/^\\\(([\s\S]*?)\\\)/.exec(s)||/^\$([^\s$](?:[^$\n]*?[^\s$])?)\$(?!\d)/.exec(s);
  if(m)return {type:'inlineMath',raw:m[0],text:m[1]};
 },renderer(t){mathCount++;return '<span class="math-inline">\\('+esc(t.text)+'\\)</span>';}}
],renderer:{
 heading(t){const id='section-'+(headings.length+1);headings.push({id,depth:t.depth,text:t.text.replace(/<[^>]*>/g,'')});return '<h'+t.depth+' id="'+id+'">'+this.parser.parseInline(t.tokens)+'</h'+t.depth+'>\n';},
 code(t){if((t.lang||'').trim()==='mermaid'){diagramCount++;return '<pre class="mermaid">'+esc(t.text)+'</pre>\n';}return '<pre><code>'+esc(t.text)+'</code></pre>\n';},
 html(t){return esc(t.text);},
 link(t){
  let href=t.href;const label=this.parser.parseInline(t.tokens);
  if(/^https?:\/\//i.test(href))return '<a href="'+esc(href)+'" rel="noopener noreferrer">'+label+'</a>';
  if(href.startsWith('#'))return '<a href="'+esc(href)+'">'+label+'</a>';
  let clean=decodeURIComponent(href).replaceAll('\\','/').replace(/^D:\/MuJoCo\//i,'').replace(/:\d+$/,'');
  const relative=path.posix.normalize(path.posix.join(path.posix.dirname(current.source),clean));
  const d=bySource.get(clean.toLowerCase())||bySource.get(relative.toLowerCase());
  if(d)return '<a href="'+d.id+'.html">'+label+'</a>';
  const canonical=relative.startsWith('Cabinet_Experiments/rerun_grounded_20260917/')?relative.slice('Cabinet_Experiments/rerun_grounded_20260917/'.length):null;
  if(canonical){const published=canonical.startsWith('audits/')?'reports/'+canonical:'experiment/'+canonical;if(fs.existsSync(path.join(ROOT,published))&&fs.statSync(path.join(ROOT,published)).isFile())return '<a href="'+repo+'/blob/main/'+url(published)+'">'+label+'</a>';}
  unresolved.push({document:current.id,target:href});
  return '<span class="archive-link" title="原始归档路径：'+esc(href)+'">'+label+'</span>';
 },
 image(t){if(current.assets?.[t.href])return '<img src="../'+url(current.assets[t.href])+'" alt="'+esc(t.text)+'" loading="lazy">';unresolved.push({document:current.id,target:t.href,image:true});return '<span class="archive-link">[原文图片：'+esc(t.text||t.href)+']</span>';}
}});
function head(title,base,math=false,diagram=false){
 return '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="theme-color" content="#f5f5ee"><title>'+esc(title)+' · UR5 文档库</title><link rel="stylesheet" href="'+base+'library.css"><script src="'+base+'reader.js" defer></script>'+(math?'<script src="'+base+'vendor/mathjax-3.2.2.js" defer></script>':'')+(diagram?'<script src="'+base+'vendor/mermaid-12.0.0.js" defer></script>':'')+'</head><body><a class="skip" href="#content">跳转到正文</a>';
}
function nav(base){return '<header class="library-nav"><a class="brand" href="'+base+'../index.html">↗ CABINET <span>/ RESEARCH LIBRARY</span></a><nav><a href="'+base+'index.html">文档目录</a><a href="'+base+'../index.html#experiment">实验回放</a><a href="'+repo+'">GitHub ↗</a></nav></header>';}
const footer=base=>'<footer><a href="'+base+'index.html">返回文档目录</a><span>保留原文 · 数学与实验记录 · UR5 Cabinet Experiments</span><a href="'+base+'licenses.html">渲染组件许可</a></footer>';
const records=[],allUnresolved=[];
fs.mkdirSync(path.join(LIB,'pages'),{recursive:true});
for(const d of manifest.documents){
 current=d;headings=[];mathCount=0;diagramCount=0;unresolved=[];displayFixes=0;
 const file='../'+url(d.file);
 const body=d.format==='md'?marked.parse(fs.readFileSync(path.join(LIB,d.file),'utf8')):'<section class="word-panel"><span class="format-icon">W</span><h2>英文扩展版 Word 原件</h2><p>下载后用 Word 打开，可查看原始排版、目录和可编辑公式。文件按原始字节保留，第一章 Markdown 是独立文档。</p><a class="primary" download href="'+file+'">下载 Word 原件 ↓</a><p class="file-name">'+esc(d.filename)+'</p></section>';
 const toc=headings.filter(h=>h.depth<=3).map(h=>'<a class="depth-'+h.depth+'" href="#'+h.id+'">'+esc(h.text.replaceAll(String.fromCharCode(96),''))+'</a>').join('');
 const group=manifest.groups.find(g=>g.id===d.group).title;
 const html=head(d.title,'../',mathCount>0,diagramCount>0)+nav('../')+'<main class="reading"><aside class="toc"><a class="back-link" href="../index.html">← 全部文档</a><details open><summary>本页目录</summary><div>'+toc+'</div></details></aside><div class="document"><header class="document-header"><div class="eyebrow">'+esc(group)+' / '+d.format.toUpperCase()+'</div><h1>'+esc(d.title)+'</h1><p>'+esc(d.note)+'</p><div class="actions"><a class="primary" download href="'+file+'">下载原文件 ↓</a><a href="'+repo+'/blob/main/docs/library/'+url(d.file)+'">在 GitHub 查看 ↗</a></div><div class="version-note">阅读说明：正文按原文件保留。历史版本的代理数量、场景和结论不自动代表当前 19 组重跑。原文中的本机路径对应其原始实验目录。'+(displayFixes?' 此页公式排版修正了一处多余的闭括号；下载的 Markdown 原件未改动。':'')+'</div></header><article id="content" class="prose">'+body+'</article><p class="source-note">原始来源：<code>'+esc(d.source)+'</code><br>SHA-256：<code>'+d.sha256+'</code></p></div></main>'+footer('../')+'</body></html>';
 fs.writeFileSync(path.join(LIB,'pages',d.id+'.html'),html);
 records.push({id:d.id,math:mathCount,diagrams:diagramCount,headings:headings.length,unresolved:unresolved.length,displayFixes});allUnresolved.push(...unresolved);
}
const cards=manifest.documents.map(d=>'<article class="doc-card" data-group="'+d.group+'" data-search="'+esc(d.title+' '+d.note+' '+d.filename)+'"><span class="badge">'+d.format.toUpperCase()+'</span><h3><a href="pages/'+d.id+'.html">'+esc(d.title)+'</a></h3><p>'+esc(d.note)+'</p><div><a class="read-link" href="pages/'+d.id+'.html">'+(d.format==='md'?'在线阅读 →':'文件详情 →')+'</a><a download href="'+url(d.file)+'">下载 '+d.format.toUpperCase()+' ↓</a></div></article>').join('');
const filters='<button class="active" data-filter="all" aria-pressed="true">全部</button>'+manifest.groups.map(g=>'<button data-filter="'+g.id+'" aria-pressed="false">'+esc(g.title)+'</button>').join('');
const index=head('数学、代码与实验文档','')+nav('')+'<main id="content" class="library-main"><section class="library-hero"><div class="eyebrow">DOCUMENTS / 数学 · 实现 · 证据</div><h1>从公式，读到实验。</h1><p>集中阅读 LiuQP 理论讲义、椭球数学推导、代码指南和后续实验记录。<br>Markdown 可在线阅读，Word 原件和所有 Markdown 均可下载。</p><div class="reading-path"><a href="pages/chapter1-word.html">理论讲义</a><span>→</span><a href="pages/ch02.html">椭球推导</a><span>→</span><a href="pages/ch03.html">代码阅读</a><span>→</span><a href="pages/current-overview.html">当前实验</a></div></section><aside class="collection-note"><strong>先确认版本，再比较结论。</strong><p>这里同时收录早期 v5.1 重复对照与后续“实验 5.1”连续支持体项目，它们不是同一批实验。当前演示页对应 2026-09-17 的落地柜子重跑。未找到独立的第四、五章成稿，因此目录列出已有结果报告和版本审计。</p></aside><section class="library-browser" aria-label="查找文档"><label for="doc-search">搜索文档</label><input type="search" id="doc-search" placeholder="输入章节、MVT、NEO、5.1…" autocomplete="off"><div class="filters" role="group" aria-label="文档分类">'+filters+'</div><p id="doc-count" aria-live="polite">'+manifest.documents.length+' 份文档</p><div class="doc-grid">'+cards+'</div><p id="no-documents" hidden>没有匹配的文档，请尝试其他关键词。</p></section></main>'+footer('')+'</body></html>';
fs.writeFileSync(path.join(LIB,'index.html'),index);
fs.writeFileSync(path.join(LIB,'licenses.html'),head('渲染组件许可','')+nav('')+'<main id="content" class="library-main prose"><h1>网页渲染组件</h1><p>固定版本随网站本地提供，用于显示 Markdown、公式与流程图。</p><ul><li>Marked 17.0.5 · <a href="vendor/Marked-LICENSE.md">MIT license</a> · <a href="https://marked.js.org/">项目文档</a></li><li>MathJax 3.2.2 · <a href="vendor/MathJax-LICENSE.txt">Apache 2.0 license</a> · <a href="https://docs.mathjax.org/en/v3.2/">项目文档</a></li><li>Mermaid 12.0.0 · <a href="vendor/Mermaid-LICENSE.txt">MIT license</a> · <a href="https://mermaid.js.org/">项目文档</a></li></ul><p>文档内容的权利不因网页渲染组件的开源许可而改变。</p></main>'+footer('')+'</body></html>');
fs.writeFileSync(path.join(LIB,'build-report.json'),JSON.stringify({documents:records,unresolved:allUnresolved},null,2));
console.log(JSON.stringify({documents:records.length,math:records.reduce((n,r)=>n+r.math,0),diagrams:records.reduce((n,r)=>n+r.diagrams,0),unresolved:allUnresolved}));

