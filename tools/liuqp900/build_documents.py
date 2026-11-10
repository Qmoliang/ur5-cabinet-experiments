from pathlib import Path
import ast,json,hashlib,re,zipfile,io,shutil,argparse
W=Path(__file__).resolve().parents[2];parser=argparse.ArgumentParser();parser.add_argument('--methods-root',type=Path,required=True);parser.add_argument('--pure-evidence-archive',type=Path,required=True);args=parser.parse_args();SRC=args.methods_root;D=W/'docs/library/files/liuqp900';D.mkdir(parents=True,exist_ok=True);PUB=W/'docs/liuqp900';CODE=W/'experiment/liuqp900';CODE.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,s):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s,encoding='utf-8',newline='\n')
def tex_md(s):
 s=re.sub(r'\\section\{([^{}]*)\}',r'## \1',s);s=re.sub(r'\\subsection\{([^{}]*)\}',r'### \1',s);s=re.sub(r'\\textbf\{([^{}]*)\}',r'**\1**',s)
 s=re.sub(r'\\begin\{equation\}(.*?)\\end\{equation\}',lambda m:'\n$$\n'+m[1].strip()+'\n$$\n',s,flags=re.S)
 s=re.sub(r'\\\[(.*?)\\\]',lambda m:'\n$$\n'+m[1].strip()+'\n$$\n',s,flags=re.S)
 s=re.sub(r'\\begin\{enumerate\}(.*?)\\end\{enumerate\}',lambda m:'\n'+ '\n'.join(f'{i}. '+x.strip() for i,x in enumerate(m[1].split(r'\item')[1:],1))+'\n',s,flags=re.S)
 return s.strip()+'\n'
def txt_md(s):
 lines=s.splitlines();out=[]
 for i,line in enumerate(lines):
  if i==0:out.append('# '+line)
  elif re.match(r'^\d+\. ',line):out.append('## '+line)
  elif re.match(r'^（\d+）',line):out.append('- '+re.sub(r'^（\d+）','',line))
  else:out.append(line)
 return '\n'.join(out).strip()+'\n'
provenance=[]
def doc(name,title,source,kind='tex'):
 p=SRC/source;s=p.read_text(encoding='utf-8');body=tex_md(s) if kind=='tex' else txt_md(s)
 if kind=='tex':body='# '+title+'\n\n'+body
 else:body=re.sub(r'^# .*', '# '+title,body,count=1)
 prefix='> 版本：2026-11-10，依据最终冻结代码整理。数学命题的假设和实现缺口均保留；这不是全局收敛或硬实时保证。原始实验没有改写。\n\n'
 first,rest=body.split('\n',1);body=first+'\n\n'+prefix+rest.lstrip();put(D/name,body);provenance.append(dict(file=name,source=source,source_sha256=sha(p),derived_markdown_sha256=sha(D/name),transformation='Headings, emphasis, display-math delimiters and lists converted; mathematical expressions preserved.'))
doc('methods.md','可行性保持执行与终态引导持续绕行：方法初稿','methods_body.tex')
doc('proofs.md','数学性质、条件证明与未证明边界','proofs.tex')
doc('property-status.md','证明、契约与实验结论速查','PROPERTY_STATUS.txt','txt')
doc('author-notes.md','冻结代码与历史推导的更正说明','AUTHOR_NOTES.txt','txt')
# Publish the active local-module closure as browsable byte-exact source.
S=SRC/'source_snapshot';available={p.stem:p for p in (S/'frozen').glob('*.py')};needed=set();todo=['pilot_worker_goal_plane_memory','goal_plane_memory_policy','dispatch_policy','baseline_policy','configuration_recurrence_policy','build_goal_ports','build_rank_aware_ports','verified_qp_recovery','audit_pilot','original_row_certificate','certified_auxiliary']
while todo:
 n=todo.pop()
 if n in needed or n not in available:continue
 needed.add(n);t=ast.parse(available[n].read_text(encoding='utf-8'))
 for node in ast.walk(t):
  mods=[node.module] if isinstance(node,ast.ImportFrom) else [x.name for x in node.names] if isinstance(node,ast.Import) else []
  for mod in mods:
   if mod in available and mod not in needed:todo.append(mod)
for n in sorted(needed):
 p=CODE/'frozen'/available[n].name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(available[n].read_bytes())
claims=json.loads((SRC/'CLAIM_CODE_MAP.json').read_text(encoding='utf-8'));lines=['# 主张与冻结代码映射','','证明状态和限制与方法初稿一致。源码按原字节公开，哈希可核验。','','| ID | 主张 | 类型 | 边界 |','| --- | --- | --- | --- |']
for c in claims:lines.append('| '+c['id']+' | '+c['claim']+' | `'+c['status']+'` | '+c['limits']+' |')
for c in claims:
 lines+=['','## '+c['id'],'',c['claim'],'','**边界：** '+c['limits'],'']
 for ref in c['code']:
  src=SRC/ref['file'];rel=Path(ref['file']).relative_to('source_snapshot');dest=CODE/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(src.read_bytes());assert sha(dest)==ref['sha256'];url='https://github.com/Qmoliang/ur5-cabinet-experiments/blob/main/experiment/liuqp900/'+rel.as_posix()+'#L'+str(ref['line_start']);lines.append('- ['+rel.as_posix()+':'+str(ref['line_start'])+']('+url+') · `'+ref['sha256']+'`')
 if c.get('proof_section'):lines.append('- 证明：[条件与推导](proofs.md)，'+c['proof_section'])
 if c.get('evidence'):lines.append('- 实验证据：[三组对照](results.md)')
put(D/'code-alignment.md','\n'.join(lines)+'\n')
# Full frozen-source bytes remain reconstructable from bounded downloadable ZIP parts.
sourcefiles=[p for p in sorted(S.rglob('*')) if p.is_file()];sourceparts=[];filemap=[]
def sourcepack(entries,part):
 b=io.BytesIO()
 with zipfile.ZipFile(b,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in entries:
   n=p.relative_to(S).as_posix();zi=zipfile.ZipInfo(n,(2026,11,10,0,0,0));zi.compress_type=zipfile.ZIP_DEFLATED;zi.external_attr=0o100644<<16;z.writestr(zi,p.read_bytes())
 raw=b.getvalue()
 if len(raw)>6*1024**2 and len(entries)>1:
  mid=len(entries)//2;sourcepack(entries[:mid],part+'a');sourcepack(entries[mid:],part+'b');return
 assert len(raw)<8*1024**2;name='evidence/source-'+part+'.zip';(PUB/name).write_bytes(raw);sourceparts.append(dict(file=name,sha256=sha(PUB/name),bytes=len(raw),files=len(entries)))
 for p in entries:filemap.append(dict(path=p.relative_to(S).as_posix(),sha256=sha(p),bytes=p.stat().st_size,bundle=name))
sourcepack(sourcefiles,'frozen')
# C's exact worker and audit driver overrides, saved independently from AB sources.
with zipfile.ZipFile(args.pure_evidence_archive) as z:
 for n in ['frozen/pure_worker.py','frozen/legacy_worker_reference.py','frozen/preflight_reference_worker.py','study/run_formal.py','study/audit_formal.py','study/audit_run.py','study/audit_original_feasibility.py']:
  dest=CODE/'pure-C'/n;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(z.read(n))
put(CODE/'README.md','# 冻结运行源码\n\n`frozen/` 是 A/B 活跃本地模块闭包及方法映射引用，按原字节保留。`dependencies/` 与 `study/` 包含文档直接引用的底层实现和审计。完整冻结源码及依赖分包见[来源清单](../../docs/liuqp900/evidence/source-manifest.json)。`pure-C/` 是第三组原执行入口与审计适配。\n\n这些代码带原环境依赖和路径，尚未验证干净环境一键重跑。浏览器播放使用保存状态和正运动学，不运行此控制代码。\n\n第三方文件保留原许可；本次发布未为原创研究代码新增 blanket license。\n')
put(PUB/'evidence/source-manifest.json',json.dumps(dict(parts=sourceparts,files=filemap,original_bytes=True,browsable_modules=sorted(needed)),ensure_ascii=False,indent=2))
for n in ['methods.tex','methods_body.tex','proofs.tex','VALIDATION.json','ALGEBRA_CHECKS.json','CONTRIBUTION_EVIDENCE.json','SOURCE_CHECKS.json','REFERENCE_FREEZE.json','CLASS_CHAIN.json','EFFECTIVE_METHODS.json']:
 p=PUB/'evidence/methods'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((SRC/n).read_bytes())
put(PUB/'evidence/markdown-provenance.json',json.dumps(provenance,ensure_ascii=False,indent=2))
# Put MD counterparts beside the original authoring files too, without overwriting TXT/TeX.
for name,alias in [('methods.md','METHODS_DRAFT.md'),('proofs.md','PROOFS_AND_LIMITS.md'),('property-status.md','PROPERTY_STATUS.md'),('author-notes.md','AUTHOR_NOTES.md'),('code-alignment.md','CLAIM_CODE_MAP.md')]:
 (SRC/alias).write_bytes((D/name).read_bytes())
print(json.dumps(dict(markdown_documents=len(provenance)+1,active_modules=len(needed),source_files=len(filemap),source_parts=len(sourceparts))))
