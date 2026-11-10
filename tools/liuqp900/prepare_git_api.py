"""Prepare immutable Git objects for the authenticated GitHub Git-data connector."""
from pathlib import Path
import subprocess,json,hashlib,base64,sys,argparse
parser=argparse.ArgumentParser();parser.add_argument('--output',default='.local/github-api');args=parser.parse_args()
W=Path(__file__).resolve().parents[2];OUT=(W/args.output).resolve();assert OUT.is_relative_to(W),'Output must remain inside the repository';OUT.mkdir(parents=True,exist_ok=True);(OUT/'objects').mkdir(exist_ok=True)
def git(*args):return subprocess.check_output(['git',*args],cwd=W)
old={line.split(b'\t')[0].split()[2].decode() for line in git('ls-tree','-r','-z','HEAD').split(b'\0') if line};stage={}
for line in git('ls-files','-s','-z').decode('utf-8').split('\0'):
 if line:
  info,p=line.split('\t',1);mode,h,_=info.split();stage[p]=(mode,h)
paths=[x for x in git('diff','--cached','--name-only','-z','HEAD').decode('utf-8').split('\0') if x];entries=[];blobs={};text_ext={'.json','.html','.css','.js','.mjs','.cjs','.py','.md','.txt','.xml','.tex','.yml','.yaml'}
for p in paths:
 assert p in stage,'Deletions require explicit handling';mode,h=stage[p]
 if h in old:entries.append(dict(path=p,mode=mode,type='blob',sha=h));continue
 b=git('cat-file','blob',h);assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==h
 inline=Path(p).suffix.lower() in text_ext and len(b)<=48000
 if inline:
  try:s=b.decode('utf-8')
  except UnicodeDecodeError:inline=False
 if inline:entries.append(dict(path=p,mode=mode,type='blob',content=s))
 else:
  entries.append(dict(path=p,mode=mode,type='blob',sha=h))
  if h not in blobs:
   (OUT/'objects'/h).write_bytes(b);blobs[h]=dict(sha=h,bytes=len(b),parts=(len(b)+716798)//716799,paths=[p])
  else:blobs[h]['paths'].append(p)
plan=dict(repository='Qmoliang/ur5-cabinet-experiments',parent=git('rev-parse','HEAD').decode().strip(),base_tree=git('rev-parse','HEAD^{tree}').decode().strip(),expected_tree=git('write-tree').decode().strip(),entries=entries,blobs=list(blobs.values()),files=len(paths),total_blob_bytes=sum(x['bytes'] for x in blobs.values()))
(OUT/'plan.json').write_text(json.dumps(plan,ensure_ascii=False),encoding='utf-8',newline='\n');meta={k:v for k,v in plan.items() if k!='entries'};(OUT/'meta.json').write_text(json.dumps(meta),encoding='utf-8');
# Split tree entries into bounded JSON batches for the orchestration tool.
batch=[];n=0
for e in entries:
 trial=batch+[e]
 if len(json.dumps(trial,ensure_ascii=True))>700000 and batch:
  (OUT/f'entries-{n:03d}.json').write_text(json.dumps(batch,ensure_ascii=True),encoding='utf-8');n+=1;batch=[e]
 else:batch=trial
if batch:(OUT/f'entries-{n:03d}.json').write_text(json.dumps(batch,ensure_ascii=True),encoding='utf-8');n+=1
(OUT/'summary.json').write_text(json.dumps(dict(parent=plan['parent'],base_tree=plan['base_tree'],expected_tree=plan['expected_tree'],files=len(paths),blobs=len(blobs),bytes=plan['total_blob_bytes'],entry_parts=n,inline_entries=sum('content' in e for e in entries))),encoding='utf-8')
print((OUT/'summary.json').read_text())
