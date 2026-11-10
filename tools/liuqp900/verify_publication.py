"""Validate public data and build its manifest from canonical Git-staged bytes."""
from pathlib import Path,PurePosixPath
from collections import Counter
from html.parser import HTMLParser
import subprocess,json,gzip,hashlib,zipfile,io,re,urllib.parse
W=Path(__file__).resolve().parents[2]
def git(*args):return subprocess.check_output(['git',*args],cwd=W)
entries={}
for line in git('ls-files','-s','-z').decode('utf-8').split('\0'):
 if not line:continue
 info,p=line.split('\t',1);entries[p]=info.split(' ')[1]
cache={}
def content(p,keep=False):
 sha=entries[p]
 if sha not in cache:
  b=git('cat-file','blob',sha)
  if keep:cache[sha]=b
  return b
 return cache[sha]
def read(p):return json.loads(content(p).decode('utf-8-sig'))
def sha(b):return hashlib.sha256(b).hexdigest()
base='docs/liuqp900/';data=read(base+'data.json');cases=data['cases'];assert len(cases)==900 and len({c['id'] for c in cases})==900 and len(data['conditions'])==300
assert {a:sum(c['strict_arrival'] for c in cases if c['arm']==a) for a in ['C','A','B']}=={'C':128,'A':186,'B':294}
assert {a:sum(c['physical_safe_arrival'] for c in cases if c['arm']==a) for a in ['C','A','B']}=={'C':174,'A':186,'B':294}
assert not any('\ufffd' in t['label'] for t in data['tasks'])
for c in cases:
 for field in ['trajectory','detail','visual','scene','scene_source','raw_evidence']:assert base+c[field] in entries,(c['id'],field)
 assert c['frames']==c['executed_cycles']+1 and (c['frames']-1)*.02==c['simulation_s']
 assert c['strict_arrival']<=c['physical_safe_arrival']<=c['goal_reached']
# Verify every public replay payload, offsets, and all original evidence member hashes.
groups={}
for c in cases:groups.setdefault(c['trajectory'],[]).append(c)
for path,cc in groups.items():
 raw=gzip.decompress(content(base+path))
 for c in cc:
  offset=c['trajectory_offset_frames']*72;b=raw[offset:offset+c['frames']*72];assert len(b)==c['frames']*72 and sha(b)==c['payload_sha256'],c['id']
raw_bundles=read(base+'evidence/raw-bundles.json');raw_members=0;details_cache={}
for bundle in raw_bundles:
 b=content(base+bundle['file']);assert len(b)==bundle['bytes'] and sha(b)==bundle['sha256']
 with zipfile.ZipFile(io.BytesIO(b)) as z:
  assert z.testzip() is None
  for ident in bundle['runs']:
   c=next(x for x in cases if x['id']==ident);assert c['raw_evidence']==bundle['file']
   if c['detail'] not in details_cache:details_cache[c['detail']]=json.loads(gzip.decompress(content(base+c['detail'])))
   rec=details_cache[c['detail']][ident]
   for item in rec['artifact_files']:
    raw=z.read(item['archive_member']);assert len(raw)==item['bytes'] and sha(raw)==item['sha256'],(ident,item['name']);raw_members+=1
   assert rec['validation']['strict_arrival']==c['strict_arrival']
# Complete frozen source parts and directly browsable code must agree with their hashes.
sources=read(base+'evidence/source-manifest.json');source_members=0
for part in sources['parts']:
 b=content(base+part['file']);assert sha(b)==part['sha256']
 with zipfile.ZipFile(io.BytesIO(b)) as z:
  assert z.testzip() is None
  for f in [x for x in sources['files'] if x['bundle']==part['file']]:
   assert sha(z.read(f['path']))==f['sha256'];source_members+=1
for f in sources['files']:
 p='experiment/liuqp900/'+f['path']
 if p in entries:assert sha(content(p))==f['sha256'],p
# No historical numeric evidence/model/replay changed.
protected=['docs/closeout/data.json','docs/closeout/packs','docs/closeout/models','docs/closeout/evidence','docs/closeout/study/artifact-manifest.json']
assert not git('diff','--cached','--name-only','HEAD','--',*protected).strip()
assert not git('diff','--cached','--name-only','--diff-filter=D').strip()
# New document hashes and equation payloads agree with the source conversion record.
manifest=read('docs/library/manifest.json')
for d in manifest['documents']:assert sha(content('docs/library/'+d['file']))==d['sha256'],d['id']
for d in read(base+'evidence/markdown-provenance.json'):assert sha(content('docs/library/files/liuqp900/'+d['file']))==d['derived_markdown_sha256']
for md,tex in [('methods.md','methods_body.tex'),('proofs.md','proofs.tex')]:
 t=content(base+'evidence/methods/'+tex).decode('utf-8-sig').replace('\r\n','\n').replace('\r','\n');m=content('docs/library/files/liuqp900/'+md).decode('utf-8-sig').replace('\r\n','\n').replace('\r','\n');src=re.findall(r'\\begin\{equation\}(.*?)\\end\{equation\}|\\\[(.*?)\\\]',t,re.S);src=[(a or b).strip() for a,b in src];dst=[x.strip() for x in re.findall(r'^\$\$\s*\n(.*?)^\$\$\s*$',m,re.M|re.S)];assert src==dst,(md,len(src),len(dst))
# Check all changed HTML local links, including absolute project URLs.
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):
  self.links.extend(v for k,v in attrs if k in ['href','src'] and v)
broken=[];checked=0
changed=git('diff','--cached','--name-only').decode().splitlines()
for p in changed:
 if not p.endswith('.html') or not p.startswith('docs/'):continue
 h=Links();h.feed(content(p).decode('utf-8-sig'))
 for link in h.links:
  u=urllib.parse.urlsplit(link)
  if not u.path or link.startswith(('data:','mailto:')):continue
  if u.netloc=='qmoliang.github.io' and u.path.startswith('/ur5-cabinet-experiments/'):target='docs/'+u.path.removeprefix('/ur5-cabinet-experiments/')
  elif u.netloc=='github.com' and u.path.startswith('/Qmoliang/ur5-cabinet-experiments/blob/main/'):target=u.path.removeprefix('/Qmoliang/ur5-cabinet-experiments/blob/main/')
  elif u.scheme or u.netloc:continue
  else:target=str(PurePosixPath(p).parent/urllib.parse.unquote(u.path))
  target=__import__('posixpath').normpath(urllib.parse.unquote(target))
  if link.split('?',1)[0].split('#',1)[0].endswith('/') or target=='docs':target+='/index.html'
  checked+=1
  if target not in entries:broken.append(dict(source=p,link=link,resolved=target))
assert not broken,broken[:12]
for name in ['app.js','replay.js']:assert not re.search(rb'\.mj_step\s*\(',content(base+name))
report=dict(passed=True,runs=900,conditions=300,all_replay_payloads_verified=True,total_saved_states=sum(c['frames'] for c in cases),raw_evidence_members_verified=raw_members,source_members_verified=source_members,historical_numeric_evidence_unchanged=True,no_files_deleted=True,checked_html_links=checked,broken_links=[],markdown_math_preserved=True,no_controller_or_physics_rerun=True,export_validation=read(base+'evidence/export-validation.json'),browser_validation=read(base+'evidence/browser-all900.json'))
# Compact browser report here; full per-run browser receipts remain separately published.
report['browser_validation']={k:v for k,v in report['browser_validation'].items() if k!='records'}
p=W/base/'evidence/publication-validation.json';p.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8',newline='\n')
assets={p.removeprefix(base):dict(bytes=len(content(p)),sha256=sha(content(p)),git_blob_sha=entries[p]) for p in entries if p.startswith(base) and p!=base+'asset-manifest.json' and p!=base+'evidence/publication-validation.json'}
assets['evidence/publication-validation.json']=dict(bytes=p.stat().st_size,sha256=sha(p.read_bytes()))
(W/base/'asset-manifest.json').write_text(json.dumps(assets,ensure_ascii=False,indent=2),encoding='utf-8',newline='\n');print(json.dumps({k:v for k,v in report.items() if k not in ['export_validation','browser_validation']},ensure_ascii=False));print('New public bytes',sum(x['bytes'] for x in assets.values()))
