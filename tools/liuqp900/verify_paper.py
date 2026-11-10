"""Read-only checks of staged paper provenance, source copies and local links."""
from pathlib import Path, PurePosixPath
from html.parser import HTMLParser
import hashlib
import json
import posixpath
import re
import subprocess
import urllib.parse

ROOT=Path(__file__).resolve().parents[2]
BASE='0f402010bb8335df661fdace8745106b7ee5aede'
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)
entries={}
for row in git('ls-files','-s','-z').decode().split('\0'):
    if row:
        meta,path=row.split('\t',1);entries[path]=meta.split()[1]
def read(path):return git('cat-file','blob',entries[path])
def js(path):return json.loads(read(path))
changed=[p for p in git('diff','--cached','--name-only','-z',BASE).decode().split('\0') if p]
protected=[]
old=json.loads(git('show',BASE+':docs/library/manifest.json'))
new=js('docs/library/manifest.json')
for item in old['documents']:
    assert next(x for x in new['documents'] if x['id']==item['id']) == item,item['id']
    path='docs/library/'+item['file']
    assert read(path)==git('show',BASE+':'+path),path
for path in changed:
    if path.startswith('experiment/') or path.startswith(('docs/liuqp900/packs/','docs/liuqp900/evidence/','docs/liuqp900/models/','docs/liuqp900/visuals/','docs/closeout/')) or path in ['docs/liuqp900/data.json','docs/liuqp900/app.js','docs/liuqp900/replay.js','docs/liuqp900/recordings.js']:
        protected.append(path)
assert not protected,protected
assert not git('diff','--cached','--diff-filter=D','--name-only',BASE)
folder='docs/library/files/liuqp900/'
e=js(folder+'paper-evidence.json')
assert e['input_data_sha256']==hashlib.sha256(git('show',BASE+':docs/liuqp900/data.json')).hexdigest()
assert e['n_runs']==900 and e['n_conditions']==300
assert len(e['failures_B'])==6
manifest=js(folder+'history-evidence/manifest.json')
counts={}
for item in manifest['files']:
    b=read(folder+'history-evidence/'+item['file'])
    assert len(b)==item['bytes'] and hashlib.sha256(b).hexdigest()==item['sha256'],item['file']
    if item['study'].startswith('H') and item['kind']=='original_results':
        d=json.loads(b)
        if item['study']=='H01':n=2*len(d['pairs'])
        elif item['study']=='H02':n=len(d['all_runs'])
        else:n=len(d['records'])
        assert n==item['source_table_entries'],item['file']
        counts[item['study']]=n
assert len(counts)==12 and len(manifest['files'])==29
text=read(folder+'explanation-under-2000.md').decode('utf-8-sig')
length=len(re.sub(r'\s','',text));assert length<=2000,length
paper=read(folder+'paper-draft.md').decode('utf-8-sig')
assert '<!-- TABLE' not in paper and '\ufffd' not in paper
for phrase in ['全部B组失败','新增59','新增109','1—2 mm','无条件','全新独立测试集']:
    assert phrase in paper,phrase
tables=json.loads((ROOT/'.local/paper-tables.json').read_text(encoding='utf-8'))
for key in ['outcomes','paired','timing','failures']:assert tables[key] in paper,key
for doc in new['documents'][:4]:
    b=read('docs/library/'+doc['file']);assert hashlib.sha256(b).hexdigest()==doc['sha256']
class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ['href','src'] and v)
broken=[];checked=0
for path in changed:
    if not path.endswith('.html'):continue
    h=Links();h.feed(read(path).decode('utf-8-sig'))
    for link in h.links:
        u=urllib.parse.urlsplit(link)
        if not u.path or link.startswith(('data:','mailto:')):continue
        if u.netloc=='qmoliang.github.io' and u.path.startswith('/ur5-cabinet-experiments/'):
            target='docs/'+u.path.removeprefix('/ur5-cabinet-experiments/')
        elif u.netloc=='github.com' and u.path.startswith('/Qmoliang/ur5-cabinet-experiments/blob/main/'):
            target=u.path.removeprefix('/Qmoliang/ur5-cabinet-experiments/blob/main/')
        elif u.scheme or u.netloc:continue
        else:target=str(PurePosixPath(path).parent/urllib.parse.unquote(u.path))
        target=posixpath.normpath(urllib.parse.unquote(target))
        if u.path.endswith('/') or target=='docs':target+='/index.html'
        checked+=1
        if target not in entries:broken.append(dict(page=path,link=link,target=target))
assert not broken,broken
report=dict(passed=True,frozen_commit=BASE,main_runs=900,matched_conditions=300,historical_source_files=29,history_table_counts=counts,
            historical_counts_are_not_independent_samples=True,original_documents_unchanged=len(old['documents']),
            controller_inputs_results_and_replays_unchanged=True,explanation_nonwhitespace_characters=length,
            checked_links=checked,broken_links=[],new_simulations=0)
(ROOT/'.local/paper-verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False))
