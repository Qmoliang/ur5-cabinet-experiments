"""Prepare a public snapshot without modifying the sealed experiment."""
from pathlib import Path
import json, shutil, hashlib
ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path(r'D:\MuJoCo\Cabinet_Experiments\rerun_grounded_20260917')
def copy(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(src,dst)
def write(rel,text):
    p=ROOT/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text,encoding='utf-8')
frozen=json.loads((SOURCE/'formal_runtime_manifest.json').read_text(encoding='utf-8'))
excluded='runtimes/known/external_neo/vendor/quadprog.cp311-win_amd64.pyd'
for rel,digest in frozen.items():
    if rel==excluded:continue
    src=SOURCE/rel
    assert hashlib.sha256(src.read_bytes()).hexdigest()==digest, rel
    copy(src,ROOT/'experiment'/rel)
shutil.copytree(SOURCE/'assets',ROOT/'experiment/assets',dirs_exist_ok=True)
for name in ['batch.py','run_case.py','prepare.py','preflight.py','audit_results.py','formal_runtime_manifest.json','runtime_manifest.json','source_changes.json','catalog.json','comparison.json']:
    copy(SOURCE/name,ROOT/'experiment'/name)
shutil.copytree(SOURCE/'plan',ROOT/'experiment/plan',dirs_exist_ok=True)
shutil.copytree(SOURCE/'audits',ROOT/'reports/audits',dirs_exist_ok=True)
overview=(SOURCE/'README.md').read_text(encoding='utf-8')
notice='# Archived experiment overview\n\nThis document describes the complete original Windows experiment directory. The public repository is a compact snapshot: large pair-state tables, full proxy histories, local launcher files, and workstation outputs are omitted. Paths and links below refer to that original archive; use the repository README for the public layout and supported commands.\n\n---\n\n'
write('reports/experiment-overview.md',notice+overview)
write('docs/assets/experiment-overview.md',notice+overview)
cat=json.loads((SOURCE/'catalog.json').read_text(encoding='utf-8'))
index=[]
for c in cat['cases']:
    dst=ROOT/'evidence'/c['id']
    for name in ['replay_q.npy','replay_error_mm.npy']:
        copy(SOURCE/c['directory']/name,dst/name)
    meta={k:c[k] for k in ['id','group','method','representation','complete','final_error_mm','confirmed_time_s','duration_s','states','proxy_mode']}
    write('evidence/'+c['id']+'/result.json',json.dumps(meta,indent=2))
    index.append(meta)
write('evidence/index.json',json.dumps(index,indent=2))
digests={}
for folder in ['experiment','evidence']:
    for p in sorted((ROOT/folder).rglob('*')):
        if p.is_file():digests[p.relative_to(ROOT).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
write('reports/public-file-manifest.json',json.dumps(digests,indent=2))
write('docs/.nojekyll','')
print('Frozen runtime files:',len(frozen))
print('Evidence cases:',len(index))
print('Manifest files:',len(digests))

