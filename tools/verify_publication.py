"""Verify copied evidence and static links; does not run or modify an experiment."""
from pathlib import Path
from urllib.parse import unquote
import hashlib,json,re
ROOT=Path(__file__).resolve().parents[1]
def main():
    manifest=json.loads((ROOT/'reports/public-file-manifest.json').read_text())
    for rel,digest in manifest.items():
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==digest,rel
    frozen=json.loads((ROOT/'experiment/formal_runtime_manifest.json').read_text())
    exclusions=json.loads((ROOT/'reports/public-omissions.json').read_text())['formal_runtime_exclusions']
    for rel,digest in frozen.items():
        if rel in exclusions:continue
        assert hashlib.sha256((ROOT/'experiment'/rel).read_bytes()).hexdigest()==digest,rel
    data=json.loads((ROOT/'docs/data/experiments.json').read_text())
    assert len(data['cases'])==19
    assert sum(not c['complete'] for c in data['cases'])==4
    for c in data['cases']:
        d=ROOT/'evidence'/c['id']
        meta=json.loads((d/'result.json').read_text())
        assert abs(meta['final_error_mm']-c['final_error_mm'])<1e-10
        assert (d/'replay_q.npy').exists() and (d/'replay_error_mm.npy').exists()
        if c['demo']:
            for mode in ['real','cert']:
                for ext in ['mp4','webp']:assert (ROOT/f"docs/assets/{c['id']}-{mode}.{ext}").exists()
    html=(ROOT/'docs/index.html').read_text(encoding='utf-8')
    for link in re.findall(r'(?:src|href)="([^"]+)"',html):
        if link.startswith(('#','http')):continue
        assert (ROOT/'docs'/unquote(link.split('#')[0])).exists(),link
    files=[p for p in ROOT.rglob('*') if p.is_file() and not any(x in p.relative_to(ROOT).parts for x in ['.git','verification','__pycache__'])]
    largest=max(files,key=lambda p:p.stat().st_size)
    assert largest.stat().st_size<100*1024**2
    # Report only filenames if an accidentally embedded credential signature is found.
    suspicious=[]
    pattern=re.compile(r'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)')
    for p in files:
        if p.suffix.lower() in ['.py','.js','.json','.md','.yml','.yaml','.txt','.xml','.toml','.ini','.cfg']:
            if pattern.search(p.read_text(encoding='utf-8',errors='ignore')):suspicious.append(p.relative_to(ROOT).as_posix())
    assert not suspicious,suspicious
    result={'manifest_files':len(manifest),'frozen_files_checked':len(frozen)-len(exclusions),'cases':19,'aborted':4,'public_files':len(files),'total_mb':round(sum(p.stat().st_size for p in files)/1024**2,2),'largest_file':largest.relative_to(ROOT).as_posix(),'largest_mb':round(largest.stat().st_size/1024**2,2),'credential_signature_matches':0}
    out=ROOT/'verification';out.mkdir(exist_ok=True)
    (out/'package-check.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))
if __name__=='__main__':main()

