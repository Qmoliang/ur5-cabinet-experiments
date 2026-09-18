"""Verify delivered files without depending on any historical project."""
from pathlib import Path
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True


def main():
    manifest=json.loads((ROOT/'MANIFEST.json').read_text(encoding='utf-8'))
    failures=[]
    for name,meta in manifest['files'].items():
        path=ROOT/name
        if not path.is_file():failures.append(name+': missing');continue
        content=path.read_bytes()
        if len(content)!=meta['bytes'] or hashlib.sha256(content).hexdigest()!=meta['sha256']:
            failures.append(name+': changed')
    if failures:
        raise SystemExit('\n'.join(failures))
    print(json.dumps({'verified_files':len(manifest['files']),'all_tracked_files_unchanged':True,
        'baseline_cases':['v43_ellipsoid','v44_sphere'],
        'new_user_runs_allowed':True},indent=2))


if __name__=='__main__':main()
