"""Local provenance only; no runtime reads from another experiment directory."""
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def source_hashes():
    files = list((ROOT / 'src').rglob('*.py')) + list((ROOT / 'configs').glob('*.json'))
    files += [ROOT / 'run.py', ROOT / 'assets/drawer.xml', ROOT / 'assets/scene.json',
              ROOT / 'native/collision.cpp', ROOT / 'native/collision.dll']
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(files)}


def forbid_external_experiment_reads():
    workspace = ROOT.parent.resolve()
    forbidden = ROOT / 'baselines'
    def hook(event, args):
        if event not in ('open', 'ctypes.dlopen') or not args or not isinstance(args[0], (str,bytes,Path)):
            return
        path = Path(args[0]).resolve()
        if path.is_relative_to(forbidden) or (path.is_relative_to(workspace) and not path.is_relative_to(ROOT)):
            raise PermissionError('Online control may not read reference results or another experiment: '+str(path))
    sys.addaudithook(hook)
