"""Repack derived replay copies only, preserving every case's payload hash."""
from pathlib import Path
import argparse,json,gzip,hashlib

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();r=a.root.resolve();d=json.loads((r/'data.json').read_text(encoding='utf-8'));groups={};changes=[]
 for c in d['cases']:groups.setdefault(c['trajectory'],[]).append(c)
 def save_group(stem,items):
  b=gzip.compress(b''.join(x[1] for x in items),compresslevel=8,mtime=0)
  if len(b)>6*1024**2 and len(items)>1:
   mid=len(items)//2;save_group(stem,items[:mid]);save_group(stem,items[mid:]);return
  assert len(b)<8*1024**2;name='packs/'+stem+'__s%02d-s%02d.bin.gz'%(items[0][0]['seed'],items[-1][0]['seed']);(r/name).write_bytes(b);offset=0
  for c,payload in items:c['trajectory']=name;c['trajectory_offset_frames']=offset;offset+=c['frames']
  changes.append(dict(file=name,bytes=len(b)))
 for name,cases in groups.items():
  p=r/name
  if p.stat().st_size<=6*1024**2:continue
  raw=gzip.decompress(p.read_bytes());items=[]
  for c in sorted(cases,key=lambda x:x['seed']):
   lo=c['trajectory_offset_frames']*72;b=raw[lo:lo+c['frames']*72];assert hashlib.sha256(b).hexdigest()==c['payload_sha256'];items.append((c,b))
  save_group(p.name.removesuffix('.bin.gz'),items);assert p.resolve().is_relative_to((r/'packs').resolve());p.unlink()
 (r/'data.json').write_text(json.dumps(d,ensure_ascii=False,separators=(',',':')),encoding='utf-8',newline='\n');(r/'evidence/repack-validation.json').write_text(json.dumps(dict(passed=True,recorded_frames_changed=False,new_parts=changes),indent=2),encoding='utf-8')
 print(json.dumps(dict(changed_groups=len(changes),max_pack_bytes=max(p.stat().st_size for p in (r/'packs').glob('*.bin.gz')))))
if __name__=='__main__':main()
