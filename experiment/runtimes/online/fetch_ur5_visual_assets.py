"""Fetch only the UR5 visual assets referenced by the existing local MJCF."""
from pathlib import Path
import urllib.request,json,concurrent.futures,hashlib,xml.etree.ElementTree as ET
root=Path(__file__).resolve().parent/'third_party/mujoco_menagerie/universal_robots_ur5e'
headers={'User-Agent':'experiment07-asset-import'}
def get(url):
    with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=45) as r:return r.read()
sha=json.loads(get('https://api.github.com/repos/google-deepmind/mujoco_menagerie/commits/main'))['sha']
files=['universal_robots_ur5e/assets/'+n.get('file') for n in ET.parse(root/'ur5e.xml').findall('.//asset/mesh')]
files+=['universal_robots_ur5e/LICENSE']
def fetch(path):
    dest=root/Path(path).relative_to('universal_robots_ur5e')
    dest.parent.mkdir(parents=True,exist_ok=True)
    data=get('https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/'+sha+'/'+path)
    dest.write_bytes(data)
    return {'path':path,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex: items=list(ex.map(fetch,files))
(root/'visual_asset_import_manifest.json').write_text(json.dumps({'repository':'google-deepmind/mujoco_menagerie','revision':sha,'purpose':'visual mesh restoration only; existing UR5 MJCF kinematics unchanged','files':items},indent=2),encoding='utf-8')
print(json.dumps({'files':len(items),'bytes':sum(x['bytes'] for x in items),'revision':sha}))
