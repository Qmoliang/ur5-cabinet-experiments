from pathlib import Path
import json,shutil,hashlib,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parent;ARCH=ROOT.parent;WORK=ARCH.parent
SC=WORK/'liu_qp_reproduction/ur5_liuqp_iris_scenes'
def tree(a,b):
    shutil.copytree(a,b,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc','.pytest_cache'))
def save(p,s):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s,encoding='utf-8')
def patch(p,old,new):
    s=p.read_text(encoding='utf-8');assert old in s,str(p);save(p,s.replace(old,new,1))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    rt=ROOT/'runtimes';known=rt/'known';hist=rt/'historical';online=rt/'online'
    assert not (ROOT/'runtime_manifest.json').exists(),'Already prepared; do not overwrite frozen runtimes.'
    for source,target in [(ARCH/'sources/known_liuqp',known),(ARCH/'sources/historical_v43_v44',hist)]:tree(source,target)
    for source,target in [(WORK/'LiuQP_known_volume/external_neo',known/'external_neo'),(SC/'external_neo_adaptive',online/'external_neo_adaptive')]:
        target.mkdir(parents=True,exist_ok=True)
        for p in source.glob('*.py'):shutil.copy2(p,target/p.name)
        for d in ('vendor','plan'):
            if (source/d).exists():tree(source/d,target/d)
    for rel in ('sphere_study','ablation_no_manip'):
        source=WORK/'LiuQP_known_volume/external_neo'/rel if rel=='sphere_study' else SC/'external_neo_adaptive'/rel
        target=known/'external_neo'/rel if rel=='sphere_study' else online/'external_neo_adaptive'/rel
        target.mkdir(parents=True,exist_ok=True)
        for p in source.glob('*.py'):shutil.copy2(p,target/p.name)
        tree(source/'plan',target/'plan')
    online.mkdir(parents=True,exist_ok=True)
    for p in SC.iterdir():
        if p.is_file() and p.suffix in ('.py','.bat','.json'):shutil.copy2(p,online/p.name)
    tree(SC/'native_mvt',online/'native_mvt');tree(SC/'plan',online/'plan')
    tree(ARCH/'assets/ur5e',ROOT/'assets/ur5e')
    tree(ARCH/'assets/ur5e',online/'third_party/mujoco_menagerie/universal_robots_ur5e')
    original=json.loads((WORK/'LiuQP_known_volume/assets/scene.json').read_text(encoding='utf-8'))
    root=ET.parse(ARCH/'02_historical_v43_v44/H01/presentation.xml').getroot()
    root.set('model','ur5e_grounded_cabinet_physical_20260917')
    root.find('compiler').set('meshdir',str(ROOT/'assets/ur5e/assets'))
    feet=[]
    for g in root.findall('worldbody/geom'):
        if g.get('name','').startswith('display_foot_'):
            name=g.get('name').replace('display_foot_','cabinet_ground_')
            g.set('name',name);g.set('group','0');g.set('contype','1');g.set('conaffinity','1')
            feet.append(dict(name=name,center=list(map(float,g.get('pos').split())),half_size=list(map(float,g.get('size').split())),material='shelf'))
    assert len(feet)==4;original['boxes']+=feet
    original['description']='Physical grounded cabinet: original 8 task panels plus 4 sensed/collidable supports; original robot/cameras/task.'
    ET.indent(root);xml=ET.tostring(root,encoding='unicode')
    save(ROOT/'assets/scene.xml',xml);save(ROOT/'assets/scene.json',json.dumps(original,indent=2))
    for target in (known,hist,online):
        save(target/'assets/drawer.xml',xml);save(target/'assets/scene.json',json.dumps(original,indent=2))
    patch(online/'model.py','def build_xml(scene: SceneDefinition) -> str:\n','def build_xml(scene: SceneDefinition) -> str:\n    # This isolated rerun uses the single audited physical cabinet XML.\n    return (ROOT / "assets/drawer.xml").read_text(encoding="utf-8")\n')
    save(online/'grounded_scene.py','''from pathlib import Path
import json
from model import BoxObstacle,SceneDefinition
def load_scene():
    x=json.loads((Path(__file__).resolve().parent/'assets/scene.json').read_text(encoding='utf-8'))
    return SceneDefinition(name=x['name'],description=x['description'],boxes=tuple(BoxObstacle(b['name'],tuple(b['center']),tuple(b['half_size']),b['material']) for b in x['boxes']),q0=tuple(x['q0']),waypoints=tuple(tuple(p) for p in x['waypoints']),duration=x['duration'])
''')
    patch(online/'protocol_drawer_scene.py','def formal_drawer_camera_quarter_scene() -> SceneDefinition:\n','def formal_drawer_camera_quarter_scene() -> SceneDefinition:\n    from grounded_scene import load_scene\n    return load_scene()\n')
    patch(online/'external_neo_adaptive/neo_online.py',"BASE.parents[1]/'LiuQP_known_volume/external_neo/vendor'","BASE.parent/'known/external_neo/vendor'")
    # Keep all pre-existing math unchanged; save extra diagnostic matrices on failure.
    patch(online/'external_neo_adaptive/neo_online.py','H,g,A,lo,hi,info=self.assemble(np.asarray(target)); assembled=time.perf_counter()','H,g,A,lo,hi,info=self.assemble(np.asarray(target)); self.failed_qp_candidate=(H,g,A,lo,hi,info); assembled=time.perf_counter()')
    changes=[]
    for a,b in [(ARCH/'sources/known_liuqp',known),(ARCH/'sources/historical_v43_v44',hist),(SC,online)]:
        for p in b.rglob('*.py'):
            ref=a/p.relative_to(b)
            if ref.exists() and sha(ref)!=sha(p):changes.append(dict(file=str(p.relative_to(ROOT)),original=str(ref),original_sha256=sha(ref),new_sha256=sha(p)))
    save(ROOT/'source_changes.json',json.dumps(changes,indent=2))
    files=[p for p in rt.rglob('*') if p.is_file() and p.suffix!='.pyc']
    save(ROOT/'runtime_manifest.json',json.dumps({p.relative_to(ROOT).as_posix():sha(p) for p in files},indent=2))
    print('Prepared',len(files),'runtime files, 12 physical boxes, 4 sensed feet; changed sources',len(changes))
if __name__=='__main__':main()
