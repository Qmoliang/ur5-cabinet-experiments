"""Verify imported previews preserve the original UR5 kinematics and collisions."""
import json
from pathlib import Path
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parent
folder=ROOT/'assets/experiment_07_official'
m=json.loads((folder/'import_manifest.json').read_text(encoding='utf-8'))
results=[]
for item in m['scenes']:
    new=mujoco.MjModel.from_xml_path(item['mjcf']);old=mujoco.MjModel.from_xml_path(str(Path(item['source_scene_directory'])/'scene.xml'))
    count=0
    for ob in range(1,old.nbody):
        name=old.body(ob).name;nb=new.body(name).id
        for attr in ['body_quat','body_ipos','body_iquat','body_inertia','body_mass']:
            assert np.allclose(getattr(old,attr)[ob],getattr(new,attr)[nb],rtol=0,atol=1e-12),(name,attr)
        if name!='base':assert np.allclose(old.body_pos[ob],new.body_pos[nb],rtol=0,atol=1e-12)
        oldg=[i for i in range(old.ngeom) if old.geom_bodyid[i]==ob and old.geom_contype[i]!=0]
        newg=[i for i in range(new.ngeom) if new.geom_bodyid[i]==nb and new.geom_contype[i]!=0]
        assert len(oldg)==len(newg)
        for i,j in zip(oldg,newg):
            for attr in ['geom_type','geom_size','geom_pos','geom_quat','geom_contype','geom_conaffinity']:
                assert np.allclose(getattr(old,attr)[i],getattr(new,attr)[j],rtol=0,atol=1e-12),(name,attr)
            count+=1
    results.append(dict(scene=item['scene'],robot_bodies_checked=old.nbody-1,collision_geoms_checked=count,passed=True))
(folder/'validation.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
print(json.dumps(results))
