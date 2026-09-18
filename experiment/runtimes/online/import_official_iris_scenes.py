"""Import author benchmark environment assets into MuJoCo with the existing UR5.

Only static SDF/URDF box/mesh assets and fixed joints used by these sources are
supported. Source collision/visual differences are retained and recorded.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
from copy import deepcopy
import json,hashlib,csv,argparse,time
import numpy as np
from scipy.spatial.transform import Rotation
import mujoco
from model import JOINT_NAMES,set_configuration
ROOT=Path(__file__).resolve().parent
UP=ROOT/'third_party/iris_benchmarks'
DEST=ROOT/'assets/experiment_07_official'
FIG=ROOT/'figures/experiment_07'
MEN=ROOT/'third_party/mujoco_menagerie/universal_robots_ur5e'

def vec(x):return ' '.join(f'{v:.10g}' for v in np.asarray(x).ravel())
def pose(text):
    p=np.fromstring(text or '0 0 0 0 0 0',sep=' ')
    T=np.eye(4);T[:3,3]=p[:3];T[:3,:3]=Rotation.from_euler('xyz',p[3:]).as_matrix();return T

def body(parent,name,T):
    quat=Rotation.from_matrix(T[:3,:3]).as_quat()
    return ET.SubElement(parent,'body',name=name,pos=vec(T[:3,3]),quat=vec(quat[[3,0,1,2]]))

def geom(parent,asset,node,T,name,role,source):
    g=node.find('geometry'); attrs=dict(name=name,pos=vec(T[:3,3]),quat=vec(Rotation.from_matrix(T[:3,:3]).as_quat()[[3,0,1,2]]))
    if role=='visual':attrs.update(contype='0',conaffinity='0',group='2')
    else:attrs.update(contype='1',conaffinity='1',group='3',rgba='.85 .25 .20 .45')
    box=g.find('box'); mesh=g.find('mesh')
    if box is not None:
        size=np.fromstring(box.get('size') or box.findtext('size'),sep=' ')
        attrs.update(type='box',size=vec(size/2))
    elif mesh is not None:
        file=source.parent/mesh.findtext('uri')
        attrs.update(type='mesh',mesh=name+'_mesh')
        ET.SubElement(asset,'mesh',name=name+'_mesh',file=str(file.resolve()))
        # The official table OBJ carries a single texture material.
        mtl=file.with_suffix('.mtl')
        if mtl.exists():
            for line in mtl.read_text().splitlines():
                if line.startswith('map_Kd '):
                    tex=(mtl.parent/line.split(maxsplit=1)[1]).resolve()
                    ET.SubElement(asset,'texture',name=name+'_texture',type='2d',file=str(tex))
                    ET.SubElement(asset,'material',name=name+'_material',texture=name+'_texture',specular='.15',shininess='.15')
                    attrs['material']=name+'_material'
    else:raise ValueError('unsupported official geometry')
    color=node.findtext('material/diffuse')
    c=node.find('material/color')
    if color:attrs['rgba']=color.strip()
    elif c is not None:attrs['rgba']=c.get('rgba')
    elif role=='visual' and 'material' not in attrs:attrs['rgba']='.8 .8 .8 1'
    return ET.SubElement(parent,'geom',**attrs)

def import_sdf(world,asset,path,name,T):
    sdf=ET.parse(path).getroot().find('model')
    parent=body(world,'official_'+name,T)
    for link in sdf.findall('link'):
        lb=body(parent,name+'_'+link.get('name'),pose(link.findtext('pose')))
        for role in ['visual','collision']:
            for i,node in enumerate(link.findall(role)):
                geom(lb,asset,node,pose(node.findtext('pose')),name+'_'+role+'_'+link.get('name')+'_'+str(i),role,path)
    for joint in sdf.findall('joint'):
        if joint.get('type')!='fixed' or joint.find('pose') is not None:raise ValueError('nontrivial SDF joint unsupported')

def import_urdf(world,asset,path,name,T):
    urdf=ET.parse(path).getroot(); transforms={}
    links={l.get('name'):l for l in urdf.findall('link')}
    children={j.find('child').get('link') for j in urdf.findall('joint')}
    for root in set(links)-children:transforms[root]=np.eye(4)
    pending=list(urdf.findall('joint'))
    while pending:
        before=len(pending)
        for j in pending[:]:
            if j.get('type')!='fixed':raise ValueError('dynamic URDF joint unsupported')
            p=j.find('parent').get('link'); c=j.find('child').get('link')
            if p not in transforms:continue
            o=j.find('origin');t=pose((o.get('xyz','0 0 0')+' '+o.get('rpy','0 0 0')) if o is not None else None)
            transforms[c]=transforms[p]@t;pending.remove(j)
        if len(pending)==before:raise ValueError('unresolved URDF tree')
    materials={m.get('name'):m for m in urdf.findall('material')}
    parent=body(world,'official_'+name,T)
    for key,link in links.items():
        lb=body(parent,name+'_'+key,transforms[key])
        for role in ['visual','collision']:
            for i,original in enumerate(link.findall(role)):
                node=deepcopy(original);m=node.find('material')
                if m is not None and m.find('color') is None and m.get('name') in materials:
                    node.remove(m);node.append(deepcopy(materials[m.get('name')]))
                o=node.find('origin');t=pose((o.get('xyz','0 0 0')+' '+o.get('rpy','0 0 0')) if o is not None else None)
                geom(lb,asset,node,t,name+'_'+role+'_'+key+'_'+str(i),role,path)

def base():
    rows=list(csv.DictReader((ROOT/'tables/experiment_07/T27_drawer_formal_runs_07_2.csv').open(encoding='utf-8')))
    run=ROOT/next(r['run'] for r in rows if r['group']=='ET30' and r['seed']=='0')
    tree=ET.parse(run/'scene.xml');root=tree.getroot();world=root.find('worldbody');asset=root.find('asset')
    for g in list(world.findall('geom')):world.remove(g)
    for site in list(world.findall('site')):world.remove(site)
    original=ET.parse(MEN/'ur5e.xml').getroot()
    root.find('compiler').set('meshdir',str((MEN/'assets').resolve()))
    for mesh in original.findall('asset/mesh'):asset.append(deepcopy(mesh))
    for ob in original.findall('.//body'):
        local=root.find(".//body[@name='"+ob.get('name')+"']")
        if local is None:continue
        for g in ob.findall('geom'):
            if g.get('class')=='visual':local.append(deepcopy(g))
    for light in list(world.findall('light')):world.remove(light)
    ET.SubElement(world,'light',name='preview_key',pos='-1 -2 4',dir='.2 .3 -1',diffuse='.8 .8 .8',ambient='.25 .25 .25',castshadow='true')
    ET.SubElement(world,'light',name='preview_fill',pos='2 2 3',dir='-.3 -.3 -1',diffuse='.5 .5 .5',castshadow='false')
    visual=root.find('visual')
    if visual is None:visual=ET.SubElement(root,'visual')
    for old in list(visual):
        if old.tag in ('headlight','global','rgba'):visual.remove(old)
    ET.SubElement(visual,'headlight',ambient='.35 .35 .35',diffuse='.55 .55 .55',specular='.1 .1 .1')
    ET.SubElement(visual,'global',offwidth='1600',offheight='1000')
    ET.SubElement(visual,'rgba',haze='.93 .95 .97 1')
    ET.SubElement(asset,'texture',name='preview_sky',type='skybox',builtin='gradient',rgb1='.86 .91 .97',rgb2='.96 .97 .98',width='512',height='3072')
    q0=np.load(run/'q_history.npy')[0]
    return tree,root,world,asset,q0,run

def build(kind):
    tree,root,world,asset,q0,run=base()
    sources=[];modifications=['UR5 retained; KUKA robots and WSG grippers excluded','UR5 official visual meshes restored; original joint and collision definitions retained','Brighter MuJoCo lighting and floor presentation']
    if kind=='2iiwas_ur5':
        table=UP/'iris_environments/assets_bimanual/models/table/table_wide.sdf'
        verts=np.array([list(map(float,l.split()[1:4])) for l in table.with_suffix('.obj').read_text().splitlines() if l.startswith('v ')])
        lift=-float(np.min(verts[:,2]))
        rb=world.find("body[@name='base']");v=np.fromstring(rb.get('pos','0 0 0'),sep=' ');v[2]+=lift;rb.set('pos',vec(v))
        import_sdf(world,asset,table,'table',pose(f'.4 .3825 {lift} 0 0 0'));sources.append(table)
        rack=UP/'iris_environments/assets_bimanual/models/mug_rack.sdf'
        import_sdf(world,asset,rack,'mug_rack',pose(f'.8 .3825 {lift} 0 0 0'));sources.append(rack)
        ET.SubElement(asset,'material',name='preview_bright_floor',rgba='.86 .88 .90 1',specular='.08')
        ET.SubElement(world,'geom',name='preview_floor',type='plane',size='4 4 .02',material='preview_bright_floor',group='2')
        modifications+=['Entire tabletop/UR5/rack assembly translated upward by '+str(lift)+' m so original table feet reach ground','Added bright ground plane; official 2IIWAs source does not include a separate ground asset']
        warnings=['Official mug_rack back is visual-only; preserved as such. Do not use this preview as a certified cabinet experiment.','Official table collision is a 5 x 5 x 0.2 m slab; table legs are visual mesh only.']
        camera=dict(lookat=[.4,.38,.82],distance=4.15,azimuth=35,elevation=-18)
    else:
        shelf=UP/'iris_environments/assets/shelves1.sdf'
        for i,(x,y) in enumerate([(.48,-.56),(.48,.56),(-.48,-.56),(-.48,.56)]):
            import_sdf(world,asset,shelf,'shelves'+str(i+1),pose(f'{x} {y} .4 0 0 0'))
        sources.append(shelf)
        ground=UP/'iris_environments/assets/ground_big.urdf';lid=UP/'iris_environments/assets/lid.urdf'
        import_urdf(world,asset,ground,'ground',pose('0 0 -.05 0 0 0'))
        import_urdf(world,asset,lid,'lid',pose('0 0 .95 0 0 0'));sources +=[ground,lid]
        for g in world.findall(".//geom"):
            if g.get('name','').startswith('ground_visual'):g.set('rgba','.86 .88 .90 1')
        modifications+=['Only ground color changed; original four shelf poses, dimensions and lid retained']
        warnings=['Preview of source 4Shelves with a 6-DOF UR5; no official KUKA seed trajectory transferred.']
        camera=dict(lookat=[0,0,.42],distance=2.8,azimuth=135,elevation=-22)
    path=DEST/(kind+'.xml');tree.write(path,encoding='utf-8',xml_declaration=True)
    model=mujoco.MjModel.from_xml_path(str(path));data=mujoco.MjData(model);set_configuration(model,data,q0)
    ref=mujoco.MjModel.from_xml_path(str(run/'scene.xml'))
    assert model.nq==6 and model.nv==6
    for name in JOINT_NAMES:
        i=model.joint(name).id;j=ref.joint(name).id
        assert np.array_equal(model.jnt_axis[i],ref.jnt_axis[j])
        assert np.array_equal(model.jnt_range[i],ref.jnt_range[j])
    option=mujoco.MjvOption();option.geomgroup[1]=0;option.geomgroup[3]=0;option.sitegroup[:]=0
    cam=mujoco.MjvCamera();mujoco.mjv_defaultCamera(cam)
    for k,v in camera.items():
        if k=='lookat':cam.lookat[:]=v
        else:setattr(cam,k,v)
    with mujoco.Renderer(model,height=900,width=1440) as renderer:
        renderer.update_scene(data,camera=cam,scene_option=option)
        from PIL import Image
        Image.fromarray(renderer.render()).save(FIG/('F16_official_'+kind+'.png'))
    return dict(scene=kind,mjcf=str(path),q0=q0.tolist(),camera=camera,sources=[str(s.relative_to(UP)) for s in sources],
        modifications=modifications,source_caveats=warnings,nq=model.nq,nv=model.nv,robot_joint_axes_ranges_equal=True,
        source_scene_directory=str(run),formal_experiment=False)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--view',choices=['2iiwas_ur5','4shelves_ur5']);args=parser.parse_args()
    DEST.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)
    if args.view:
        meta=json.loads((DEST/'import_manifest.json').read_text(encoding='utf-8'))
        item=next(r for r in meta['scenes'] if r['scene']==args.view)
        model=mujoco.MjModel.from_xml_path(item['mjcf']);data=mujoco.MjData(model);set_configuration(model,data,np.array(item['q0']))
        from mujoco import viewer as mjviewer
        with mjviewer.launch_passive(model,data) as viewer:
            viewer.opt.geomgroup[1]=0;viewer.opt.geomgroup[3]=0;viewer.opt.sitegroup[:]=0
            for k,v in item['camera'].items():
                if k=='lookat':viewer.cam.lookat[:]=v
                else:setattr(viewer.cam,k,v)
            while viewer.is_running():viewer.sync();time.sleep(.02)
        return
    scenes=[build(k) for k in ['2iiwas_ur5','4shelves_ur5']]
    files=[dict(path=str(p.relative_to(UP)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in UP.rglob('*') if p.is_file() and p.name!='upstream_tree.json']
    manifest=dict(repository='https://github.com/wernerpe/iris_benchmarks',revision=json.loads((UP/'upstream_tree.json').read_text(encoding='utf-8-sig'))['sha'],
        scenes=scenes,files=files,licensing_note='No root license file found in author repository tree; preserve provenance, no new license asserted.')
    (DEST/'import_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({'scenes':[s['mjcf'] for s in scenes],'source_files':len(files)}))
if __name__=='__main__':main()

