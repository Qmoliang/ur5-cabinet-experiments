from pathlib import Path
import sys,json,hashlib,importlib.util
import numpy as np,mujoco
ROOT=Path(__file__).resolve().parent
rt=sys.argv[1];folder=ROOT/'runtimes'/rt
if rt=='online':
    sys.path.insert(0,str(folder));from model import build_model,build_robot_certificate,set_configuration
    from protocol_drawer_scene import formal_drawer_camera_quarter_scene as load_scene
    from depth_camera_perception import UR5MountedDepthCamera
else:
    sys.path[:0]=[str(folder),str(folder/'src'),str(folder/('src/core' if rt=='known' else 'src/common'))]
    from robot import build_model,build_robot_certificate,set_configuration
    if rt=='known':from run import load_scene
    else:from scene import formal_drawer_camera_quarter_scene as load_scene
    camera=ROOT/'runtimes/historical/src/common/camera.py'
    spec=importlib.util.spec_from_file_location('audit_camera',camera);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod);UR5MountedDepthCamera=mod.UR5MountedDepthCamera
scene=load_scene();model=build_model(scene);data=mujoco.MjData(model);set_configuration(model,data,np.array(scene.q0));robot=build_robot_certificate(model)
assert len(scene.boxes)==12 and len(robot)==65
feet=[i for i in range(model.ngeom) if model.geom(i).name.startswith('cabinet_ground_')]
assert len(feet)==4
assert np.all(model.geom_group[feet]==0) and np.all(model.geom_contype[feet]==1) and np.all(model.geom_conaffinity[feet]==1)
old=mujoco.MjModel.from_xml_path(str(ROOT.parent/'02_historical_v43_v44/H01/raw/scene.xml'))
for key in ('body_pos','body_quat','body_mass','body_inertia','jnt_pos','jnt_axis','jnt_range','cam_pos','cam_quat','cam_fovy'):assert np.array_equal(getattr(model,key),getattr(old,key)),key
oldmask=(old.geom_contype!=0)|(old.geom_conaffinity!=0)
newmask=((model.geom_contype!=0)|(model.geom_conaffinity!=0));newmask[feet]=False
for key in ('geom_type','geom_size','geom_pos','geom_quat','geom_bodyid'):assert np.array_equal(getattr(model,key)[newmask],getattr(old,key)[oldmask]),key
observed=0
with UR5MountedDepthCamera(model,camera_names=('ur5_depth_wrist','ur5_depth_forearm'),width=320,height=180,pixel_stride=1,minimum_range=.07,optical_depth_error_bound=.003,occluding_self_filter=True) as cam:
    for obs in cam.capture(data):observed+=int(np.isin(obs.geom_ids,feet).sum())
assert observed>0,'New physical supports must be observed at the initial camera pose'
report=dict(runtime=rt,boxes=12,robot_spheres=65,initial_foot_depth_returns=observed,foot_camera_and_collision_enabled=True,old_panels_robot_and_cameras_equal=True,initial_penetrating_contacts=sum(data.contact[i].dist < -1e-8 for i in range(data.ncon)))
assert report['initial_penetrating_contacts']==0
if rt=='known':
    from known_volume import build_volume_cover,coverage_audit
    for kind in ('sphere','ellipsoid'):
        c=build_volume_cover(scene.boxes,kind,.075);a=coverage_audit(c);assert a['complete_solid_cell_volume_covered'];assert len(c.centers)>368
        report[kind+'_proxy_count']=len(c.centers);report[kind+'_coverage']=a
(ROOT/'preflight').mkdir(exist_ok=True)
(ROOT/'preflight'/f'{rt}.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report),flush=True)
