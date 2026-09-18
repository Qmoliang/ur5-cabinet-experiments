"""Replay sphere failures and ellipsoid success, using their actual proxies."""
import argparse, json, queue, time
import numpy as np
import mujoco
import mujoco.viewer
from experiment import HERE, PARENT, ROOT, original, build_model, build_robot_certificate
from experiment import set_configuration, DT
from robot import certificate_world_positions
from replay import camera_setup, display_lighting
from view import append_geom, shape_to_axes_rotation, IDENTITY, ROBOT_RGBA, TARGET_RGBA

SPECS=[('NEO / spheres / influence 300 mm',HERE/'results/neo_sphere_paper','sphere'),
       ('NEO / spheres / influence 46 mm',HERE/'results/neo_sphere_matched','sphere'),
       ('NEO / ellipsoids / influence 300 mm',PARENT/'results/neo_paper_influence','ellipsoid'),
       ('LiuQP / spheres / historical',ROOT/'results/sphere','sphere')]

def load():
    result=[]
    for label,folder,kind in SPECS:
        r=dict(label=label,folder=folder,kind=kind,q=np.load(folder/'q_history.npy'),
               ee=np.load(folder/'ee_history.npy'),proxies=np.load(folder/'proxies.npz'),
               summary=json.loads((folder/'summary.json').read_text(encoding='utf-8')))
        assert r['q'].shape==(3001,6) and r['ee'].shape==(3001,3)
        result.append(r)
    return result

def draw(overlay,data,robot,r,target,show_proxies,show_robot):
    if show_proxies:
        p=r['proxies']
        for i,center in enumerate(p['centers']):
            if r['kind']=='sphere':
                append_geom(overlay,mujoco.mjtGeom.mjGEOM_SPHERE,np.full(3,p['sphere_radii'][i]),
                            center,IDENTITY,np.array([.2,.65,1.,.20]))
            else:
                axes,rotation=shape_to_axes_rotation(p['ellipsoid_shapes'][i])
                append_geom(overlay,mujoco.mjtGeom.mjGEOM_ELLIPSOID,axes,center,rotation,np.array([.2,.65,1.,.20]))
    if show_robot:
        for pos,s in zip(certificate_world_positions(data,robot),robot):
            append_geom(overlay,mujoco.mjtGeom.mjGEOM_SPHERE,np.full(3,s.radius),pos,IDENTITY,ROBOT_RGBA)
    append_geom(overlay,mujoco.mjtGeom.mjGEOM_SPHERE,np.full(3,.008),target,IDENTITY,TARGET_RGBA)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--check',action='store_true'); ap.add_argument('--render',action='store_true')
    args=ap.parse_args(); results=load(); scene=original.load_scene(); target=np.array(scene.waypoints[-1])
    model=build_model(scene); data=mujoco.MjData(model); robot=build_robot_certificate(model)
    display_lighting(model)
    if args.check:
        for r in results:
            for q in r['q'][::250]: set_configuration(model,data,q)
            p=r['proxies']; assert len(p['centers'])==368
        print('PASS: four saved MuJoCo replays load; sphere rendering uses sphere_radii.'); return
    if args.render:
        from PIL import Image,ImageDraw,ImageFont
        model.vis.global_.offwidth=1100; model.vis.global_.offheight=800
        cam=mujoco.MjvCamera(); camera_setup(cam); renderer=mujoco.Renderer(model,height=800,width=1100)
        font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',24)
        try:
            for i,r in enumerate(results[:3]):
                for overlay in [False,True]:
                    set_configuration(model,data,r['q'][-1]); renderer.update_scene(data,camera=cam)
                    draw(renderer.scene,data,robot,r,target,overlay,overlay)
                    im=Image.fromarray(renderer.render()); canvas=ImageDraw.Draw(im)
                    canvas.rectangle((0,0,1100,76),fill=(18,22,30))
                    err=np.linalg.norm(r['ee'][-1]-target)*1000
                    canvas.text((18,8),r['label'],font=font,fill='white')
                    canvas.text((18,42),f't=60 s | error={err:.3f} mm | proxies={overlay}',font=font,fill='white')
                    im.save(HERE/'results'/f'view_{i+1}_{"proxies" if overlay else "physical"}.png')
        finally: renderer.close()
        print('Rendered six actual saved states.'); return
    cmd=queue.Queue()
    def key(k):
        if 0<=k<128: cmd.put(chr(k).upper())
    case=0; frame=0; paused=False; show_proxies=True; show_robot=False; last=time.perf_counter()
    with mujoco.viewer.launch_passive(model,data,key_callback=key) as viewer:
        camera_setup(viewer.cam)
        while viewer.is_running():
            while not cmd.empty():
                c=cmd.get_nowait()
                if c in '1234': case=int(c)-1; frame=0; paused=False
                elif c==' ': paused=not paused
                elif c=='0': frame=0; paused=False
                elif c=='F': frame=3000; paused=True
                elif c=='V': show_proxies=not show_proxies
                elif c=='R': show_robot=not show_robot
                elif c=='S':
                    t=results[case]['summary'].get('confirmed_success_time_s')
                    frame=round(t/DT) if t is not None else 3000; paused=True
            r=results[case]; now=time.perf_counter()
            if not paused and now-last>=DT:
                frame=min(frame+1,3000); last=now
                # Failure is visibly stable by 10 s. Full 60 s remains on F.
                if frame==500 or frame==3000: paused=True
            set_configuration(model,data,r['q'][frame]); viewer.user_scn.ngeom=0
            draw(viewer.user_scn,data,robot,r,target,show_proxies,show_robot)
            err=np.linalg.norm(r['ee'][frame]-target)*1000
            viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150,mujoco.mjtGridPos.mjGRID_TOPLEFT,
                             r['label'],f't={frame*DT:.2f}s | error={err:.3f}mm | '+('PAUSED' if paused else 'PLAY')+'\n'
                             '1 sphere 300 | 2 sphere 46 | 3 ellipsoid 300 | 4 Liu sphere\n'
                             'Space pause | 0 restart | F final | V proxies | R robot spheres'))
            viewer.sync(); time.sleep(.001)

if __name__=='__main__': main()
