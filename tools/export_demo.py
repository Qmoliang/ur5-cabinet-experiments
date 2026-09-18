"""Export website data and videos from recorded MuJoCo states, never a controller."""
from pathlib import Path
import argparse, json, sys, shutil
import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
NAMES = {
 'K01':'Known LiuQP / spheres', 'K02':'Known LiuQP / ellipsoids',
 'K03':'Known spheres / same-QP solver check',
 'H01':'Historical 4.3 / ellipsoids', 'H02':'Historical 4.4 / adaptive spheres',
 'N01':'Known LiuQP / reference rerun', 'N02':'Known NEO / ellipsoids / 46 mm',
 'N03':'Known NEO / ellipsoids / 300 mm', 'N04':'Known NEO / spheres / 46 mm',
 'N05':'Known NEO / spheres / 300 mm', 'O01':'Online LiuQP / adaptive spheres',
 'O02':'Online LiuQP / ellipsoids', 'O03':'Online NEO / spheres / 300 mm',
 'O04':'Online NEO / ellipsoids / 300 mm', 'O05':'Online NEO / spheres / 46 mm',
 'O06':'Online NEO / ellipsoids / 46 mm', 'A01':'No-manipulability NEO / spheres / 300 mm',
 'A02':'No-manipulability NEO / ellipsoids / 300 mm', 'A03':'No-manipulability NEO / ellipsoids / 46 mm',
}
DEMO_IDS = ['K01','K02','H01','H02','O01','O02','N04','N02','A03']

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('source',type=Path)
    parser.add_argument('--data-only',action='store_true')
    args=parser.parse_args();source=args.source.resolve()
    sys.path.insert(0,str(source));import replay
    assets=ROOT/'docs/assets';data=ROOT/'docs/data'
    assets.mkdir(parents=True,exist_ok=True);data.mkdir(parents=True,exist_ok=True)
    cases=[]
    for entry in replay.CAT['cases']:
        c=replay.load(entry['id']);z=c['proxy'];maps=[]
        if c['proxy_mode']=='causal':
            maps=[{'published':int(pub)*.02,'source':int(src)*.02,'count':int(hi-lo)} for pub,src,lo,hi in zip(z['publish_cycles'],z['source_cycles'],z['offsets'][:-1],z['offsets'][1:])]
        record={k:c[k] for k in ['id','group','representation','complete','final_error_mm','confirmed_time_s','duration_s','proxy_mode']}
        record.update(name=NAMES[c['id']],outcome='Aborted' if not c['complete'] else 'Reached & held' if c['status']=='到达并保持' else 'Not reached',
                      demo=c['id'] in DEMO_IDS,curve=[[round(i*.02,2),round(float(c['error'][i]),6)] for i in range(0,len(c['q']),10)],maps=maps,
                      static_count=len(z['centers']) if c['proxy_mode']=='static' else None)
        if record['curve'][-1][0] != round((len(c['q'])-1)*.02,2):record['curve'].append([round((len(c['q'])-1)*.02,2),float(c['error'][-1])])
        cases.append(record)
    payload={'batch':'2026-09-17 grounded cabinet rerun','dt':.02,'playback_speed':6,'video_fps':20,'cases':cases}
    (data/'experiments.json').write_text(json.dumps(payload,separators=(',',':')),encoding='utf-8')
    if args.data_only:return
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',20)
    for cid in DEMO_IDS:
        c=replay.load(cid);model=replay.model_for(c);d=mujoco.MjData(model)
        cert=replay.build_robot_certificate(model);cam=mujoco.MjvCamera();replay.camera(cam)
        cam.distance=1.8
        cabinet=model.geom_matid==model.material('shelf').id;model.geom_group[cabinet]=4
        for mode in ['real','cert']:
            target=assets/f'{cid}-{mode}.mp4'
            if target.exists():print('Retained',target.name,flush=True);continue
            opt=replay.options();opt.geomgroup[4]=int(mode=='real');opt.geomgroup[5]=int(mode=='real')
            # Use 6x playback consistently: 15 control steps per video frame.
            frames=list(range(0,len(c['q']),15))
            if frames[-1]!=len(c['q'])-1:frames.append(len(c['q'])-1)
            with mujoco.Renderer(model,height=600,width=960,max_geom=30000) as renderer:
                with imageio.get_writer(target,fps=20,codec='libx264',quality=7,macro_block_size=1,ffmpeg_params=['-movflags','+faststart','-pix_fmt','yuv420p']) as writer:
                    for frame in frames:
                        replay.setq(model,d,c['q'][frame]);renderer.update_scene(d,camera=cam,scene_option=opt)
                        if mode=='cert':replay.overlay(renderer.scene,c,frame,d,cert,True,True)
                        im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im)
                        draw.rectangle((0,558,960,600),fill=(242,244,241))
                        draw.text((18,568),f'{cid}  |  t = {frame*.02:05.2f} s  |  error = {c["error"][frame]:.3f} mm  |  recorded replay / 6x',font=font,fill=(34,45,39))
                        writer.append_data(np.asarray(im))
                        if frame==0:im.save(assets/f'{cid}-{mode}.webp',quality=86)
                print('Rendered',target.name,round(target.stat().st_size/1024**2,2),'MB',flush=True)
    # A photographic record of the simulation, not a synthesized scene.
    shutil.copy2(assets/'O02-real.webp',assets/'hero.webp')

if __name__=='__main__':main()
