"""Render existing saved q histories. This never runs an experiment or mj_step."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import argparse,json
import numpy as np
import mujoco,imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 index=json.loads((a.output/'data.json').read_text());font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
 for c in index['cases']:
  if not c['video']:continue
  dest=a.output/c['video']
  if dest.exists() and dest.stat().st_size>1000:continue
  base=a.source/('dev/v5' if c['mode']=='known' else 'closeout/final_r17')/'runs'/c['id'];qfile=next(base.rglob('q_history.npy'));q=np.load(qfile);ee=np.load(qfile.with_name('ee_history.npy'))
  model=mujoco.MjModel.from_xml_path(str(qfile.with_name('scene.xml')));data=mujoco.MjData(model)
  model.vis.global_.offwidth=640;model.vis.global_.offheight=400
  cam=mujoco.MjvCamera();cam.lookat[:]=[c['target'][0]*.6,c['target'][1]*.6,.42];cam.distance=1.9;cam.azimuth=140;cam.elevation=-22
  opt=mujoco.MjvOption();opt.geomgroup[3]=0;opt.sitegroup[:]=0
  frames=list(range(0,len(q),30))
  if frames[-1]!=len(q)-1:frames.append(len(q)-1)
  with mujoco.Renderer(model,height=400,width=640) as renderer:
   with imageio.get_writer(dest,fps=10,codec='libx264',quality=6,macro_block_size=1,ffmpeg_params=['-movflags','+faststart','-pix_fmt','yuv420p']) as writer:
    for n,i in enumerate(frames):
     data.qpos[:6]=q[i];mujoco.mj_forward(model,data);renderer.update_scene(data,camera=cam,scene_option=opt)
     # The recorded task target, added for review only.
     sc=renderer.scene;g=sc.geoms[sc.ngeom];mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.012,.012,.012]),np.array(c['target']),np.eye(3).ravel(),np.array([.9,.45,.08,1]));sc.ngeom+=1
     im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im);draw.rectangle((0,365,640,400),fill=(245,245,238));error=np.linalg.norm(ee[i]-c['target'])*1000
     draw.text((10,369),f"{c['id']} | {i*.02:.2f}s | {error:.3f}mm | saved replay 6x",font=font,fill=(38,53,45));writer.append_data(np.asarray(im))
     if n==0:im.save(dest.with_suffix('.webp'),quality=85)
  print('rendered',c['id'],len(frames),dest.stat().st_size,flush=True)

if __name__=='__main__':main()
