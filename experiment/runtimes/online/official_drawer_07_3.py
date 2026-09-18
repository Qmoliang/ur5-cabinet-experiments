"""Adapt author shelf panel assets to the frozen drawer task (not original 4Shelves)."""
from pathlib import Path
from dataclasses import replace
from copy import deepcopy
import xml.etree.ElementTree as ET
import numpy as np
from model import BoxObstacle
from experiment_07_scenes import experiment_07_drawer_scene
ROOT=Path(__file__).resolve().parent

def scene():
    old=experiment_07_drawer_scene()
    legs=[]
    for b in old.boxes:
        if b.name.startswith('cabinet_ground_'):
            legs.append(BoxObstacle(b.name+'_front',(.375,b.center[1],b.center[2]),b.half_size,b.material))
    return replace(old,name='experiment_07_drawer_official',boxes=old.boxes+tuple(legs),description='07.3 official shelves1 panel adaptation; frozen drawer interior; four grounded supports')

def decorate(root):
    men=ROOT/'third_party/mujoco_menagerie/universal_robots_ur5e'
    source=ET.parse(ROOT/'third_party/iris_benchmarks/iris_environments/assets/shelves1.sdf').getroot()
    asset=root.find('asset'); world=root.find('worldbody')
    # Consume official panel geometry/material templates; preserve the task dimensions.
    template=source.find(".//visual[@name='bottom']")
    source_size=np.fromstring(template.findtext('geometry/box/size'),sep=' ')
    assert np.all(source_size>0)
    color=np.fromstring(template.findtext('material/diffuse'),sep=' ');color[3]=1.0
    asset.find("material[@name='experiment_07_cabinet']").set('rgba',' '.join(map(str,color)))
    for g in world.findall('geom'):
        if g.get('name','').startswith(('drawer_','cabinet_')):
            assert g.get('type')=='box'
            adapted=np.fromstring(g.get('size'),sep=' ')*2/source_size
            assert np.allclose(source_size*adapted/2,np.fromstring(g.get('size'),sep=' '),rtol=0,atol=1e-15)
            # Keep original decimal text to avoid altering collision geometry by roundoff.
    original=ET.parse(men/'ur5e.xml').getroot()
    root.find('compiler').set('meshdir',str(men/'assets'))
    for mesh in original.findall('asset/mesh'):asset.append(deepcopy(mesh))
    for ob in original.findall('.//body'):
        local=root.find(".//body[@name='"+ob.get('name')+"']")
        if local is not None:
            for g in ob.findall('geom'):
                if g.get('class')=='visual':
                    visual=deepcopy(g);visual.set('group','5');local.append(visual)
    asset.find("material[@name='floor']").set('rgba','.86 .88 .90 1')
    for light in list(world.findall('light')):world.remove(light)
    ET.SubElement(world,'light',name='official_key',pos='-1 -2 4',dir='.2 .3 -1',diffuse='.8 .8 .8',ambient='.25 .25 .25')
    ET.SubElement(world,'light',name='official_fill',pos='2 2 3',dir='-.3 -.3 -1',diffuse='.5 .5 .5',castshadow='false')
    vis=root.find('visual')
    h=vis.find('headlight')
    if h is None:h=ET.SubElement(vis,'headlight')
    h.set('ambient','.35 .35 .35');h.set('diffuse','.55 .55 .55')
    vis.find('global').set('offwidth','1440');vis.find('global').set('offheight','900')
    ET.SubElement(asset,'texture',name='official_sky',type='skybox',builtin='gradient',rgb1='.86 .91 .97',rgb2='.96 .97 .98',width='512',height='3072')
    return root
