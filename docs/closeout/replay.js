import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import loadMujoco from './vendor/mujoco.js';

async function fetchOK(url){const r=await fetch(url);if(!r.ok)throw Error(`${url}: HTTP ${r.status}`);return r;}
function dispose(group){group.traverse(o=>{o.geometry?.dispose();if(o.material){for(const m of Array.isArray(o.material)?o.material:[o.material])m.dispose();}});group.clear();}
export class Replay{
  static async create(host,callbacks){const r=new Replay(host,callbacks);await r.init();return r;}
  constructor(host,callbacks){this.host=host;this.callbacks=callbacks;this.frame=0;this.playing=false;this.speed=6;this.view=1;this.loaded=false;this.cloudIndex=-1;}
  async init(){
    document.getElementById('loading').textContent='正在下载压缩引擎与模型（约7 MB），首次加载请稍候…';
    const [wasm,assets]=await Promise.all([
      fetchOK('vendor/mujoco.wasm.gz').then(async r=>new Uint8Array(await new Response(r.body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer())),
      fetchOK('models/assets.json.gz').then(async r=>JSON.parse(await new Response(r.body.pipeThrough(new DecompressionStream('gzip'))).text()))
    ]);
    document.getElementById('loading').textContent='资源已下载，正在初始化 MuJoCo 与原机器人模型…';
    this.mj=await loadMujoco({wasmBinary:wasm});this.robotInfo=await (await fetchOK('models/robot.json')).json();
    this.mj.FS.mkdir('/replay');this.mj.FS.mkdir('/replay/assets');
    for(const name of this.robotInfo.assets)this.mj.FS.writeFile(`/replay/assets/${name}`,assets[name]);
    this.mj.FS.writeFile('/replay/robot.xml',await (await fetchOK('models/robot.xml')).text());
    this.model=this.mj.MjModel.from_xml_path('/replay/robot.xml');this.data=new this.mj.MjData(this.model);
    this.scene=new THREE.Scene();this.scene.background=new THREE.Color('#e6eae3');
    this.camera=new THREE.PerspectiveCamera(43,1,.01,100);this.camera.up.set(0,0,1);
    this.renderer=new THREE.WebGLRenderer({antialias:true,preserveDrawingBuffer:true});this.renderer.setPixelRatio(Math.min(devicePixelRatio,2));this.renderer.outputColorSpace=THREE.SRGBColorSpace;this.host.appendChild(this.renderer.domElement);
    this.controls=new OrbitControls(this.camera,this.renderer.domElement);this.controls.enableDamping=true;this.controls.minDistance=.25;this.controls.maxDistance=6;
    this.scene.add(new THREE.HemisphereLight(0xffffff,0x71806c,2.2));const light=new THREE.DirectionalLight(0xffffff,3);light.position.set(-1,-2,4);this.scene.add(light);
    this.robot=new THREE.Group();this.cert=new THREE.Group();this.environment=new THREE.Group();this.path=new THREE.Group();this.cloud=new THREE.Group();this.scene.add(this.robot,this.cert,this.environment,this.path,this.cloud);
    const floor=new THREE.Mesh(new THREE.PlaneGeometry(4,4),new THREE.MeshStandardMaterial({color:0xd9ded4,roughness:1}));floor.position.z=-.025;this.scene.add(floor);
    const grid=new THREE.GridHelper(3,30,0xb4bfb1,0xcbd3c6);grid.rotation.x=Math.PI/2;grid.position.z=-.024;this.scene.add(grid);
    this.robotGeoms=[];const m=this.model;
    for(let i=0;i<m.ngeom;i++){
      // Frozen scene uses group 5 for visual meshes and group 1 for sensor housings.
      if(m.geom_group[i]!==5 && !(m.geom_group[i]===1 && m.geom_type[i]===6))continue;
      let geometry;
      const type=m.geom_type[i],s=Array.from(m.geom_size.slice(i*3,i*3+3));
      if(type===7){const id=m.geom_dataid[i],start=m.mesh_vertadr[id]*3,n=m.mesh_vertnum[id]*3,fa=m.mesh_faceadr[id]*3,fn=m.mesh_facenum[id]*3;geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(Array.from(m.mesh_vert.slice(start,start+n)),3));geometry.setIndex(Array.from(m.mesh_face.slice(fa,fa+fn)));geometry.computeVertexNormals();}
      else if(type===2)geometry=new THREE.SphereGeometry(s[0],16,12);
      else if(type===6)geometry=new THREE.BoxGeometry(s[0]*2,s[1]*2,s[2]*2);
      else continue;
      const matid=m.geom_matid[i],rgba=matid>=0?m.mat_rgba.slice(matid*4,matid*4+4):m.geom_rgba.slice(i*4,i*4+4);
      const mesh=new THREE.Mesh(geometry,new THREE.MeshStandardMaterial({color:new THREE.Color(rgba[0],rgba[1],rgba[2]),roughness:.47,metalness:.16}));this.robot.add(mesh);this.robotGeoms.push([i,mesh]);
    }
    for(const s of this.robotInfo.certificate){const mesh=new THREE.Mesh(new THREE.SphereGeometry(s.radius,14,10),new THREE.MeshStandardMaterial({color:0x4c94a7,transparent:true,opacity:.43,depthWrite:false}));mesh.userData=s;this.cert.add(mesh);}
    this.target=new THREE.Mesh(new THREE.SphereGeometry(.012,18,12),new THREE.MeshStandardMaterial({color:0xc77c35,emissive:0x663400,emissiveIntensity:.2}));this.scene.add(this.target);
    this.eeMarker=new THREE.Mesh(new THREE.SphereGeometry(.007,14,10),new THREE.MeshStandardMaterial({color:0x217993}));this.scene.add(this.eeMarker);
    this.rotation=new THREE.Matrix4();this.qrotation=new THREE.Quaternion();
    new ResizeObserver(()=>{const w=this.host.clientWidth,h=this.host.clientHeight;this.renderer.setSize(w,h);this.camera.aspect=w/h;this.camera.updateProjectionMatrix();}).observe(this.host);
    this.last=performance.now();this.resetCamera();this.setView(1);this.animate();
  }
  async load(c,token){
    this.pause();this.loaded=false;this.case=c;document.getElementById('loading').hidden=false;document.getElementById('loading').textContent='加载保存轨迹与地图发布记录…';
    const [packed,detailPack,scene]=await Promise.all([fetchOK(c.trajectory),fetchOK(c.detail).then(async r=>JSON.parse(await new Response(r.body.pipeThrough(new DecompressionStream('gzip'))).text())),fetchOK(`models/${c.task}.json`).then(r=>r.json())]);
    const buffer=await new Response(packed.body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
    if(!this.callbacks.isCurrent(token))return;
    const start=c.trajectory_offset_frames*9;this.values=new Float64Array(buffer).subarray(start,start+c.frames*9);if(this.values.length!==c.frames*9)throw Error('轨迹长度与索引不一致');this.detail=detailPack[c.id];this.sceneInfo=scene;this.frame=0;this.cloudIndex=-1;
    dispose(this.environment);for(const b of scene.boxes){const mesh=new THREE.Mesh(new THREE.BoxGeometry(...b.half_size.map(x=>2*x)),new THREE.MeshStandardMaterial({color:b.material==='obstacle'?0x9b5b4c:0x8f9c8d,transparent:true,opacity:.7,roughness:.85,depthWrite:true}));mesh.position.fromArray(b.center);this.environment.add(mesh);const edges=new THREE.LineSegments(new THREE.EdgesGeometry(mesh.geometry),new THREE.LineBasicMaterial({color:0x637560,transparent:true,opacity:.45}));edges.position.copy(mesh.position);this.environment.add(edges);}
    dispose(this.path);const points=[];for(let i=0;i<c.frames;i++)points.push(...this.values.slice(i*9+6,i*9+9));const geo=new THREE.BufferGeometry();geo.setAttribute('position',new THREE.Float32BufferAttribute(points,3));this.pathLine=new THREE.Line(geo,new THREE.LineBasicMaterial({color:0x237f9a}));this.path.add(this.pathLine);this.target.position.fromArray(c.target);this.loaded=true;this.setFrame(0);this.resetCamera();this.setView(this.view);document.getElementById('loading').hidden=true;
  }
  resetCamera(){const t=this.case?.target||[.5,.25,.5];this.controls.target.set(t[0]*.6,t[1]*.6,.42);this.camera.position.set(-1.2,-1.5,1.25);this.controls.update();}
  pose(mesh,position,matrix){mesh.position.fromArray(position);this.rotation.set(matrix[0],matrix[1],matrix[2],0,matrix[3],matrix[4],matrix[5],0,matrix[6],matrix[7],matrix[8],0,0,0,0,1);mesh.quaternion.setFromRotationMatrix(this.rotation);}
  setFrame(frame){
    if(!this.loaded)return;this.frame=Math.max(0,Math.min(this.case.frames-1,Math.round(frame)));const offset=this.frame*9;this.data.qpos.set(this.values.subarray(offset,offset+6));this.data.qvel.fill(0);this.mj.mj_forward(this.model,this.data);
    for(const [id,mesh] of this.robotGeoms)this.pose(mesh,this.data.geom_xpos.subarray(id*3,id*3+3),this.data.geom_xmat.subarray(id*9,id*9+9));
    for(const mesh of this.cert.children){const s=mesh.userData,b=s.body,r=this.data.xmat.subarray(b*9,b*9+9),p=this.data.xpos.subarray(b*3,b*3+3),l=s.center;mesh.position.set(p[0]+r[0]*l[0]+r[1]*l[1]+r[2]*l[2],p[1]+r[3]*l[0]+r[4]*l[1]+r[5]*l[2],p[2]+r[6]*l[0]+r[7]*l[1]+r[8]*l[2]);}
    const ee=Array.from(this.values.subarray(offset+6,offset+9));this.eeMarker.position.fromArray(ee);this.pathLine.geometry.setDrawRange(0,this.frame+1);
    const error=Math.hypot(...ee.map((x,i)=>x-this.case.target[i]))*1000;this.callbacks.onTime(this.frame,error);
    let ci=-1;for(let j=0;j<this.detail.cloud_frames.length;j++)if(this.detail.cloud_frames[j].cycle<=this.frame)ci=j;
    if(ci!==this.cloudIndex){this.cloudIndex=ci;dispose(this.cloud);if(ci>=0){const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(this.detail.cloud_frames[ci].points.flat(),3));this.cloud.add(new THREE.Points(g,new THREE.PointsMaterial({color:0x36836f,size:.006,sizeAttenuation:true})));}}
  }
  setView(n){this.view=n;this.robot.visible=n!==2;this.cert.visible=n===2;this.environment.visible=n!==4;this.path.visible=n===3;this.cloud.visible=n===4;this.eeMarker.visible=n===3;}
  pause(){this.playing=false;this.callbacks.onPause();}
  toggle(){if(!this.loaded)return;if(this.frame>=this.case.frames-1)this.setFrame(0);this.playing=!this.playing;this.playhead=this.frame;this.last=performance.now();}
  animate(){requestAnimationFrame(()=>this.animate());const now=performance.now(),delta=Math.min(.1,(now-this.last)/1000);this.last=now;if(this.playing&&this.loaded){this.playhead+=delta*this.speed/.02;this.setFrame(this.playhead);if(this.frame>=this.case.frames-1)this.pause();}this.controls.update();this.renderer.render(this.scene,this.camera);}
  verifyPose(q){this.data.qpos.set(q);this.mj.mj_forward(this.model,this.data);return Array.from(this.data.site_xpos.subarray(this.robotInfo.site_id*3,this.robotInfo.site_id*3+3));}
}
