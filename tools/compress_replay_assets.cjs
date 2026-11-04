// Lossless transport optimization. Physics model, engine version and trajectories stay unchanged.
const fs=require('fs'),zlib=require('zlib'),path=require('path');
const root=path.resolve(__dirname,'../docs/closeout'),assets={};
for(const name of JSON.parse(fs.readFileSync(path.join(root,'models/robot.json'),'utf8')).assets)assets[name]=fs.readFileSync(path.join(root,'models/assets',name),'utf8');
fs.writeFileSync(path.join(root,'models/assets.json.gz'),zlib.gzipSync(JSON.stringify(assets),{level:9}));
fs.writeFileSync(path.join(root,'vendor/mujoco.wasm.gz'),zlib.gzipSync(fs.readFileSync(path.join(root,'vendor/mujoco.wasm')),{level:9}));
