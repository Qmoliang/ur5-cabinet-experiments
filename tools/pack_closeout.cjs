// Combine 20 cases per task/cohort into lossless gzip packs for publication.
const fs=require('fs'),path=require('path'),zlib=require('zlib'),crypto=require('crypto');
const root=path.resolve(__dirname,'../docs/closeout'),index=JSON.parse(fs.readFileSync(path.join(root,'data.json'),'utf8'));
const groups=new Map();for(const c of index.cases){const k=c.mode+'__'+c.task;if(!groups.has(k))groups.set(k,[]);groups.get(k).push(c);}
fs.mkdirSync(path.join(root,'packs'),{recursive:true});
for(const [key,cases] of groups){let offset=0;const buffers=[],details={};for(const c of cases){if(c.trajectory_offset_frames!=null)throw Error('Already packed');const b=zlib.gunzipSync(fs.readFileSync(path.join(root,c.trajectory)));if(b.length!==c.frames*72)throw Error(c.id);buffers.push(b);details[c.id]=JSON.parse(fs.readFileSync(path.join(root,c.detail),'utf8'));c.trajectory_offset_frames=offset;offset+=c.frames;c.trajectory=`packs/${key}.bin.gz`;c.detail=`packs/${key}.json.gz`;}
fs.writeFileSync(path.join(root,'packs',key+'.bin.gz'),zlib.gzipSync(Buffer.concat(buffers),{level:9}));fs.writeFileSync(path.join(root,'packs',key+'.json.gz'),zlib.gzipSync(JSON.stringify(details),{level:9}));}
fs.writeFileSync(path.join(root,'data.json'),JSON.stringify(index));
// Only move our generated per-case publication copies; raw experimental data is elsewhere.
const archive=path.resolve(__dirname,'../.local/individual-replay-exports');fs.mkdirSync(path.dirname(archive),{recursive:true});fs.renameSync(path.join(root,'replays'),archive);
console.log('Packed',index.cases.length,'cases in',groups.size,'task/cohort packs');
