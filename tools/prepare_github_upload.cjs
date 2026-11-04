// Prepare bounded transfer chunks for the connected GitHub Git-data API.
// Existing repository blobs are reused; raw source files are never modified.
const fs=require('fs'),path=require('path'),cp=require('child_process'),crypto=require('crypto');
const git=(...args)=>cp.execFileSync('git',args,{encoding:'utf8',maxBuffer:10*1024*1024}).trim();
const config=process.argv[2]?JSON.parse(fs.readFileSync(process.argv[2],'utf8')):null;
const old=new Set(config?[]:git('ls-tree','-r','HEAD').split('\n').map(s=>s.split(/\s+/)[2]));
const stage=new Map(git('ls-files','-s').split('\n').map(s=>{const [info,p]=s.split('\t');return [p,info.split(' ')[1]];}));
const paths=config?.paths||git('diff','--cached','--name-only','--diff-filter=AM').split('\n');const records=[],entries=[],seen=new Set();
const dir=path.resolve('.local/github-upload');fs.mkdirSync(dir,{recursive:true});
for(const p of paths){const b=fs.readFileSync(p),sha=crypto.createHash('sha1').update(Buffer.from(`blob ${b.length}\0`)).update(b).digest('hex');if(sha!==stage.get(p))throw Error('Staged bytes differ '+p);
 const text=/\.(json|html|css|js|cjs|py|md|txt|xml)$/.test(p)||p==='.gitignore';entries.push({path:p,mode:'100644',type:'blob',sha,inline:text&&!old.has(sha)});
 if(old.has(sha)||seen.has(sha))continue;seen.add(sha);const content=text?b.toString('utf8'):b.toString('base64'),parts=Math.ceil(content.length/450000);
 for(let i=0;i<parts;i++)records.push({sha,encoding:text?'utf-8':'base64',part:i,parts,data:content.slice(i*450000,(i+1)*450000)});
}
let chunk=[],size=0,count=0;function flush(){if(!chunk.length)return;fs.writeFileSync(path.join(dir,String(count++).padStart(4,'0')+'.json'),JSON.stringify(chunk));chunk=[];size=0;}
for(const r of records){const bytes=Buffer.byteLength(JSON.stringify(r));if(size+bytes>800000)flush();chunk.push(r);size+=bytes;}flush();
const plan={parent:config?.parent||git('rev-parse','HEAD'),baseTree:config?.baseTree||git('rev-parse','HEAD^{tree}'),chunks:count,entries,newBlobs:seen.size};fs.writeFileSync(path.join(dir,'plan.json'),JSON.stringify(plan));console.log({files:entries.length,chunks:count,newBlobs:seen.size,binary:entries.filter(e=>!e.inline&&!old.has(e.sha)).length});
