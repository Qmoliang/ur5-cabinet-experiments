const buffers=new Map(),jsons=new Map();
async function ok(url){const r=await fetch(url);if(!r.ok)throw Error(`${url}: HTTP ${r.status}`);return r;}
function memo(cache,key,load,max=6){if(cache.has(key))return cache.get(key);const p=load().catch(e=>{cache.delete(key);throw e;});cache.set(key,p);while(cache.size>max)cache.delete(cache.keys().next().value);return p;}
export async function gzipBuffer(url){return memo(buffers,url,async()=>new Response((await ok(url)).body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer(),4);}
export async function gzipJSON(url){return memo(jsons,url,async()=>JSON.parse(await new Response((await ok(url)).body.pipeThrough(new DecompressionStream('gzip'))).text()),6);}
export async function json(url){return (await ok(url)).json();}
export async function recording(c){
 const b=await gzipBuffer(c.trajectory),offset=c.trajectory_offset_frames*72,size=c.frames*72;
 if(offset+size>b.byteLength)throw Error('轨迹长度与索引不一致：'+c.id);
 const bytes=new Uint8Array(b,offset,size);const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))).map(x=>x.toString(16).padStart(2,'0')).join('');
 if(hash!==c.payload_sha256)throw Error('轨迹哈希不一致：'+c.id);
 return {case:c,values:new Float64Array(b,offset,c.frames*9),verified:true};
}
export async function triple(cases){return Promise.all(cases.map(recording));}
export function errorAt(record,frame){const i=Math.min(record.case.frames-1,Math.max(0,Math.round(frame)))*9;return Math.hypot(...record.case.target.map((t,j)=>record.values[i+6+j]-t))*1000;}
export function modeAt(detail,frame){if(frame===0)return 'INITIAL';const rows=detail?.control_modes_rle||[];let lo=0,hi=rows.length-1;while(lo<=hi){const m=(lo+hi)>>1,r=rows[m];if(frame<r[0])hi=m-1;else if(frame>=r[1])lo=m+1;else return r[2];}return '—';}
