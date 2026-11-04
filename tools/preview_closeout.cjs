// Local read-only static preview, including byte ranges for recorded MP4 files.
const http=require('http'),fs=require('fs'),path=require('path');
const root=path.resolve(__dirname,'../docs'),mime={'.html':'text/html; charset=utf-8','.js':'text/javascript; charset=utf-8','.css':'text/css; charset=utf-8','.json':'application/json; charset=utf-8','.wasm':'application/wasm','.mp4':'video/mp4','.webp':'image/webp','.txt':'text/plain; charset=utf-8'};
http.createServer((req,res)=>{
 try{let p=path.resolve(root,'.'+decodeURIComponent(new URL(req.url,'http://localhost').pathname));if(p!==root&&!p.startsWith(root+path.sep)){res.writeHead(403).end();return;}if(fs.statSync(p).isDirectory())p=path.join(p,'index.html');const b=fs.readFileSync(p);const headers={'Content-Type':mime[path.extname(p)]||'application/octet-stream','Accept-Ranges':'bytes','Cache-Control':'no-cache'};
 const range=/^bytes=(\d+)-(\d*)$/.exec(req.headers.range||'');if(range){const start=Number(range[1]),end=range[2]?Math.min(Number(range[2]),b.length-1):b.length-1;if(start>=b.length){res.writeHead(416,{'Content-Range':`bytes */${b.length}`}).end();return;}res.writeHead(206,{...headers,'Content-Range':`bytes ${start}-${end}/${b.length}`,'Content-Length':end-start+1});res.end(b.subarray(start,end+1));}else{res.writeHead(200,{...headers,'Content-Length':b.length});res.end(b);}
 }catch(e){res.writeHead(404).end('Not found');}
}).listen(8765,'127.0.0.1',()=>console.log('Preview http://127.0.0.1:8765/closeout/'));
