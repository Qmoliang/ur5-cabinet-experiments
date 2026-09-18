"""Serve the static demo with byte-range support for reliable video seeking."""
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import argparse, functools, re, shutil
class Handler(SimpleHTTPRequestHandler):
    def send_head(self):
        self.byte_range=None
        path=Path(self.translate_path(self.path))
        header=self.headers.get('Range')
        if not header or not path.is_file():
            return super().send_head()
        size=path.stat().st_size
        match=re.fullmatch(r'bytes=(\d*)-(\d*)',header.strip())
        if not match or not size:
            self.send_error(416);return None
        first,last=match.groups()
        start=int(first) if first else max(0,size-int(last or 0))
        end=min(size-1,int(last)) if first and last else size-1
        if start>=size or start>end:
            self.send_response(416);self.send_header('Content-Range',f'bytes */{size}');self.end_headers();return None
        stream=path.open('rb');stream.seek(start)
        self.send_response(206)
        self.send_header('Content-type',self.guess_type(str(path)))
        self.send_header('Content-Length',str(end-start+1))
        self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
        self.send_header('Accept-Ranges','bytes')
        self.end_headers();self.byte_range=(start,end)
        return stream
    def copyfile(self,source,outputfile):
        try:
            if self.byte_range is None:return shutil.copyfileobj(source,outputfile)
            remaining=self.byte_range[1]-self.byte_range[0]+1
            while remaining:
                chunk=source.read(min(65536,remaining))
                if not chunk:break
                outputfile.write(chunk);remaining-=len(chunk)
        except (BrokenPipeError,ConnectionResetError):pass
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8765);args=p.parse_args()
    root=Path(__file__).resolve().parents[1]/'docs'
    print(f'Preview: http://127.0.0.1:{args.port}',flush=True)
    ThreadingHTTPServer(('127.0.0.1',args.port),functools.partial(Handler,directory=str(root))).serve_forever()

