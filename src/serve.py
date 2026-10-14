"""Local-only static demonstration server. No external publishing or uploads."""
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from functools import partial
import argparse
from data_pipeline import ROOT

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port',type=int,default=8765);args=p.parse_args()
    server=ThreadingHTTPServer(('127.0.0.1',args.port),partial(SimpleHTTPRequestHandler,directory=str(ROOT)))
    print(f'Open http://127.0.0.1:{args.port}/成果演示.html ; Ctrl+C stops the server',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:server.server_close()
