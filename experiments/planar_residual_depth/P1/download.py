"""Official data/model downloads; credentials never serialized or printed."""
from pathlib import Path
import urllib.request, urllib.error, re, json, hashlib, concurrent.futures, time, sys
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'data/planar_residual_depth'
def fetch(task):
 url,path,auth=task; path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists(): return {'url':url,'path':str(path.relative_to(ROOT)),'status':'cached','bytes':path.stat().st_size}
 req=urllib.request.Request(url,headers={'Authorization':'Bearer '+auth} if auth else {})
 tmp=path.with_suffix(path.suffix+'.partial')
 try:
  with urllib.request.urlopen(req,timeout=90) as r,tmp.open('wb') as f:
   while b:=r.read(1024*1024): f.write(b)
  tmp.rename(path)
  print('DONE',path.name,path.stat().st_size,flush=True)
  return {'url':url,'path':str(path.relative_to(ROOT)),'status':'downloaded','bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
 except Exception as exc:
  code=getattr(exc,'code',None)
  print('FAILED',path.name,type(exc).__name__,code,flush=True)
  return {'url':url,'path':str(path.relative_to(ROOT)),'status':'failed','error_type':type(exc).__name__,'http_code':code}
def main():
 tasks=[]
 for name in ['MiddEval3-GT0-Q.zip','MiddEval3-GT1-Q.zip','MiddEval3-GTy-Q.zip']:
  tasks.append(('https://vision.middlebury.edu/stereo/submit3/zip/'+name,BASE/'middlebury/archives'/name,None))
 for scene in ['office','pipes','delivery_area','courtyard']:
  for kind in ['dslr_undistorted','dslr_depth']:
   name=f'{scene}_{kind}.7z';tasks.append(('https://www.eth3d.net/data/'+name,BASE/'eth3d/archives'/name,None))
 # Public official KITTI archive endpoints; failed access is reported, never bypassed.
 for name in ['data_scene_flow.zip','data_scene_flow_calib.zip']:
  tasks.append(('https://s3.eu-central-1.amazonaws.com/avg-kitti/'+name,BASE/'kitti2015/archives'/name,None))
 tokenfile=Path('/home/hlee221212/Documents/VGGT omega hf token.txt')
 matches=re.findall(r'hf_[A-Za-z0-9]+',tokenfile.read_text()) if tokenfile.exists() else []
 tasks.append(('https://huggingface.co/facebook/VGGT-Omega/resolve/main/vggt_omega_1b_512.pt',ROOT/'models/vggt_omega/vggt_omega_1b_512.pt',matches[0] if matches else None))
 if '--model-only' in sys.argv: tasks=[t for t in tasks if 'huggingface.co' in t[0]]
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool: results=list(pool.map(fetch,tasks))
 report=ROOT/'results/planar_residual_depth/P1-downloads';report.mkdir(parents=True,exist_ok=True)
 path=report/('attempt-'+time.strftime('%Y%m%d-%H%M%S')+'.json');path.write_text(json.dumps(results,indent=2)+'\n')
 print('AUDIT',path,flush=True)
if __name__=='__main__': main()
