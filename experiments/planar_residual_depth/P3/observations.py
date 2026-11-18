"""Fixed RGB SIFT tracks, geometric gating, track-level split; no depth input."""
import json,hashlib,itertools
from pathlib import Path
import numpy as np
import cv2
ROOT=Path(__file__).resolve().parents[3]
def tracks(images):
 keys=[];desc=[];parent={};edges=[]
 def find(x):
  parent.setdefault(x,x)
  if parent[x]!=x:parent[x]=find(parent[x])
  return parent[x]
 def unite(a,b):parent[find(a)]=find(b)
 for item in images:
  im=cv2.imread(str(ROOT/item['processed']),cv2.IMREAD_GRAYSCALE)
  k,d=cv2.SIFT_create().detectAndCompute(im,None)
  if 'keypoint_to_calibrated' in item:
   A=np.array(item['keypoint_to_calibrated'])
   for point in k:
    uv=A@np.array([*point.pt,1.]);point.pt=(float(uv[0]/uv[2]),float(uv[1]/uv[2]))
  keys.append(k);desc.append(d)
 for i,j in itertools.combinations(range(len(images)),2):
  if desc[i] is None or desc[j] is None:continue
  maps=[]
  for a,b in [(i,j),(j,i)]:
   m={}
   for pair in cv2.BFMatcher().knnMatch(desc[a],desc[b],k=2):
    if len(pair)==2 and pair[0].distance<.75*pair[1].distance:m[pair[0].queryIdx]=pair[0].trainIdx
   maps.append(m)
  Ri=np.array(images[i]['R']);Rj=np.array(images[j]['R']);Ci=np.array(images[i]['C']);Cj=np.array(images[j]['C'])
  t=Rj@(Ci-Cj);R=Rj@Ri.T
  tx=np.array([[0,-t[2],t[1]],[t[2],0,-t[0]],[-t[1],t[0],0]])
  F=np.linalg.inv(np.array(images[j]['K'])).T@tx@R@np.linalg.inv(np.array(images[i]['K']))
  for a,b in sorted(maps[0].items()):
   if maps[1].get(b)!=a:continue
   x=np.array([*keys[i][a].pt,1]);y=np.array([*keys[j][b].pt,1]);ly=F@x;lx=F.T@y
   epi=abs(y@F@x)/max(1e-12,min(np.linalg.norm(ly[:2]),np.linalg.norm(lx[:2])))
   if epi<=1:unite((i,a),(j,b));edges.append(((i,a),(j,b)))
 groups={}
 for node in parent:groups.setdefault(find(node),[]).append(node)
 out=[]
 for nodes in groups.values():
  nodes.sort()
  if len(set(n[0] for n in nodes))!=len(nodes):continue
  key=';'.join(f'{i}:{a}' for i,a in nodes);tid=hashlib.sha256(key.encode()).hexdigest()
  bucket=int(hashlib.sha256(('0:'+tid).encode()).hexdigest()[:8],16)%10
  out.append({'track_id':tid,'split':'heldout' if bucket<3 else 'fit',
   'observations':[{'view':i,'uv':list(keys[i][a].pt)} for i,a in nodes]})
 return sorted(out,key=lambda x:x['track_id'])
if __name__=='__main__':
 import sys
 path=Path(sys.argv[1]);samples=json.loads((path/'samples.json').read_text())
 target=path/'artifacts/observations/tracks.json'
 if target.exists():raise FileExistsError(target)
 result={s['sample']:tracks(s['images']) for s in samples};target.parent.mkdir(parents=True,exist_ok=True)
 target.write_text(json.dumps(result,indent=2)+'\n');print({k:len(v) for k,v in result.items()})
