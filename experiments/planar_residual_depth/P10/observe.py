"""Native-RGB SIFT scale graph and temporally seeded LK plane tracks; no depth/point GT."""
import sys,json,time,shutil,hashlib,argparse
from pathlib import Path
import numpy as np,cv2
from PIL import Image
ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'results/planar_residual_depth'
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def convert(uv,A):
 x=np.c_[np.asarray(uv).reshape(-1,2),np.ones(len(np.asarray(uv).reshape(-1,2)))]@A.T;return x[:,:2]/x[:,2:]
def epi(a,b,u,v):
 Ri,Rj=np.array(a['R']),np.array(b['R']);Ci,Cj=np.array(a['C']),np.array(b['C']);t=Rj@(Ci-Cj);R=Rj@Ri.T;tx=np.array([[0,-t[2],t[1]],[t[2],0,-t[0]],[-t[1],t[0],0]]);F=np.linalg.inv(b['K']).T@tx@R@np.linalg.inv(a['K']);x=np.c_[u,np.ones(len(u))];y=np.c_[v,np.ones(len(v))];ly=x@F.T;lx=y@F;return abs(np.sum(y*ly,axis=1))/np.maximum(1e-12,np.minimum(np.linalg.norm(ly[:,:2],axis=1),np.linalg.norm(lx[:,:2],axis=1)))
def splitid(tid):return 'heldout' if int(hashlib.sha256(('0:'+tid).encode()).hexdigest()[:8],16)%10<3 else 'fit'
def sift_tracks(views,grays):
 keys=[];desc=[];parent={}
 def find(x):
  parent.setdefault(x,x)
  if parent[x]!=x:parent[x]=find(parent[x])
  return parent[x]
 def join(a,b):parent[find(a)]=find(b)
 for v,im in zip(views,grays):
  k,d=cv2.SIFT_create(nfeatures=3000).detectAndCompute(im,None);uv=convert(np.array([p.pt for p in k]).reshape(-1,2),np.array(v['keypoint_to_calibrated']));keys.append(uv);desc.append(d)
 edges=set((i,i+1) for i in range(len(views)-1))|set((0,j) for j in range(1,len(views)))|set((15,j) for j in range(16,len(views)));stats=[]
 for i,j in sorted(edges):
  if desc[i] is None or desc[j] is None:continue
  maps=[]
  for a,b in [(i,j),(j,i)]:
   m={}
   for pp in cv2.BFMatcher().knnMatch(desc[a],desc[b],k=2):
    if len(pp)==2 and pp[0].distance<.75*pp[1].distance:m[pp[0].queryIdx]=pp[0].trainIdx
   maps.append(m)
  pairs=[(a,b) for a,b in maps[0].items() if maps[1].get(b)==a]
  if not pairs:continue
  u=np.array([keys[i][a] for a,b in pairs]);v=np.array([keys[j][b] for a,b in pairs]);ee=epi(views[i],views[j],u,v);count=0
  for (a,b),e in zip(pairs,ee):
   if e<=1:join((i,a),(j,b));count+=1
  stats.append({'i':i,'j':j,'epipolar_pass':count})
 groups={}
 for node in parent:groups.setdefault(find(node),[]).append(node)
 ts=[]
 for nodes in groups.values():
  nodes.sort()
  if len({i for i,k in nodes})!=len(nodes):continue
  tid=hashlib.sha256(';'.join(f'{i}:{k}' for i,k in nodes).encode()).hexdigest();ts.append({'track_id':tid,'split':splitid(tid),'observations':[{'view':i,'uv':keys[i][k].tolist()} for i,k in nodes]})
 return ts,stats

def lk_tracks(case,gray):
 views=case['images'];poly=case['polygons'];A=[np.array(v['keypoint_to_calibrated']) for v in views];records=[];active=[];stats=[];params=dict(winSize=(15,15),maxLevel=4,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01),minEigThreshold=1e-4)
 for frame in range(len(views)):
  if frame>0 and active:
   p0=np.float32([r['native'] for r in active]).reshape(-1,1,2);u0=convert(p0,A[frame-1]);pp,sf,_=cv2.calcOpticalFlowPyrLK(gray[frame-1],gray[frame],p0,None,**params);back,sb,_=cv2.calcOpticalFlowPyrLK(gray[frame],gray[frame-1],pp,None,**params);uv=convert(pp,A[frame]);fb=np.linalg.norm(convert(back,A[frame-1])-u0,axis=1);ee=epi(views[frame-1],views[frame],u0,uv);kept=[];cells=set()
   for k in np.argsort(fb):
    if not(sf[k,0] and sb[k,0] and np.isfinite(uv[k]).all() and fb[k]<=.5 and ee[k]<=1):continue
    if cv2.pointPolygonTest(np.float32(poly[frame]),tuple(uv[k]),False)<0:continue
    cell=tuple(np.floor(uv[k]/2).astype(int))
    if cell in cells:continue
    cells.add(cell);rec=active[k];rec['native']=pp[k,0].copy();rec['track']['observations'].append({'view':frame,'uv':uv[k].tolist(),'fb_px':float(fb[k]),'epipolar_px':float(ee[k])});kept.append(rec)
   active=kept
  if frame<16:
   nativepoly=convert(poly[frame],np.linalg.inv(A[frame]));mask=np.zeros_like(gray[frame]);cv2.fillPoly(mask,[np.rint(nativepoly).astype('int32')],255)
   for rec in active:cv2.circle(mask,tuple(np.rint(rec['native']).astype(int)),6,0,-1)
   corners=cv2.goodFeaturesToTrack(gray[frame],max(1,1000-len(active)),.01,3,mask=mask,blockSize=3)
   if corners is not None:
    for point in corners[:,0]:
     uv=convert(point[None],A[frame])[0]
     if cv2.pointPolygonTest(np.float32(poly[frame]),tuple(uv),False)<0:continue
     tid=hashlib.sha256((case['sample']+f':{frame}:'+','.join(f'{v:.4f}' for v in uv)).encode()).hexdigest();track={'track_id':tid,'split':splitid(tid),'seed_frame':frame,'observations':[{'view':frame,'uv':uv.tolist()}]};records.append(track);active.append({'native':point.copy(),'track':track})
  stats.append({'frame':frame,'active_tracks':len(active),'total_seeded_tracks':len(records),'new_seeds_allowed':frame<16})
 return [r for r in records if len(r['observations'])>=2],stats

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--masks',default='P10masks-three-sequences-20261117-001');args=ap.parse_args()
 parent=BASE/args.masks;prefix='three-sequences-P10obs-'+time.strftime('%Y%m%d')+'-';i=1
 while (BASE/(prefix+f'{i:03}')).exists():i+=1
 out=BASE/(prefix+f'{i:03}');out.mkdir();(out/'source').mkdir();shutil.copy2(__file__,out/'source/observe.py');dump(out/'config_resolved.json',json.loads((parent/'config_resolved.json').read_text()));summary=[]
 for case in json.loads((parent/'samples.json').read_text()):
  dest=out/case['sample'];dest.mkdir();dump(dest/'case.json',case);grays=[cv2.imread(str(ROOT/v['matching_RGB']),cv2.IMREAD_GRAYSCALE) for v in case['images']];ts,stats=sift_tracks(case['images'],grays);dump(dest/'scale_SIFT_tracks.json',ts);dump(dest/'SIFT_edges.json',stats);lk,steps=lk_tracks(case,grays);dump(dest/'plane_LK_tracks.json',lk);dump(dest/'LK_steps.json',steps);r={'sequence':case['sample'],'SIFT_tracks':len(ts),'LK_tracks':len(lk),'LK_FIT':sum(t['split']=='fit' for t in lk),'LK_heldout':sum(t['split']=='heldout' for t in lk)};summary.append(r);dump(out/'summary.json',summary);print(r,flush=True)
 dump(out/'manifest.json',{'run_id':out.name,'parent':parent.name,'status':'RGB_OBSERVATIONS_COMPLETED','depth_point_GT_values_loaded':False,'raw_depth_values_loaded':False,'provided_pose_control':all(c.get('metadata',{}).get('provided_pose',True) for c in json.loads((parent/'samples.json').read_text()))});print(out.name,flush=True)
if __name__=='__main__':main()
