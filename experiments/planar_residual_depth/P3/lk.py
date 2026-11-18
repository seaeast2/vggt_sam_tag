"""Unchanged LK extraction block from P1c/run.py; RGB only."""
import hashlib
import numpy as np
import cv2
def transform(uv,T):
 x=np.c_[np.asarray(uv).reshape(-1,2),np.ones(len(np.asarray(uv).reshape(-1,2)))]@T.T
 return x[:,:2]/x[:,2:]
def track_plane(views, gray, Ts, c):
 native_poly=transform(c['polygon'],np.linalg.inv(Ts[0]));mask=np.zeros_like(gray[0]);cv2.fillPoly(mask,[np.round(native_poly).astype(np.int32)],255)
 corners=cv2.goodFeaturesToTrack(gray[0],1000,.01,3,mask=mask,blockSize=3)
 records=[];reasons={};accepted_by_source={};stats=[]
 if corners is not None:
  p0=corners.astype(np.float32);uv0=transform(p0,Ts[0])
  inside=np.array([cv2.pointPolygonTest(np.array(c['polygon'],np.float32),tuple(uv),False)>=0 for uv in uv0]);p0=p0[inside];uv0=uv0[inside]
  for j in range(1,len(views)):
   params=dict(winSize=(15,15),maxLevel=4,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01),minEigThreshold=1e-4)
   pj,sf,ef=cv2.calcOpticalFlowPyrLK(gray[0],gray[j],p0,None,**params)
   if pj is None:continue
   pb,sb,eb=cv2.calcOpticalFlowPyrLK(gray[j],gray[0],pj,None,**params)
   if pb is None:continue
   uvj=transform(pj,Ts[j]);back=transform(pb,Ts[0]);fb=np.linalg.norm(back-uv0,axis=1)
   a,b=views[0],views[j];R0,Rj=np.array(a['R']),np.array(b['R']);C0,Cj=np.array(a['C']),np.array(b['C']);K0,Kj=np.array(a['K']),np.array(b['K'])
   t=Rj@(C0-Cj);R=Rj@R0.T;tx=np.array([[0,-t[2],t[1]],[t[2],0,-t[0]],[-t[1],t[0],0]])
   F=np.linalg.inv(Kj).T@tx@R@np.linalg.inv(K0);x=np.c_[uv0,np.ones(len(uv0))];y=np.c_[uvj,np.ones(len(uvj))];ly=x@F.T;lx=y@F
   epi=abs(np.sum(y*ly,axis=1))/np.maximum(1e-12,np.minimum(np.linalg.norm(ly[:,:2],axis=1),np.linalg.norm(lx[:,:2],axis=1)))
   w,h=b['size'];valid=(sf.ravel()==1)&(sb.ravel()==1)&np.isfinite(uvj).all(1)&(fb<=.5)&(epi<=1)&(uvj[:,0]>=0)&(uvj[:,0]<w)&(uvj[:,1]>=0)&(uvj[:,1]<h)
   # Eliminate collapsed tracks landing on same target neighborhood (processed grid).
   target_used=set();kept=[]
   for k in np.argsort(fb):
    if not valid[k]:continue
    cell=tuple(np.floor(uvj[k]/2).astype(int))
    if cell in target_used:continue
    target_used.add(cell);kept.append(int(k))
   for k in kept:
    ra=R0.T@np.linalg.inv(K0)@x[k];rb=Rj.T@np.linalg.inv(Kj)@y[k]
    par=float(np.degrees(np.arccos(np.clip(ra@rb/np.linalg.norm(ra)/np.linalg.norm(rb),-1,1))))
    accepted_by_source.setdefault(k,[]).append({'view':j,'uv':uvj[k].tolist(),'fb_px':float(fb[k]),'epipolar_px':float(epi[k]),'parallax_deg':par})
   stats.append({'target_view':j,'seeds':len(p0),'LK_forward_passed':int(sf.sum()),'LK_backward_passed':int(sb.sum()),'gated':len(kept)})
  for k,other in sorted(accepted_by_source.items()):
   tid=hashlib.sha256((c['id']+':'+','.join(f'{v:.4f}' for v in uv0[k])).encode()).hexdigest();bucket=int(hashlib.sha256(('0:'+tid).encode()).hexdigest()[:8],16)%10
   records.append({'track_id':tid,'split':'heldout' if bucket<3 else 'fit','observations':[{'view':0,'uv':uv0[k].tolist()}]+other})
 return records,stats
