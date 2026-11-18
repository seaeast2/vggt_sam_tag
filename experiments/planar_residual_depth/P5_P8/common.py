import json,sys,hashlib,time
from pathlib import Path
import numpy as np,cv2
ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'results/planar_residual_depth'
sys.path.insert(0,str(Path(__file__).parents[1]/'P3'))
from solver import rays,depths,solve,residual_jacobian,objective
from fit import plane

def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def create(prefix):
 seq=1;prefix+='-'+time.strftime('%Y%m%d')+'-'
 while (BASE/(prefix+f'{seq:03}')).exists():seq+=1
 out=BASE/(prefix+f'{seq:03}');out.mkdir();return out

def load_case(row):
 edir=ROOT/row['evaluation_dir'];fit=ROOT/json.loads((edir/'manifest.json').read_text())['fitdir']
 if (fit/'case.json').exists():
  case=json.loads((fit/'case.json').read_text());inputs=json.loads((fit/'inputs.json').read_text());rawpath=ROOT/case['raw_path'];arms=fit/'arms';obs={s:[(np.array(o['uv']),np.array(o['target_uv']),o['i'],o['j']) for o in oo] for s,oo in inputs['observations'].items()}
 else:
  mf=json.loads((fit/'manifest.json').read_text());case=json.loads((BASE/mf['parent']/'case.json').read_text());inputs=json.loads((fit/'artifacts/inputs.json').read_text());rawpath=BASE/case['raw_parent']/'artifacts/raw'/case['sample']/'predictions.npz';arms=fit/'artifacts/arms';obs={}
  for split in ['fit','heldout']:
   pp=[]
   for t in inputs['tracks']:
    if t['split']!=split:continue
    oo={o['view']:o['uv'] for o in t['observations']};pp.append((oo[0],oo[1]))
   u=np.array([a for a,b in pp]);v=np.array([b for a,b in pp]);obs[split]=[(u,v,0,1),(v,u,1,0)]
 z=np.load(arms/'B0/depths.npz');rawcache=np.load(rawpath);raw=rawcache['depth'][0,...,0];conf=rawcache['depth_conf'][0];assert raw.shape==conf.shape and np.array_equal(raw,z['raw_depth'])
 ps={a:np.load(arms/a/'depths.npz')['p'] for a in ['B1','B2','B3']}
 return dict(row=row,fit=fit,rawpath=rawpath,raw=raw,conf=conf,baseline=z['baseline'],scale=float(z['scale']),masks=z['masks'],views=inputs['views'],polygons=inputs['polygons'],obs=obs,ps=ps)

def angle(a,b):return float(np.degrees(np.arccos(np.clip(abs(a@b/np.linalg.norm(a)/np.linalg.norm(b)),0,1))))
def groups(z,c,mask):
 ii=np.flatnonzero(mask.ravel()&np.isfinite(z.ravel())&(z.ravel()>0)&np.isfinite(c.ravel()));zz=z.ravel()[ii];cc=c.ravel()[ii];d=np.argsort(zz,kind='stable');s=np.argsort(cc,kind='stable');a=np.array_split(d,3);b=np.array_split(s,3)
 return {'ALL':ii,'NEAR':ii[a[0]],'MID':ii[a[1]],'FAR':ii[a[2]],'LOW_CONF':ii[b[0]],'MID_CONF':ii[b[1]],'HIGH_CONF':ii[b[2]],'NEAR_HIGH':np.intersect1d(ii[a[0]],ii[b[2]])}

def normal_info(indices,z,view,roi,protocol):
 h,w=z.shape;y,x=np.mgrid[:h,:w];uv=np.c_[x.ravel(),y.ravel()];n=len(indices);maskuv=uv[roi.ravel()];area=cv2.contourArea(cv2.convexHull(maskuv.astype(np.float32)));fraction=float(cv2.contourArea(cv2.convexHull(uv[indices].astype(np.float32)))/area) if n>=3 and area else 0.
 info={'points':n,'hull_fraction':fraction,'status':'INSUFFICIENT_NORMAL_SUPPORT'}
 if n<protocol['normal_min_points'] or fraction<protocol['normal_min_hull_fraction']:return None,info
 X=np.array(view['C'])+z.ravel()[indices,None]*rays(uv[indices],view);sv=np.linalg.svd(X-X.mean(0),compute_uv=False);ratio=float(sv[1]/max(1e-12,sv[0]));info['3D_second_to_first_singular_ratio']=ratio
 if ratio<protocol['normal_min_3D_second_to_first_singular_ratio']:return None,info
 p=plane(X);info.update(status='NORMAL_ESTIMATED',p=p.tolist(),median_estimated_camera_Z_m=float(np.median(z.ravel()[indices])))
 return p,info

def bootstrap_dispersion(indices,z,view,anchor,protocol):
 h,w=z.shape;y,x=np.mgrid[:h,:w];uv=np.c_[x.ravel(),y.ravel()];X=np.array(view['C'])+z.ravel()[indices,None]*rays(uv[indices],view);pix=uv[indices];tile=protocol['P7_bootstrap']['tile_pixels'];key=(pix[:,1]//tile)*((w+tile-1)//tile)+pix[:,0]//tile;cells=[np.flatnonzero(key==a) for a in np.unique(key)];rng=np.random.default_rng(0);aa=[]
 for _ in range(protocol['P7_bootstrap']['replicates']):
  sel=np.concatenate([cells[k] for k in rng.integers(0,len(cells),len(cells))]);p=plane(X[sel]);aa.append(angle(p,anchor))
 return float(np.sqrt(np.mean(np.array(aa)**2))),len(cells)
