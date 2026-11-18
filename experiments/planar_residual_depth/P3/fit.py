"""FIT-only plane correction, requires explicit reviewed target polygon record. Never opens GT."""
import json,time,hashlib,shutil,argparse
from pathlib import Path
import numpy as np,cv2
from PIL import Image,ImageDraw
from solver import solve,rays,depths,residual_jacobian,objective
ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'results/planar_residual_depth'
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def plane(X):
 w=np.ones(len(X))
 for _ in range(10):
  c=np.average(X,axis=0,weights=w);_,_,v=np.linalg.svd((X-c)*np.sqrt(w[:,None]),full_matrices=False);n=v[-1];e=-n@c
  r=abs(X@n+e);sigma=max(1e-8,1.4826*np.median(abs(r-np.median(r))));w=np.minimum(1,1.345*sigma/np.maximum(r,1e-12))
 return n/e

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--observations',required=True);args=ap.parse_args()
 parent=BASE/args.observations;approval=parent/'approval.json'
 a=json.loads(approval.read_text());assert a['human_approved'] is True
 case=json.loads((parent/'case.json').read_text());assert a['case_sha256']==sha(parent/'case.json')
 assert json.loads((parent/'evaluation/support.json').read_text())['passed']
 polygons=case['polygons'];views=case['views']
 rawpath=BASE/case['raw_parent']/'artifacts/raw'/case['sample']/'predictions.npz';raw=np.load(rawpath)['depth'][0,...,0]
 ts=json.loads((parent/'artifacts/all_LK_tracks.json').read_text());selected=json.loads((parent/'artifacts/tracks.json').read_text())
 frozen=json.loads((BASE/json.loads((parent/'config_resolved.json').read_text())['parent']/'config_resolved.json').read_text())
 assert sha(Path(__file__).parent/'solver.py')==frozen['solver_sha256']
 counts={s:sum(t['split']==s for t in selected) for s in ['fit','heldout']};assert counts['fit']>=12 and counts['heldout']>=6,counts
 uvfit=np.float32([t['observations'][0]['uv'] for t in selected if t['split']=='fit']);area=cv2.contourArea(cv2.convexHull(uvfit))/cv2.contourArea(np.float32(polygons[0]));assert area>=.1
 # Exclude any prior scale-FIT SIFT observations near new LK heldout observations.
 holdout_by_view={j:np.array([o['uv'] for t in ts if t['split']=='heldout' for o in t['observations'] if o['view']==j]) for j in [0,1]}
 scale_tracks=json.loads((parent/'artifacts/scale_tracks.json').read_text());ratios=[];scale_ids=[];excluded=[]
 for t in scale_tracks:
  if t['split']!='fit':continue
  obs={o['view']:np.array(o['uv']) for o in t['observations']}
  if 0 not in obs or 1 not in obs:continue
  if any(len(holdout_by_view[j]) and np.min(np.linalg.norm(holdout_by_view[j]-obs[j],axis=1))<=3 for j in [0,1]):excluded.append(t['track_id']);continue
  Ps=[];norm=[]
  for j in [0,1]:
   v=views[j];R=np.array(v['R']);C=np.array(v['C']);Ps.append(np.c_[R,-R@C]);u=np.linalg.inv(np.array(v['K']))@np.r_[obs[j],1];norm.append(u[:2,None])
  X=cv2.triangulatePoints(Ps[0],Ps[1],norm[0],norm[1]).ravel();X=X[:3]/X[3];Z=[(np.array(views[j]['R'])@(X-np.array(views[j]['C'])))[2] for j in [0,1]]
  if min(Z)<=0:continue
  x,y=obs[0];val=cv2.getRectSubPix(raw[0].astype(np.float32),(1,1),(float(x),float(y)))[0,0];ratios.append(float(Z[0]/val));scale_ids.append(t['track_id'])
 assert len(ratios)>=12;scale=float(np.median(ratios))
 observations={}
 for split in ['fit','heldout']:
  u=[];v=[]
  for t in selected:
   if t['split']!=split:continue
   obs={o['view']:o['uv'] for o in t['observations']};u.append(obs[0]);v.append(obs[1])
  u,v=np.array(u),np.array(v);observations[split]=[(u,v,0,1),(v,u,1,0)]
 fit=observations['fit'];angles=[]
 for u,v in zip(fit[0][0],fit[0][1]):
  qa=rays(u[None],views[0])[0];qb=rays(v[None],views[1])[0];angles.append(float(np.degrees(np.arccos(np.clip(qa@qb/np.linalg.norm(qa)/np.linalg.norm(qb),-1,1)))))
 assert np.median(angles)>=1
 seq=1;prefix=f"kitti-{case['sample']}-P3-"+time.strftime('%Y%m%d')+'-'
 while (BASE/(prefix+f'{seq:03}')).exists():seq+=1
 rid=prefix+f'{seq:03}';out=BASE/rid;out.mkdir(parents=True,exist_ok=False);(out/'source').mkdir()
 for file in ['fit.py','solver.py','check_solver.py']:shutil.copy2(Path(__file__).parent/file,out/'source'/file)
 dump(out/'manifest.json',{'run_id':rid,'stage':'P3','status':'RUNNING','GT_read':False,'parent':parent.name,'approval_sha256':sha(approval),'raw_sha256':sha(rawpath)})
 dump(out/'config_resolved.json',{'FIT_count':counts['fit'],'heldout_count':counts['heldout'],'hull_area_fraction':area,'median_FIT_parallax_deg':float(np.median(angles)),
 'scale':scale,'scale_FIT_tracks':len(ratios),'scale_excluded_near_holdout':len(excluded),'scale_exclusion_radius_processed_px':3,
 'objective':'bidirectional radial Huber(delta=1px), FIT only','solver':'SLSQP analytic gradient, positive-depth linear constraints, max500,ftol1e-12',
 'initializations':['raw-plane','FIT-only linear seed'],'selection':'successful lowest FIT objective; no heldout/GT selection','evaluation_scope':f"predefined evaluation sample ID{case['sample']}; no parameter tuning; scene/sequence independence unverified"})
 dump(out/'artifacts/inputs.json',{'views':views,'polygons':polygons,'tracks':selected,'scale_FIT_ids':scale_ids,'scale_excluded_ids':excluded})
 baseline=raw.astype(float)*scale;masks=[];grid=[]
 for j,v in enumerate(views):
  h,w=baseline[j].shape;yy,xx=np.mgrid[:h,:w];uv=np.c_[xx.ravel(),yy.ravel()];grid.append(uv)
  mask=Image.new('L',(w,h),0);ImageDraw.Draw(mask).polygon([tuple(p) for p in polygons[j]],fill=255);masks.append(np.array(mask)>0)
 # Shared initial plane from source view0 only; target raw plane is not averaged into it.
 v=views[0];p0=plane(np.array(v['C'])+baseline[0][masks[0],None]*rays(grid[0][masks[0].ravel()],v))
 p2,log2=solve(p0,fit,views,polygons,fixed=True);p3,log3=solve(p0,fit,views,polygons)
 dump(out/'artifacts/solver.json',{'B1_p':p0.tolist(),'B2':log2,'B3':log3});assert p2 is not None and p3 is not None
 r,J=residual_jacobian(p3,fit,views);eps=1e-6;Jn=np.empty_like(J)
 for k in range(3):Jn[:,k]=(residual_jacobian(p3+np.eye(3)[k]*eps,fit,views)[0]-residual_jacobian(p3-np.eye(3)[k]*eps,fit,views)[0])/(2*eps)
 jacerr=float(np.max(abs(J-Jn))/max(1,np.max(abs(J))));assert jacerr<1e-6
 sv=np.linalg.svd(J,compute_uv=False);assert sv[-1]/sv[0]>1e-6
 arms={};summary=[]
 for arm,p in [('B0',None),('B1',p0),('B2',p2),('B3',p3)]:
  corrected=baseline.copy()
  if p is not None:
   for j in [0,1]:corrected[j][masks[j]]=depths(p,grid[j][masks[j].ravel()],views[j])
  residual=corrected-baseline
  for j in [0,1]:
   assert np.array_equal(corrected[j][~masks[j]],baseline[j][~masks[j]])
   assert np.all(corrected[j][masks[j]]>0);assert np.allclose(baseline[j]+residual[j],corrected[j],rtol=0,atol=1e-12)
  folder=out/'artifacts/arms'/arm;folder.mkdir(parents=True)
  np.savez_compressed(folder/'depths.npz',raw_depth=raw,scale=scale,baseline=baseline,residual_depth=residual,corrected_depth=corrected,masks=np.array(masks),p=np.full(3,np.nan) if p is None else p)
  # Heldout observation losses are reporting only. Raw uses cached per-view depths.
  losses={}
  for split,obs in observations.items():
   if p is not None:r,_=residual_jacobian(p,obs,views)
   else:
    rr=[]
    for uv,vv,i,j in obs:
     vals=cv2.remap(baseline[i].astype(np.float32),uv[:,0,None].astype(np.float32),uv[:,1,None].astype(np.float32),cv2.INTER_LINEAR).ravel()
     X=np.array(views[i]['C'])+vals[:,None]*rays(uv,views[i]);h=(X-np.array(views[j]['C']))@np.array(views[j]['R']).T@np.array(views[j]['K']).T;pred=h[:,:2]/h[:,2:];rr.append((pred-vv).ravel())
    r=np.concatenate(rr)
   norms=np.linalg.norm(r.reshape(-1,2),axis=1);losses[split+'_median_px']=float(np.median(norms));losses[split+'_mean_px']=float(np.mean(norms))
  summary.append({'arm':arm,**losses})
 dump(out/'evaluation/observation_metrics.json',summary)
 dump(out/'evaluation/technical_checks.json',{'analytic_J_relative_max_error':jacerr,'Jacobian_singular_values':sv.tolist(),'positive_depth':True,'raw_preserved':True,'non_target_unchanged':True,'residual_roundtrip':True,'GT_read':False})
 dump(out/'manifest.json',{'run_id':rid,'stage':'P3','status':'FIT_COMPLETED_GT_NOT_READ','GT_read':False,'parent':parent.name,'approval_sha256':sha(approval),'raw_sha256':sha(rawpath),'proposed_primary':'B3','model_updated':False})
 print(json.dumps({'run_id':rid,'scale':scale,'counts':counts,'observations':summary},indent=2))
if __name__=='__main__':main()
