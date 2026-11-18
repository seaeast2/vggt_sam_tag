"""Shared-plane temporal prefixes with fixed16-frame raw cache; depth/point GT never read."""
import sys,json,time,hashlib,shutil,traceback,itertools,argparse
from pathlib import Path
import numpy as np,cv2
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'results/planar_residual_depth'
sys.path.insert(0,str(Path(__file__).parents[1]/'P3'))
from solver import rays,depths,solve,residual_jacobian
from fit import plane
sys.path.insert(0,str(Path(__file__).parents[1]/'P5_P8'))
from run_fit import constrained_solve
# Avoid module-name collision: robust plane helper is identical to P2 snapshot.
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def masks_for(case,count=16):
 result=[];grids=[]
 for j,v in enumerate(case['images'][:count]):
  w,h=v['size'];m=Image.new('L',(w,h));ImageDraw.Draw(m).polygon([tuple(p) for p in case['polygons'][j]],fill=255);result.append(np.array(m)>0);y,x=np.mgrid[:h,:w];grids.append(np.c_[x.ravel(),y.ravel()])
 return np.array(result),grids

def obs_for(tracks,views,N,split):
 group={};selected=[];angles=[]
 for t in tracks:
  if t['split']!=split:continue
  oo=[o for o in t['observations'] if o['view']<N]
  if len(oo)<2:continue
  selected.append(t['track_id'])
  for a,b in itertools.combinations(oo,2):
   i,j=a['view'],b['view'];group.setdefault((i,j),[]).append((a['uv'],b['uv']));qa=rays(np.array(a['uv'])[None],views[i])[0];qb=rays(np.array(b['uv'])[None],views[j])[0];angles.append(float(np.degrees(np.arccos(np.clip(qa@qb/np.linalg.norm(qa)/np.linalg.norm(qb),-1,1)))))
 obs=[]
 for (i,j),pp in group.items():
  u=np.array([a for a,b in pp]);v=np.array([b for a,b in pp]);obs.extend([(u,v,i,j),(v,u,j,i)])
 return obs,selected,angles

def scale_rgb(case,raw,sift,lk):
 views=case['images'];held={j:np.array([o['uv'] for t in lk if t['split']=='heldout' for o in t['observations'] if o['view']==j]) for j in range(16)};ratios=[];records=[];excluded=[]
 for t in sift:
  if t['split']!='fit':continue
  oo=[o for o in t['observations'] if o['view']<16]
  if len(oo)<2:continue
  if any(len(held[o['view']]) and np.min(np.linalg.norm(held[o['view']]-o['uv'],axis=1))<=3 for o in oo):excluded.append(t['track_id']);continue
  Q=np.array([rays(np.array(o['uv'])[None],views[o['view']])[0] for o in oo]);Q/=np.linalg.norm(Q,axis=1)[:,None];pairs=list(itertools.combinations(range(len(oo)),2));pairs.sort(key=lambda ij:Q[ij[0]]@Q[ij[1]])
  for ai,bi in pairs:
   aa,bb=oo[ai],oo[bi];i,j=aa['view'],bb['view'];Ps=[];us=[]
   for obs in [aa,bb]:
    v=views[obs['view']];R=np.array(v['R']);C=np.array(v['C']);Ps.append(np.c_[R,-R@C]);u=np.linalg.inv(v['K'])@np.r_[obs['uv'],1];us.append(u[:2,None])
   X=cv2.triangulatePoints(Ps[0],Ps[1],us[0],us[1]).ravel();X=X[:3]/X[3];Z=[(np.array(views[k]['R'])@(X-views[k]['C']))[2] for k in [i,j]]
   if not np.isfinite(Z).all() or min(Z)<=0:continue
   d=cv2.getRectSubPix(raw[i].astype('float32'),(1,1),tuple(map(float,aa['uv'])))[0,0]
   if not np.isfinite(d) or d<=0:continue
   ratio=float(Z[0]/d);ratios.append(ratio);records.append({'track_id':t['track_id'],'views':[i,j],'ratio':ratio,'parallax_deg':float(np.degrees(np.arccos(np.clip(Q[ai]@Q[bi],-1,1))))});break
 assert len(ratios)>=12,('INSUFFICIENT_SCALE_FIT_TRACKS',len(ratios))
 return float(np.median(ratios)),{'FIT_tracks':len(ratios),'ratio_percentiles':np.percentile(ratios,[5,25,50,75,95]).tolist(),'records':records,'excluded_within3px_of_plane_heldout':excluded,'future16_19_used':False}

def save_arm(out,arm,p,raw,scale,baseline,masks,grids,views,N,status):
 cor=baseline.copy();active=np.zeros_like(masks);active[:N]=masks[:N]
 if p is not None:
  for j in range(N):cor[j][masks[j]]=depths(p,grids[j][masks[j].ravel()],views[j])
 if not(np.isfinite(cor[active]).all() and np.all(cor[active]>0)):
  p=None;cor=baseline.copy();status='FALLBACK_RAW_NONPOSITIVE_PROPOSAL'
 delta=cor-baseline;assert np.array_equal(cor[~active],baseline[~active]);assert np.allclose(baseline+delta,cor,rtol=0,atol=1e-12);d=out/arm;d.mkdir(parents=True);np.savez_compressed(d/'depths.npz',p=np.full(3,np.nan) if p is None else p,residual_depth=delta,corrected_depth=cor,active_masks=active)
 return {'arm':arm,'status':status,'plane_retained':p is not None}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--observations',default='P10obs-three-sequences-20261117-001');args=ap.parse_args()
 obsparent=BASE/args.observations;rawparent=BASE/'P10raw-three-sequences-20261117-001';prefix='three-sequences-P10fit-'+time.strftime('%Y%m%d')+'-';i=1
 while (BASE/(prefix+f'{i:03}')).exists():i+=1
 out=BASE/(prefix+f'{i:03}');out.mkdir();(out/'source').mkdir();shutil.copy2(__file__,out/'source/fit.py');shutil.copy2(ROOT/'experiments/planar_residual_depth/P3/fit.py',out/'source/P3_fit_helper.py');shutil.copy2(ROOT/'experiments/planar_residual_depth/P3/solver.py',out/'source/P3_solver.py');shutil.copy2(ROOT/'experiments/planar_residual_depth/P5_P8/run_fit.py',out/'source/P8_solver_source.py');dump(out/'config_resolved.json',{**json.loads((Path(__file__).parent/'protocol.json').read_text()),'observation_protocol':json.loads((obsparent/'config_resolved.json').read_text())});summary=[]
 for name in ['indoor','outdoor','drone']:
  d=out/name;d.mkdir();case=json.loads((obsparent/name/'case.json').read_text());dump(d/'case.json',case);views=case['images'];lk=json.loads((obsparent/name/'plane_LK_tracks.json').read_text());sift=json.loads((obsparent/name/'scale_SIFT_tracks.json').read_text());rawpath=rawparent/name/'predictions.npz';raw=np.load(rawpath)['depth'][0,...,0];masks,grids=masks_for(case)
  try:
   scale,logscale=scale_rgb(case,raw,sift,lk);dump(d/'scale.json',{'scale':scale,**logscale});baseline=raw.astype(float)*scale;np.savez_compressed(d/'inputs.npz',raw_depth=raw,depth_conf=np.load(rawpath)['depth_conf'][0],scale=scale,baseline=baseline,masks=masks);p0=plane(baseline[0][masks[0],None]*rays(grids[0][masks[0].ravel()],views[0]));dump(d/'raw_plane.json',{'p0':p0.tolist(),'raw_sha256':sha(rawpath),'raw_path':str(rawpath.relative_to(ROOT))});prefixes=[]
   for N in [2,4,8,16]:
    dest=d/f'N{N:02}';dest.mkdir();fit,ids,angles=obs_for(lk,views,N,'fit');hold,hids,_=obs_for(lk,views,N,'heldout');src=np.array([o['uv'] for t in lk if t['track_id'] in set(ids) for o in t['observations'] if o['view']==0],dtype='float32');area=float(cv2.contourArea(cv2.convexHull(src))/cv2.contourArea(np.float32(case['polygons'][0]))) if len(src)>=3 else 0.;par=float(np.median(angles)) if angles else None;perview=[];idset=set(ids)
    for jj in range(N):
     pts=np.array([o['uv'] for t in lk if t['track_id'] in idset for o in t['observations'] if o['view']==jj],dtype='float32');frac=float(cv2.contourArea(cv2.convexHull(pts))/cv2.contourArea(np.float32(case['polygons'][jj]))) if len(pts)>=3 else 0.;perview.append(frac)
    passed=len(ids)>=12 and len(hids)>=6 and max(perview,default=0)>=.1 and par is not None and par>=1;sup={'FIT_tracks':len(ids),'heldout_tracks':len(hids),'FIT_source0_hull_fraction':area,'per_view_FIT_hull_fraction':perview,'max_FIT_hull_fraction':max(perview,default=0),'median_FIT_pair_parallax_deg':par,'bidirectional_FIT_pixel_pairs':sum(len(a) for a,b,i,j in fit),'passed':passed};dump(dest/'support.json',sup);dump(dest/'observations.json',{'FIT_track_ids':ids,'heldout_track_ids':hids,'FIT': [{'uv':a.tolist(),'target_uv':b.tolist(),'i':i,'j':j} for a,b,i,j in fit]});logs={};p2=p3=p8=None
    if passed:
     p2,l2=solve(p0,fit,views[:N],case['polygons'][:N],fixed=True);p3,l3=solve(p0,fit,views[:N],case['polygons'][:N]);logs.update(B2=l2,B3=l3)
     if p2 is not None and p3 is not None:p8,l8=constrained_solve(p0,p2,p3,fit,views[:N],case['polygons'][:N],1.);logs['P8']=l8
     if p3 is not None:
      _,J=residual_jacobian(p3,fit,views);sv=np.linalg.svd(J,compute_uv=False);logs['Jacobian_singular_values']=sv.tolist()
      if sv[-1]/sv[0]<=1e-6:p2=p3=p8=None;logs['rejection']='JACOBIAN_RANK_FAILED'
    dump(dest/'solver_logs.json',logs);arms=[]
    for arm,p in [('B0',None),('B1',p0),('B2',p2),('B3',p3),('P8',p8)]:
     status='RAW_BASELINE' if arm=='B0' else 'PLANE_REFINED' if p is not None else 'FALLBACK_RAW_SUPPORT_OR_SOLVER_FAILED';arms.append(save_arm(dest,arm,p,raw,scale,baseline,masks,grids,views,N,status))
    rr={'N':N,'support':sup,'arms':arms};prefixes.append(rr);dump(d/'prefix_summary.json',prefixes);print(name,N,sup,flush=True)
   summary.append({'sample':name,'status':'ALL_PREFIXES_ATTEMPTED','prefixes':prefixes,'scale':scale})
  except Exception as e:(d/'error.log').write_text(traceback.format_exc());summary.append({'sample':name,'status':'FAILED','reason':str(e)});print(name,'FAILED',str(e),flush=True)
  dump(out/'summary.json',summary)
 dump(out/'manifest.json',{'run_id':out.name,'status':'THREE_SEQUENCES_PREFIXES_ATTEMPTED','observation_parent':obsparent.name,'raw_parent':rawparent.name,'depth_point_GT_values_loaded':False,'provided_camera_pose_values_used':all(json.loads((obsparent/name/'case.json').read_text()).get('metadata',{}).get('provided_pose',True) for name in ['indoor','outdoor','drone']),'raw_context':'all16 mapping images fixed; future4 excluded','failures':sum(r['status']=='FAILED' for r in summary)});print(out.name,flush=True)
if __name__=='__main__':main()
