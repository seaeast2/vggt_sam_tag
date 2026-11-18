"""P2a: paired FIT-track bootstrap. No GT reads, no heldout-driven selection."""
import json,time,hashlib,shutil,sys,platform,importlib.util
from pathlib import Path
import numpy as np,cv2,scipy
ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'results/planar_residual_depth'
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 parent=BASE/'P2-kitti-000000-20261117-001';frozen_solver=parent/'source/solver.py'
 spec=importlib.util.spec_from_file_location('frozen_p2_solver',frozen_solver);solver=importlib.util.module_from_spec(spec);spec.loader.exec_module(solver)
 inp=json.loads((parent/'artifacts/inputs.json').read_text());views=inp['views'];polys=inp['polygons'];alltracks=inp['tracks'];fit=[t for t in alltracks if t['split']=='fit'];held=[t for t in alltracks if t['split']=='heldout']
 assert len(fit)==47 and len(held)==21
 assert not set(t['track_id'] for t in fit)&set(t['track_id'] for t in held)
 baseline=np.load(parent/'artifacts/arms/B1/depths.npz');p0=baseline['p'];depthbase=baseline['baseline'];masks=baseline['masks'];grids=[]
 for depth in depthbase:
  h,w=depth.shape;yy,xx=np.mgrid[:h,:w];grids.append(np.c_[xx.ravel(),yy.ravel()])
 pre='kitti-000000-P2a-'+time.strftime('%Y%m%d')+'-';seq=1
 while (BASE/(pre+f'{seq:03}')).exists():seq+=1
 rid=pre+f'{seq:03}';out=BASE/rid;out.mkdir(parents=True,exist_ok=False);(out/'source').mkdir();shutil.copy2(__file__,out/'source/fit_bootstrap.py');shutil.copy2(frozen_solver,out/'source/solver.py')
 config={'replicates':30,'seed':0,'bootstrap_draw_size':47,'replacement':True,'resample_unit':'FIT track; both directions kept together',
 'fixed_parent':parent.name,'solver_sha256':sha(frozen_solver),'scale_frozen':float(baseline['scale']),
 'fixed_masks_cameras_baseline_heldout':True,'holdout_track_count':21,'GT_read':False,
 'minimum_unique_FIT':12,'FIT_hull_fraction_min':.1,'median_parallax_deg_min':1,'minimum_Jacobian_singular_ratio':1e-6,
 'failure_policy':'record all draws including failures; no redraw or GT filtering','selection':'within each replicate, successful lowest bootstrap FIT Huber only'}
 dump(out/'config_resolved.json',config);dump(out/'environment.json',{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'platform':platform.platform(),'execution':'CPU, no GPU inference'})
 dump(out/'manifest.json',{'run_id':rid,'stage':'P2a','status':'RUNNING','parent':parent.name,'parent_inputs_sha256':sha(parent/'artifacts/inputs.json'),'GT_read':False})
 def obs(ts):
  a=[];b=[]
  for t in ts:
   d={o['view']:o['uv'] for o in t['observations']};a.append(d[0]);b.append(d[1])
  a,b=np.array(a),np.array(b);return [(a,b,0,1),(b,a,1,0)]
 holdobs=obs(held);rng=np.random.default_rng(0);rows=[]
 for i in range(30):
  indices=rng.integers(0,47,47);sample=[fit[j] for j in indices];unique=np.unique(indices);uv=np.float32([fit[j]['observations'][0]['uv'] for j in unique]);area=float(cv2.contourArea(cv2.convexHull(uv))/cv2.contourArea(np.float32(polys[0])))
  ob=obs(sample);qa=solver.rays(ob[0][0],views[0]);qb=solver.rays(ob[0][1],views[1]);cos=np.sum(qa*qb,axis=1)/np.linalg.norm(qa,axis=1)/np.linalg.norm(qb,axis=1);par=float(np.median(np.degrees(np.arccos(np.clip(cos,-1,1)))))
  rep=out/'artifacts'/f'replicate{i:03}';rep.mkdir(parents=True);record={'replicate':i,'draw_indices':indices.tolist(),'draw_track_ids':[t['track_id'] for t in sample],
    'unique_FIT':len(unique),'FIT_hull_fraction':area,'median_parallax_deg':par,'arms':{}}
  if len(unique)<12 or area<.1 or par<1:
   record['status']='INSUFFICIENT_SUPPORT';dump(rep/'fit.json',record);rows.append(record);continue
  for arm,fixed in [('B2',True),('B3',False)]:
   p,logs=solver.solve(p0,ob,views,polys,fixed=fixed);res={'solver':logs,'success':False}
   if p is not None:
    try:
     _,J=solver.residual_jacobian(p,ob,views);sv=np.linalg.svd(J,compute_uv=False)
     if not fixed:assert sv[-1]/sv[0]>1e-6
     corrected=depthbase.copy()
     for j in [0,1]:corrected[j][masks[j]]=solver.depths(p,grids[j][masks[j].ravel()],views[j])
     delta=corrected-depthbase
     for j in [0,1]:
      assert np.all(corrected[j][masks[j]]>0);assert np.array_equal(corrected[j][~masks[j]],depthbase[j][~masks[j]])
      assert np.allclose(depthbase[j]+delta[j],corrected[j],rtol=0,atol=1e-12)
     cost,_=solver.objective(p,ob,views);rr,_=solver.residual_jacobian(p,holdobs,views);norm=np.linalg.norm(rr.reshape(-1,2),axis=1)
     fullcost,_=solver.objective(p,obs(fit),views)
     folder=rep/arm;folder.mkdir();np.savez_compressed(folder/'depths.npz',p=p,residual_depth=delta,corrected_depth=corrected)
     res.update(success=True,p=p.tolist(),bootstrap_FIT_cost=cost,full_original_FIT_cost=fullcost,heldout_median_px=float(np.median(norm)),heldout_mean_px=float(np.mean(norm)),Jacobian_singular_values=sv.tolist())
    except (ValueError,AssertionError,FloatingPointError) as ex:res['failure']=type(ex).__name__+': '+str(ex)
   record['arms'][arm]=res
  record['status']='PAIRED_SUCCESS' if all(r['success'] for r in record['arms'].values()) else 'SOLVER_OR_TECHNICAL_FAILURE'
  dump(rep/'fit.json',record);rows.append(record);print(i,record['status'],flush=True)
 dump(out/'fit_summary.json',rows)
 # Select visual representatives before GT is read, solely by B3 full-original-FIT cost.
 eligible=[r for r in rows if r['status']=='PAIRED_SUCCESS'];ordered=sorted(eligible,key=lambda r:r['arms']['B3']['full_original_FIT_cost'])
 representatives=[ordered[k]['replicate'] for k in [0,len(ordered)//2,len(ordered)-1]] if ordered else []
 dump(out/'visual_representatives.json',{'selection':'minimum, median, maximum B3 full-original-FIT Huber cost; not GT based','replicates':representatives})
 dump(out/'manifest.json',{'run_id':rid,'stage':'P2a','status':'FIT_COMPLETED_GT_NOT_READ','parent':parent.name,'parent_inputs_sha256':sha(parent/'artifacts/inputs.json'),
  'parent_baseline_sha256':sha(parent/'artifacts/arms/B1/depths.npz'),'GT_read':False,'success_count':len(eligible),'replicates':30,'source_solver_sha256':sha(frozen_solver)})
 print('RUN',rid,flush=True)
if __name__=='__main__':main()
