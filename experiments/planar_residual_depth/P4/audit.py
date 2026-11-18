"""Verify completed fit/evaluation artifacts against immutable raw inputs and snapshots."""
import sys,json,hashlib,shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'results/planar_residual_depth';sys.path.insert(0,str(Path(__file__).parents[1]/'P3'))
from batch_fit import dump,sha
from solver import rays,depths

def main():
 ev=BASE/sys.argv[1];batch=BASE/json.loads((ev/'config_resolved.json').read_text())['parent_batch'];rows=json.loads((ev/'case_results.json').read_text());attempts=json.loads((batch/'summary.json').read_text());snapshot=batch/'source';audit=[]
 assert len(attempts)==26 and len(rows)==21
 expected={('kitti',f'{i:06}') for i in range(5,20)}|{('middlebury',s['sample']) for s in json.loads((BASE/'P1-middlebury-all-20261117-001/samples.json').read_text()) if s['subset']=='fixed_evaluation'}|{('eth3d',s) for s in ['delivery_area','courtyard']}
 actual={(r['dataset'],r['sample']) for r in attempts}|{('kitti',f'{i:06}') for i in range(5,8)};assert actual==expected and len(actual)==29
 assert sha(snapshot/'solver.py')==sha(BASE/'P2-kitti-000000-20261117-001/source/solver.py')
 for r in rows:
  edir=ROOT/r['evaluation_dir'];mf=json.loads((edir/'manifest.json').read_text());fit=ROOT/mf['fitdir']
  for rel,h in mf['fit_NPZ_hashes'].items():assert sha(fit/rel)==h
  if (fit/'case.json').exists():
   case=json.loads((fit/'case.json').read_text());rawpath=ROOT/case['raw_path'];ar=fit/'arms';views=case['views'];scale=json.loads((fit/'scale.json').read_text());assert sha(rawpath)==scale['raw_sha256']
  else:
   cm=json.loads((fit/'manifest.json').read_text());obs=BASE/cm['parent'];case=json.loads((obs/'case.json').read_text());rawpath=BASE/case['raw_parent']/'artifacts/raw'/case['sample']/'predictions.npz';ar=fit/'artifacts/arms';views=case['views'];assert sha(rawpath)==cm['raw_sha256']
  raw=np.load(rawpath)['depth'][0,...,0];gt=np.load(edir/'GT_arrays.npz');mask=gt['ROI_mask'];evalmask=gt['evaluation_mask'];assert np.array_equal(evalmask,mask&np.isfinite(gt['GT_depth'])&(gt['GT_depth']>0))
  for a in ['B0','B1','B2','B3']:
   z=np.load(ar/a/'depths.npz');assert np.array_equal(z['raw_depth'],raw);assert np.array_equal(z['baseline'],raw.astype(float)*float(z['scale']));assert np.array_equal(z['corrected_depth'][~z['masks']],z['baseline'][~z['masks']]);assert np.allclose(z['baseline']+z['residual_depth'],z['corrected_depth'],rtol=0,atol=1e-12)
   if a=='B0':assert not np.any(z['residual_depth'])
   else:
    for j,v in enumerate(views):
     h,w=z['corrected_depth'][j].shape;y,x=np.mgrid[:h,:w];uv=np.c_[x.ravel(),y.ravel()];m=z['masks'][j].ravel();assert np.allclose(depths(z['p'],uv[m],v),z['corrected_depth'][j].ravel()[m]);assert np.isfinite(z['corrected_depth'][j][z['masks'][j]]).all() and np.all(z['corrected_depth'][j][z['masks'][j]]>0)
   err=z['corrected_depth'][0][evalmask]-gt['GT_depth'][evalmask];metric=next(m for m in r['metrics'] if m['arm']==a);assert np.isclose(metric['MAE_m'],np.mean(abs(err)),rtol=0,atol=1e-12)
  assert (edir/'depth_GT_errors.png').exists() and (edir/'residual_cross_section.png').exists();audit.append({'dataset':r['dataset'],'sample':r['sample'],'raw_matches_cache':True,'B0_unmodified':True,'baseline_residual_roundtrip':True,'non_target_unchanged':True,'shared_plane_depths_verified':True,'GT_metrics_recomputed':True,'fit_hashes_unchanged':True})
 # Physical invariance of expressing arbitrary fixed cameras in source coordinates.
 from scipy.spatial.transform import Rotation
 R0=Rotation.from_rotvec([.2,-.1,.3]).as_matrix();C0=np.array([2.,-3.,.5]);R1=Rotation.from_rotvec([-.1,.2,-.15]).as_matrix();C1=np.array([2.5,-2.9,.4]);K=np.array([[400,0,256],[0,400,168],[0,0,1.]])
 original=[{'R':R0,'C':C0,'K':K},{'R':R1,'C':C1,'K':K}];normalized=[{'R':v['R']@R0.T,'C':R0@(v['C']-C0),'K':K} for v in original];pref=np.array([-.05,-.1,-.3]);pw=R0.T@pref/(1-C0@(R0.T@pref));uv=np.array([[80.,100.],[250,170],[400,220]])
 error=max(float(np.max(abs(depths(pw,uv,a)-depths(pref,uv,b)))) for a,b in zip(original,normalized));assert error<1e-10
 result={'status':'PASSED','planned_cases':29,'attempted_cases':29,'GT_evaluated':21,'unsupported_or_no_plane':8,'evaluation_failures':0,'frozen_solver_verified':True,'reference_coordinate_depth_invariance_max_m':error,'cases':audit};dump(ev/'audit.json',result);shutil.copy2(__file__,ev/'source/audit.py')
 # Complete source dependencies for the post-fit report/evaluation snapshot without altering any existing snapshot.
 for name in ['lk.py','observations.py']:
  dest=ev/'source'/name;assert not dest.exists();shutil.copy2(snapshot/name,dest)
 dump(ev/'artifact_inventory.json',{str(p.relative_to(ev)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in ev.rglob('*') if p.is_file() and p.name!='artifact_inventory.json'})
 print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))
if __name__=='__main__':main()
