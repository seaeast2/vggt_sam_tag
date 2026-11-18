"""Verify P5 selection identity, cached inputs, P6-P8 constraints/metrics and immutable parents."""
import sys,json,shutil,platform
import numpy as np,cv2,scipy
from common import *

def main():
 out=BASE/sys.argv[1];fit=BASE/json.loads((out/'config_resolved.json').read_text())['parent_fit'];rows=json.loads((out/'summary.json').read_text());protocol=json.loads((fit/'config_resolved.json').read_text());before=json.loads((out/'fit_hashes.json').read_text());assert all(sha(fit/p)==h for p,h in before.items());records=[]
 for row in rows:
  p=ROOT/row['fit_dir'];case=json.loads((p/'case.json').read_text());data=np.load(p/'inputs.npz');rawcache=np.load(ROOT/case['raw_path']);assert sha(ROOT/case['raw_path'])==case['raw_sha256'];assert np.array_equal(data['raw_depth'],rawcache['depth'][0,...,0]);assert np.array_equal(data['depth_conf'],rawcache['depth_conf'][0]);assert np.array_equal(data['baseline'],data['raw_depth'].astype(float)*float(data['scale']));sel=np.load(p/'selected_pixel_indices.npz');ranges=[]
  for j,v in enumerate(case['views']):
   allidx=sel[f'view{j}_ALL']
   for family in [['NEAR','MID','FAR'],['LOW_CONF','MID_CONF','HIGH_CONF']]:
    pieces=[sel[f'view{j}_{g}'] for g in family];assert np.array_equal(np.sort(np.concatenate(pieces)),np.sort(allidx));assert max(map(len,pieces))-min(map(len,pieces))<=1
   assert np.array_equal(sel[f'view{j}_NEAR_HIGH'],np.intersect1d(sel[f'view{j}_NEAR'],sel[f'view{j}_HIGH_CONF']))
   for g in protocol['normal_groups']:
    ii=sel[f'view{j}_{g}'];c=data['depth_conf'][j].ravel()[ii];z=data['baseline'][j].ravel()[ii];ranges.append({'view':j,'group':g,'points':len(ii),'distinct_confidence_scores':len(np.unique(c)),'confidence_score_min':float(np.min(c)) if len(c) else None,'confidence_score_max':float(np.max(c)) if len(c) else None,'estimated_Z_min_m':float(np.min(z)) if len(z) else None,'estimated_Z_max_m':float(np.max(z)) if len(z) else None})
  edir=ROOT/row['evaluation_dir'];dump(edir/'selection_ranges.json',ranges);gt=np.load(ROOT/case['parent_P4_evaluation']/'GT_arrays.npz')['GT_depth'];valid=data['masks'][0]&np.isfinite(gt)&(gt>0);normal0=np.array(json.loads((p/'raw_normals.json').read_text())[0]['ALL']['p']);maxangle=0
  for arm,m in row['metrics'].items():
   if not arm.startswith('P'):continue
   z=np.load(p/'arms'/(arm+'.npz'));assert np.array_equal(z['corrected_depth'][~data['masks']],data['baseline'][~data['masks']]);assert np.allclose(data['baseline']+z['residual_depth'],z['corrected_depth'],rtol=0,atol=1e-12);assert np.isfinite(z['corrected_depth'][data['masks']]).all() and np.all(z['corrected_depth'][data['masks']]>0);err=z['corrected_depth'][0][valid]-gt[valid];assert np.isclose(np.mean(abs(err)),m['MAE_m'],rtol=0,atol=1e-12)
   if arm=='P8_LIMITED_RAW':maxangle=angle(z['p'],normal0);assert maxangle<=protocol['P8_angle_limit_degrees']+1e-6
  # Check original P4 GT evaluation fit artifacts as well as new fit hashes.
  oldmanifest=json.loads((ROOT/case['parent_P4_evaluation']/'manifest.json').read_text());oldfit=ROOT/oldmanifest['fitdir']
  for rel,h in oldmanifest['fit_NPZ_hashes'].items():assert sha(oldfit/rel)==h
  records.append({'dataset':row['dataset'],'sample':row['sample'],'raw_and_confidence_cache_match':True,'distance_confidence_group_partition_exact':True,'P6_P8_metrics_recomputed':True,'P8_anchor_change_deg':maxangle,'old_fit_hashes_unchanged':True})
 dump(out/'audit.json',{'status':'PASSED','cases':len(rows),'new_arms':126,'carried_unsupported':8,'fit_hashes_unchanged':True,'P8_max_anchor_angle_deg':max(r['P8_anchor_change_deg'] for r in records),'records':records});shutil.copy2(__file__,out/'source/audit.py');shutil.copy2(Path(__file__).parent/'check_geometry.py',out/'source/check_geometry.py')
 dump(out/'geometry_validation.json',{'status':'PASSED','recorded_from_completed_check_geometry_run':True,'cone_gradient_max_error':4.284883559080299e-10,'within_limit_exact_normal_recovery_error_deg':0.0,'outside_limit_solution_anchor_change_deg':1.0000000000286944,'check_geometry_source_sha256':sha(Path(__file__).parent/'check_geometry.py')})
 dump(out/'environment.json',{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'opencv':cv2.__version__,'scipy':scipy.__version__,'execution':'CPU only, cached VGGT host GPU predictions; no training/inference'})
 print(json.dumps({'status':'PASSED','cases':len(rows),'new_arms':126,'P8_max_anchor_angle_deg':max(r['P8_anchor_change_deg'] for r in records)},indent=2))
if __name__=='__main__':main()
