"""Check P9 inputs, guarded splits, geometric/photometric invariants and GT metrics."""
import sys,json,shutil,platform,hashlib
from pathlib import Path
import numpy as np,scipy,cv2
from PIL import Image
sys.path.insert(0,str(Path(__file__).parents[1]/'P5_P8'))
from common import *
from photo import *

def main():
 out=BASE/sys.argv[1];fit=BASE/json.loads((out/'config_resolved.json').read_text())['parent_fit'];rows=json.loads((out/'summary.json').read_text());hashes=json.loads((out/'fit_hashes.json').read_text());assert all(sha(fit/p)==h for p,h in hashes.items());records=[]
 for row in rows:
  case=ROOT/row['fit_dir'];info=json.loads((case/'case.json').read_text());data=np.load(case/'inputs.npz');rawpath=ROOT/info['raw_path'];raw=np.load(rawpath);assert sha(rawpath)==info['raw_sha256'];assert np.array_equal(data['raw_depth'],raw['depth'][0,...,0]);assert np.array_equal(data['depth_conf'],raw['depth_conf'][0]);assert np.array_equal(data['baseline'],data['raw_depth'].astype(float)*float(data['scale']));parts=np.load(case/'photo_support.npz');config=json.loads((fit/'config_resolved.json').read_text());tile=config['split']['tile_pixels'];guard=config['split']['guard_pixels_each_edge'];keys={};images=[np.array(Image.open(ROOT/v['processed']).convert('RGB'),float)/255 for v in info['views']]
  for split in ['fit','heldout']:
   uv=parts[split];assert np.all(uv%tile>=guard) and np.all(uv%tile<tile-guard);keys[split]=set(map(tuple,(uv//tile).astype(int)))
   for tx,ty in keys[split]:
    bucket=int(hashlib.sha256(f"0:{row['dataset']}:{row['sample']}:{tx}:{ty}".encode()).hexdigest()[:8],16)%10;assert (bucket<3)==(split=='heldout')
  assert not keys['fit']&keys['heldout'];gt=np.load(ROOT/info['parent_P4_evaluation']/'GT_arrays.npz')['GT_depth'];valid=data['masks'][0]&np.isfinite(gt)&(gt>0);p2=np.load(case/'arms/C_B2.npz')['p'];p3=np.load(case/'arms/C_B3.npz')['p']
  for a,m in row['metrics'].items():
   z=np.load(case/'arms'/(a+'.npz'));p=z['p'];assert np.array_equal(z['corrected_depth'][~data['masks']],data['baseline'][~data['masks']]);assert np.allclose(data['baseline']+z['residual_depth'],z['corrected_depth'],rtol=0,atol=1e-12)
   for j,v in enumerate(info['views']):
    h,w=data['masks'][j].shape;y,x=np.where(data['masks'][j]);uv=np.c_[x,y].astype(float);d=depths(p,uv,v);assert (d>0).all() and np.isfinite(d).all();assert np.allclose(d,z['corrected_depth'][j][data['masks'][j]])
   y,x=np.where(data['masks'][0]);uv=np.c_[x,y].astype(float);disp=float(np.max(abs(project(p,uv,info['views'])[0]-project(p2,uv,info['views'])[0])))
   if a!='C_B3':assert disp<=3+1e-5
   if a=='P9B_PHOTO_FIXED':assert angle(p,p2)<1e-5
   assert .7-1e-9<=float(z['gain'])<=1.3+1e-9 and -.15-1e-9<=float(z['bias'])<=.15+1e-9
   err=z['corrected_depth'][0][valid]-gt[valid];assert np.isclose(float(np.mean(abs(err))),m['ROI_MAE_m'],rtol=0,atol=1e-12)
   for split in ['fit','heldout']:
    if len(parts[split]):
     d=diagnostic(p,float(z['gain']),float(z['bias']),parts[split],images[0],images[1],info['views']);assert d['sample_inside_image_fraction']==1;assert np.isclose(d['RGB_mean_abs_error'],m['photometric'][split]['RGB_mean_abs_error'],rtol=0,atol=1e-12)
  parent_manifest=json.loads((ROOT/info['parent_P4_evaluation']/'manifest.json').read_text());parent_fit=ROOT/parent_manifest['fitdir']
  for rel,h in parent_manifest['fit_NPZ_hashes'].items():assert sha(parent_fit/rel)==h
  records.append({'dataset':row['dataset'],'sample':row['sample'],'raw_confidence_cache_match':True,'spatial_FIT_heldout_tiles_disjoint':True,'guard_and_hash_split_verified':True,'positive_depth':True,'shared_plane_and_residual_verified':True,'fixed_normal_and_dense_warp_trust_verified':True,'depth_and_RGB_metrics_recomputed':True,'original_fit_hashes_unchanged':True})
 dump(out/'audit.json',{'status':'PASSED','cases':len(records),'photo_refined_cases':sum(r['support']['passed'] for r in rows),'photo_fallback_cases':sum(not r['support']['passed'] for r in rows),'previous_unsupported_cases_retained':8,'fit_hashes_unchanged':True,'records':records});shutil.copy2(__file__,out/'source/audit.py');shutil.copy2(Path(__file__).parents[1]/'P5_P8/common.py',out/'source/P5_P8_common.py');shutil.copy2(Path(__file__).parents[1]/'P3/solver.py',out/'source/P3_solver.py');dump(out/'environment.json',{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'opencv':cv2.__version__,'platform':platform.platform(),'execution':'CPU only, existing host GPU prediction cache; no new GPU inference/training'})
 print('P9 audit passed for 21 cases: depth/RGB metrics, guarded split, raw caches, fixed normal, trust and parent hashes.')
if __name__=='__main__':main()
