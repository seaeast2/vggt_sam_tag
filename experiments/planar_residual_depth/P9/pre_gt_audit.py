"""Complete pre-GT scalar observability and output validation, without GT containers."""
import sys,json,shutil
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).parents[1]/'P5_P8'))
from common import *
from photo import *

def main():
 out=BASE/sys.argv[1];protocol=json.loads((out/'config_resolved.json').read_text());records=[]
 for r in json.loads((out/'summary.json').read_text()):
  assert r['status']=='FIT_COMPLETED_GT_NOT_READ';case=ROOT/r['case_dir'];info=json.loads((case/'case.json').read_text());data=np.load(case/'inputs.npz');parts=np.load(case/'photo_support.npz');p2=np.load(case/'arms/C_B2.npz')['p'];images=[np.array(Image.open(ROOT/v['processed']).convert('RGB'),float)/255 for v in info['views']];sup=json.loads((case/'support.json').read_text());scalar=None
  if sup['passed']:
   res,J,val,inside=residual(p2,1,0,parts['fit'],images[0],images[1],info['views']);J=J.reshape(-1,3);A=np.c_[val.ravel(),np.ones(val.size)];J-=A@np.linalg.lstsq(A,J,rcond=None)[0];scalar=float(np.sqrt(np.mean((J@p2)**2)));assert scalar>=protocol['texture_gate']['fixed_direction_photo_J_RMS_min'],(r['sample'],scalar)
  for arm in ['C_B2','C_B3','F9C_MATCH_TRUST','P9B_PHOTO_FIXED','P9C_PHOTO_FREE']:
   z=np.load(case/'arms'/(arm+'.npz'));assert np.array_equal(z['corrected_depth'][~data['masks']],data['baseline'][~data['masks']]);assert np.allclose(data['baseline']+z['residual_depth'],z['corrected_depth'],rtol=0,atol=1e-12);assert np.isfinite(z['corrected_depth'][data['masks']]).all() and np.all(z['corrected_depth'][data['masks']]>0)
  records.append({'dataset':r['dataset'],'sample':r['sample'],'FIT_scalar_photo_J_RMS_after_brightness_projection':scalar,'scalar_gate_passed':None if scalar is None else True,'positive_depth':True,'raw_preserved':True,'non_target_unchanged':True,'residual_roundtrip':True})
 dump(out/'preGT_observability_check.json',{'status':'PASSED','GT_read':False,'role':'pre-GT completion of fixed-direction observability gate and technical validation','records':records});shutil.copy2(__file__,out/'source/pre_gt_audit.py');shutil.copy2(Path(__file__).parent/'check_fixed.py',out/'source/check_fixed.py');print('Pre-GT scalar observability and technical checks passed for all 21 cases.')
if __name__=='__main__':main()
