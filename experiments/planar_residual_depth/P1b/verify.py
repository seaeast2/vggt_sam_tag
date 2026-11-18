"""Verify canonical P1b artifacts and official PFM reference; no optimization."""
import json,hashlib,sys
from pathlib import Path
import numpy as np
from run import pfm
ROOT=Path(__file__).resolve().parents[3]
run=ROOT/'results/planar_residual_depth/P1b-baseline-direction-20261117-004'
for cid in ['K01','M01']:
 a=np.load(run/'artifacts'/cid/'arrays.npz');m=a['mask']
 assert np.array_equal(a['B1'][~m],a['baseline'][~m])
 assert np.allclose(a['baseline']+a['residual_B1'],a['B1'],rtol=0,atol=1e-6)
 assert np.all(a['B1'][m]>0)
 assert np.all(a['evaluation_mask']<=m)
p=ROOT/'data/planar_residual_depth/middlebury/source/MiddEval3/trainingQ/Adirondack/disp0GT.pfm';d=pfm(p)
ref=ROOT/'data/planar_residual_depth/middlebury/sdk_validation/Adirondack.raw'
assert np.array_equal(d,np.fromfile(ref,dtype=np.float32).reshape(d.shape))
result={'non_target_bitwise_unchanged':True,'residual_roundtrip':True,'positive_depth':True,'evaluation_mask_subset':True,
 'official_SDK_PFM_full_array_equal':True,'SDK_reference_sha256':hashlib.sha256(ref.read_bytes()).hexdigest(),
 'sdk_archive_sha256':hashlib.sha256((ref.parent/'MiddEval3-SDK-1.6.zip').read_bytes()).hexdigest(),
 'canonical':'P1b-baseline-direction-20261117-004','GT_used_for_optimization':False,'B2_B3_executed':False}
(run/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
