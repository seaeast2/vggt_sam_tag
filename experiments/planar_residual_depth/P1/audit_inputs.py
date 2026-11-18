"""Audit frozen inputs; never inspect dense GT values."""
import json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
def main():
 results=[]
 for path in sorted((ROOT/'results/planar_residual_depth').glob('*-P1-*/samples.json')):
  entries=json.loads(path.read_text());images=0;tracks=0
  for sample in entries:
   for item in sample['images']:
    p=ROOT/item['source'];assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256']
    A=np.array(item['A']);K=np.array(item['K']);assert np.isfinite(K).all() and np.linalg.det(K)>0
    uv=np.array([[0.,0.,1.],[100.,100.,1.]]).T
    assert np.max(abs(np.linalg.inv(A)@(A@uv)-uv))<1e-9
    R=np.array(item['R']);assert np.max(abs(R@R.T-np.eye(3)))<1e-8
    images+=1
   if 'tracks' in sample:tracks+=sample['tracks']
  results.append({'run':path.parent.name,'images':images,'tracks':tracks,'checks':'source hashes, coordinate roundtrip, positive intrinsics, rotation orthogonality passed'})
 print(json.dumps(results,indent=2))
if __name__=='__main__':main()
