"""RGB-estimated-camera condition, reuses cached16 RGB inference. No external pose/depth values."""
import json,time,shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'results/planar_residual_depth'
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def main():
 parent=BASE/'P10masks-three-sequences-20261117-002';out=BASE/('three-sequences-P10RGBinput-'+time.strftime('%Y%m%d')+'-001');out.mkdir(exist_ok=False);(out/'source').mkdir();shutil.copy2(__file__,out/'source/prepare_rgb_pose.py');cases=[]
 for name in ['indoor','outdoor','drone']:
  original=json.loads((parent/name/'case.json').read_text());raw=np.load(BASE/'P10raw-three-sequences-20261117-001'/name/'predictions.npz');ext=raw['extrinsics'][0];KK=raw['intrinsics'][0];R0=ext[0,:,:3];C0=-R0.T@ext[0,:,3];views=[]
  for j,v in enumerate(original['images'][:16]):
   # Route RGB/pixel data only; remove provided T_WC and depth/point-reference metadata.
   w={k:v[k] for k in ['source','source_sha256','processed','matching_RGB','size','original_size','A','keypoint_to_calibrated','timestamp','t_sec','role']};R=ext[j,:,:3];C=-R.T@ext[j,:,3];w.update(K=KK[j].astype(float).tolist(),R=(R@R0.T).astype(float).tolist(),C=(R0@(C-C0)).astype(float).tolist())
   if j==0:w['R']=np.eye(3).tolist();w['C']=[0.,0.,0.]
   views.append(w)
  c={'sample':name,'images':views,'polygons':original['polygons'][:16],'plane_id':original['plane_id'],'mask_provenance':original['mask_provenance'],'metadata':{'dataset':original['metadata']['dataset'],'type':name,'camera_condition':'VGGT RGB estimated K and poses only','provided_pose':False},'mask_revision':original.get('mask_revision'),'external_pose_values_used':False,'GT_depth_values_loaded':False};cases.append(c);dump(out/name/'case.json',c)
 cfg=json.loads((ROOT/'experiments/planar_residual_depth/P10/protocol.json').read_text());cfg.update(camera_condition='VGGT RGB-estimated K/extrinsics, normalized camera0, no provided external pose/intrinsic values for correction',frames={'mapping':16,'future_heldout':0,'prefixes':[2,4,8,16]},scale='one RGB-native gauge scale from FIT SIFT/model poses only; metric GT evaluation later uses one common source0 20%-calibration scalar fixed across all arms/prefixes, remaining80% GT pixels evaluated',RGB_condition='separate post-hoc camera-condition ablation; existing16 mapping raw cache, no new inference; no future pose estimation or causal online claim');dump(out/'samples.json',cases);dump(out/'config_resolved.json',cfg);dump(out/'manifest.json',{'status':'RGB_ESTIMATED_CAMERA_INPUT_PREPARED','source_raw':'P10raw-three-sequences-20261117-001','parent_masks':parent.name,'external_pose_values_used':False,'GT_depth_values_loaded':False,'mask_revision_posthoc_from_prior_control':True});print(out.name)
if __name__=='__main__':main()
