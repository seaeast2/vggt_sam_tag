"""Separate RGB-only input-context ablation N2/4/8; N16 reuses original frozen run."""
import os,sys,json,time,shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'results/planar_residual_depth'
if os.environ.get('P10_HOST_GPU')!='1':raise RuntimeError('host GPU required')
sys.path.insert(0,str(ROOT/'third_party/vggt-omega'))
import torch
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera

def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')
def main():
 out=BASE/('P12-context-'+time.strftime('%Y%m%d')+'-001');out.mkdir(exist_ok=False);(out/'source').mkdir();shutil.copy2(__file__,out/'source/infer_contexts.py');dump(out/'config_resolved.json',{'resolution':384,'contexts':[8],'N16_raw_reuse':'P10raw-three-sequences-20261118-003','future_images_used':False,'purpose':'separate network RGB input count from correction-constraint count; no camera-pose/depth inputs to model','GT_role':'post-inference source0 scale-calibrated evaluation only, same fixed calibration/test pixel split'});mf={'status':'RUNNING','execution':'host GPU outside sandbox','records':[]};dump(out/'manifest.json',mf);model=VGGTOmega().eval();model.load_state_dict(torch.load(ROOT/'models/vggt_omega/vggt_omega_1b_512.pt',map_location='cpu',weights_only=True,mmap=True));model.to('cuda')
 try:
  for case in json.loads((BASE/'P10-indoor-2s-input-20261118-001/samples.json').read_text()):
   for N in [8]:
    paths=[str(ROOT/v['processed']) for v in case['images'][:N]];torch.cuda.reset_peak_memory_stats();start=time.time()
    with torch.inference_mode():
     images=load_and_preprocess_images(paths,mode='max_size',image_resolution=384).cuda();pred=model(images);ext,K=encoding_to_camera(pred['pose_enc'],pred['images'].shape[-2:])
    arrays={k:pred[k].float().cpu().numpy() for k in ['depth','depth_conf','pose_enc']};arrays.update(intrinsics=K.float().cpu().numpy(),extrinsics=ext.float().cpu().numpy());d=out/case['sample']/f'N{N:02}';d.mkdir(parents=True);np.savez_compressed(d/'predictions.npz',**arrays);r={'sequence':case['sample'],'N':N,'seconds':time.time()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated()};mf['records'].append(r);dump(out/'manifest.json',mf);print(r,flush=True);del pred,images,arrays,ext,K;torch.cuda.empty_cache()
  mf['status']='ALL_RGB_CONTEXT_ABLATIONS_COMPLETED'
 except Exception as e:mf['status']='FAILED';mf['error']=str(e);raise
 finally:dump(out/'manifest.json',mf)
 print(out.name)
if __name__=='__main__':main()
