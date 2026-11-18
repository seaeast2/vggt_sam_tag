"""Host-GPU frozen16-frame VGGT cache; future4 images and depth/point GT excluded."""
import os,sys,json,time,shutil,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'results/planar_residual_depth'
if os.environ.get('P10_HOST_GPU')!='1':raise RuntimeError('Host GPU execution required: P10_HOST_GPU=1 outside sandbox')
sys.path.insert(0,str(ROOT/'third_party/vggt-omega'))
import torch
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera

def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(8*1024*1024):h.update(b)
 return h.hexdigest()
def main():
 parent=BASE/sys.argv[1];cfg=json.loads((parent/'config_resolved.json').read_text());prefix='three-sequences-P10raw-'+time.strftime('%Y%m%d')+'-';i=1
 while (BASE/(prefix+f'{i:03}')).exists():i+=1
 out=BASE/(prefix+f'{i:03}');out.mkdir();(out/'source').mkdir();shutil.copy2(__file__,out/'source/infer.py');ckpt=ROOT/'models/vggt_omega/vggt_omega_1b_512.pt';mf={'run_id':out.name,'parent':parent.name,'status':'RUNNING','execution':'host GPU outside sandbox','depth_point_GT_values_loaded':False,'checkpoint_sha256':sha(ckpt),'sequences':[]};dump(out/'manifest.json',mf);dump(out/'environment.json',{'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'device_capability':list(torch.cuda.get_device_capability(0)),'precision':'float32 checkpoint weights, official internal autocast','resolution':cfg['model']['long_side'],'raw_context_frames':16,'future_images_in_model':False})
 try:
  model=VGGTOmega().eval();model.load_state_dict(torch.load(ckpt,map_location='cpu',weights_only=True,mmap=True));model.to('cuda')
  for sample in json.loads((parent/'samples.json').read_text()):
   views=sample['images'][:16];paths=[str(ROOT/v['processed']) for v in views];torch.cuda.reset_peak_memory_stats();start=time.time()
   with torch.inference_mode():
    images=load_and_preprocess_images(paths,mode='max_size',image_resolution=cfg['model']['long_side']).cuda();assert list(images.shape[-2:])==list(reversed(views[0]['size']));pred=model(images);ext,K=encoding_to_camera(pred['pose_enc'],pred['images'].shape[-2:])
   arrays={k:pred[k].float().cpu().numpy() for k in ['depth','depth_conf','pose_enc']};arrays.update(intrinsics=K.float().cpu().numpy(),extrinsics=ext.float().cpu().numpy());assert np.isfinite(arrays['depth']).all() and (arrays['depth']>0).all();d=out/sample['sample'];d.mkdir();np.savez_compressed(d/'predictions.npz',**arrays);row={'sample':sample['sample'],'seconds':time.time()-start,'shape':list(images.shape),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'output_sha256':sha(d/'predictions.npz'),'input_sha256':[sha(Path(p)) for p in paths]};mf['sequences'].append(row);dump(out/'manifest.json',mf);print(sample['sample'],row,flush=True);del pred,images,ext,K,arrays;torch.cuda.empty_cache()
  mf['status']='ALL_THREE_RAW_CACHES_COMPLETED'
 except Exception as e:mf['status']='FAILED';mf['error']=type(e).__name__+': '+str(e);raise
 finally:dump(out/'manifest.json',mf)
 print(out.name,flush=True)
if __name__=='__main__':main()
