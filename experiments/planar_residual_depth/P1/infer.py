"""Host GPU only. Cache immutable VGGT predictions; no GT/masks used."""
import os,sys,json,time,hashlib,shutil,subprocess
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
if os.environ.get('P1_HOST_GPU')!='1':raise RuntimeError('Run outside sandbox with P1_HOST_GPU=1')
sys.path.insert(0,str(ROOT/'third_party/vggt-omega'))
import torch
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera

def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(8*1024*1024):h.update(b)
 return h.hexdigest()
def main():
 parent=Path(sys.argv[1]);parent=parent if parent.is_absolute() else ROOT/parent
 ckpt=ROOT/'models/vggt_omega/vggt_omega_1b_512.pt'
 if not ckpt.exists():raise FileNotFoundError('Checkpoint download not complete')
 rid=parent.name.split('-P1-')[0]+'-raw-P1-'+time.strftime('%Y%m%d')+'-';base=ROOT/'results/planar_residual_depth';seq=1
 while (base/(rid+f'{seq:03}')).exists():seq+=1
 rid+=f'{seq:03}';out=base/rid;out.mkdir(parents=True,exist_ok=False)
 (out/'source').mkdir();shutil.copy2(__file__,out/'source/infer.py')
 state={'run_id':rid,'stage':'P1','status':'RUNNING','parent_run':parent.name,'execution':'host GPU outside sandbox',
 'checkpoint':str(ckpt.relative_to(ROOT)),'checkpoint_sha256':sha(ckpt),'samples':[]}
 dump(out/'manifest.json',state)
 dump(out/'environment.json',{'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),
 'capability':list(torch.cuda.get_device_capability(0)),'python':sys.version,'precision':'float32 weights; official model internal autocast',
 'source_hashes':{str(p.relative_to(ROOT/'third_party/vggt-omega')):sha(p) for p in (ROOT/'third_party/vggt-omega/vggt_omega').rglob('*.py')}})
 try:
  model=VGGTOmega().eval();model.load_state_dict(torch.load(ckpt,map_location='cpu',weights_only=True,mmap=True));model.to('cuda')
  for item in json.loads((parent/'samples.json').read_text())[int(sys.argv[2]) if len(sys.argv)>2 else 0:]:
   name=item['sample'];paths=[str(ROOT/v['processed']) for v in item['images']]
   torch.cuda.reset_peak_memory_stats();start=time.time()
   with torch.inference_mode():
    images=load_and_preprocess_images(paths,mode='max_size',image_resolution=512).cuda()
    pred=model(images);ext,K=encoding_to_camera(pred['pose_enc'],pred['images'].shape[-2:])
   d=out/'artifacts/raw'/name;d.mkdir(parents=True)
   arrays={key:pred[key].float().cpu().numpy() for key in ['depth','depth_conf','pose_enc']}
   arrays.update(intrinsics=K.float().cpu().numpy(),extrinsics=ext.float().cpu().numpy())
   assert np.isfinite(arrays['depth']).all() and (arrays['depth']>0).all()
   np.savez_compressed(d/'predictions.npz',**arrays)
   record={'sample':name,'seconds':time.time()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
    'shape':list(images.shape),'output_sha256':sha(d/'predictions.npz'),'input_sha256':[sha(Path(p)) for p in paths]}
   state['samples'].append(record);dump(out/'manifest.json',state);print(name,record['shape'],round(record['seconds'],2),record['peak_allocated_bytes'],flush=True)
   del pred,images,ext,K,arrays;torch.cuda.empty_cache()
  state['status']='RAW_INFERENCE_COMPLETED_MASKS_PENDING'
 except Exception as exc:
  state['status']='FAILED';state['error']=type(exc).__name__+': '+str(exc);raise
 finally:dump(out/'manifest.json',state)
 print('RUN',rid,flush=True)
if __name__=='__main__':main()
