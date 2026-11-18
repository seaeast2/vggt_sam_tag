"""KITTI 2015 first 20 stereo pairs: official RGB, calibration, no GT values."""
import json,time,zipfile,hashlib,shutil
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[3]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 data=ROOT/'data/planar_residual_depth/kitti2015';source=data/'source'
 for name in ['data_scene_flow.zip','data_scene_flow_calib.zip']:
  p=data/'archives'/name
  if not p.exists():raise FileNotFoundError(p)
  with zipfile.ZipFile(p) as z:
   selected=[n for n in z.namelist() if n.startswith('training/') and any(f'/{i:06}' in n for i in range(20))]
   z.extractall(source,members=selected)
 rid='kitti2015-first20-P1-'+time.strftime('%Y%m%d')+'-';base=ROOT/'results/planar_residual_depth';seq=1
 while (base/(rid+f'{seq:03}')).exists():seq+=1
 rid+=f'{seq:03}';out=base/rid;out.mkdir(parents=True,exist_ok=False);samples=[]
 for index in range(20):
  cal=source/'training/calib_cam_to_cam'/f'{index:06}.txt';vals={}
  for l in cal.read_text().splitlines():
   key,v=l.split(':',1);vals[key]=v.strip()
  images=[];centers=[]
  for view in [2,3]:
   P=np.array(list(map(float,vals[f'P_rect_0{view}'].split()))).reshape(3,4);K=P[:,:3];centers.append(-np.linalg.inv(K)@P[:,3])
   p=source/'training'/f'image_{view}'/f'{index:06}_10.png';im=Image.open(p).convert('RGB');ow,oh=im.size
   cw=min(ow,round(oh*2));left=(ow-cw)//2;tw=512;th=max(16,round(oh/cw*512/16)*16)
   A=np.array([[tw/cw,0,(-left+.5)*tw/cw-.5],[0,th/oh,.5*th/oh-.5],[0,0,1.]])
   im=im.crop((left,0,left+cw,oh)).resize((tw,th),Image.Resampling.BICUBIC)
   dest=out/'artifacts/inputs'/f'{index:06}'/f'view{view-2}.png';dest.parent.mkdir(parents=True,exist_ok=True);im.save(dest)
   images.append({'source':str(p.relative_to(ROOT)),'sha256':sha(p),'processed':str(dest.relative_to(ROOT)),
     'size':[tw,th],'original_size':[ow,oh],'crop':[left,0,left+cw,oh],'A':A.tolist(),'K':(A@K).tolist(),
     'R':np.eye(3).tolist(),'C':centers[-1].tolist()})
  origin=centers[0]
  for im in images:im['C']=(np.array(im['C'])-origin).tolist()
  samples.append({'sample':f'{index:06}','images':images,'mask_status':'AWAITING_HUMAN_RGB_REVIEW','subset':'development' if index<5 else 'fixed_evaluation'})
 (out/'samples.json').write_text(json.dumps(samples,indent=2)+'\n');(out/'source').mkdir();shutil.copy2(__file__,out/'source/prepare_kitti.py')
 (out/'manifest.json').write_text(json.dumps({'run_id':rid,'stage':'P1','status':'RGB_CALIBRATION_PREPARED_MASKS_TRACKS_MODEL_PENDING','GT_values_loaded':False,
 'source_sha256':sha(Path(__file__)),'archives':{p.name:sha(p) for p in (data/'archives').glob('*.zip')}},indent=2)+'\n')
 print(rid,flush=True)
if __name__=='__main__':main()
