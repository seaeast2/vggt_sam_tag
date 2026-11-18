"""Freeze Middlebury RGB/calibration/SIFT observations. GT values are not loaded."""
import json,hashlib,zipfile,sys,shutil
from pathlib import Path
from datetime import datetime
import numpy as np
import cv2
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[3]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v): p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def main():
 data=ROOT/'data/planar_residual_depth/middlebury'
 for p in (data/'archives').glob('*.zip'):
  with zipfile.ZipFile(p) as z: z.extractall(data/'source')
 source=data/'source/MiddEval3/trainingQ'
 folders=sorted(p for p in source.iterdir() if p.is_dir())
 assert len(folders)==15
 date=datetime.now().strftime('%Y%m%d');seq=1;base=ROOT/'results/planar_residual_depth'
 while (base/f'middlebury-all-P1-{date}-{seq:03}').exists():seq+=1
 rid=f'middlebury-all-P1-{date}-{seq:03}';out=base/rid;out.mkdir(parents=True,exist_ok=False)
 (out/'source').mkdir();shutil.copy2(__file__,out/'source/prepare_middlebury.py')
 sheet=Image.new('RGB',(1024,15*200),'white');draw=ImageDraw.Draw(sheet)
 entries=[]
 for row,folder in enumerate(folders):
  images=[];inputs=[];cal={}
  for line in (folder/'calib.txt').read_text().splitlines():
   k,v=line.split('=',1)
   if k in ['cam0','cam1']:cal[k]=np.array([[float(x) for x in part.split()] for part in v.strip('[]').split(';')]).tolist()
   else:cal[k]=float(v)
  for i in range(2):
   path=folder/f'im{i}.png';im=Image.open(path).convert('RGB');ow,oh=im.size
   cw=min(ow,round(oh*2));ch=min(oh,round(ow*2));left=(ow-cw)//2;top=(oh-ch)//2
   if cw>=ch:tw=512;th=max(16,round(ch/cw*512/16)*16)
   else:th=512;tw=max(16,round(cw/ch*512/16)*16)
   im=im.crop((left,top,left+cw,top+ch)).resize((tw,th),Image.Resampling.BICUBIC)
   A=np.array([[tw/cw,0,(-left+.5)*tw/cw-.5],[0,th/ch,(-top+.5)*th/ch-.5],[0,0,1.]])
   K=A@np.array(cal[f'cam{i}'])
   dest=out/'artifacts/inputs'/folder.name/f'view{i}.png';dest.parent.mkdir(parents=True,exist_ok=True);im.save(dest)
   inputs.append({'source':str(path.relative_to(ROOT)),'sha256':sha(path),'processed':str(dest.relative_to(ROOT)),
     'processed_sha256':sha(dest),'original_size':[ow,oh],'size':[tw,th],'A':A.tolist(),'K':K.tolist(),
     'R':np.eye(3).tolist(),'C':[i*cal['baseline']/1000,0,0]})
   images.append(np.array(im))
   thumb=im.copy();thumb.thumbnail((500,172));sheet.paste(thumb,(i*512,row*200+25))
  draw.text((4,row*200+4),folder.name+' | RGB only | '+('development' if folder.name in ['Adirondack','ArtL','Jadeplant'] else 'fixed evaluation'),fill='black')
  sift=cv2.SIFT_create(); kp=[];des=[]
  for image in images:
   k,d=sift.detectAndCompute(cv2.cvtColor(image,cv2.COLOR_RGB2GRAY),None);kp.append(k);des.append(d)
  matcher=cv2.BFMatcher();forward={};back={}
  for a,b,store in [(0,1,forward),(1,0,back)]:
   if des[a] is None or des[b] is None:continue
   for pair in matcher.knnMatch(des[a],des[b],k=2):
    if len(pair)==2 and pair[0].distance<.75*pair[1].distance:store[pair[0].queryIdx]=pair[0].trainIdx
  ua=[];ub=[]
  K0=np.array(inputs[0]['K']);K1=np.array(inputs[1]['K'])
  for a,b in sorted(forward.items()):
   if back.get(b)!=a:continue
   x=np.array([*kp[0][a].pt,1]);y=np.array([*kp[1][b].pt,1])
   xn=np.linalg.inv(K0)@x;yn=np.linalg.inv(K1)@y
   # Rectified-camera epipolar line, permitting actual residual vertical disparity.
   epi=abs(xn[1]-yn[1])*max(K0[1,1],K1[1,1])
   if epi<=1:ua.append(x[:2]);ub.append(y[:2])
  tracks=[]
  for j,(a,b) in enumerate(zip(ua,ub)):
   tid=f'{folder.name}:{j}'; bucket=int(hashlib.sha256(('0:'+tid).encode()).hexdigest()[:8],16)%10
   tracks.append({'track_id':tid,'uv0':list(a),'uv1':list(b),'split':'heldout' if bucket<3 else 'fit'})
  d=out/'artifacts/inputs'/folder.name
  dump(d/'calibration.json',cal);dump(d/'tracks.json',tracks)
  dump(d/'plane_masks.pending.json',{'status':'AWAITING_HUMAN_RGB_REVIEW','planes':[],
    'instructions':'Same physical plane in both views; polygon pixel coordinates in processed RGB; no GT/raw depth review before selection.'})
  entries.append({'sample':folder.name,'images':inputs,'tracks':len(tracks),'mask_status':'AWAITING_HUMAN_RGB_REVIEW',
    'gt_available':(folder/'disp0GT.pfm').exists(),'subset':'development' if folder.name in ['Adirondack','ArtL','Jadeplant'] else 'fixed_evaluation'})
 sheet.save(out/'rgb_contact_sheet.png')
 dump(out/'samples.json',entries)
 dump(out/'manifest.json',{'series':'planar_residual_depth','stage':'P1','run_id':rid,'status':'INPUTS_PREPARED_MASKS_AND_MODEL_PENDING',
  'GT_values_loaded':False,'samples':15,'source_sha256':sha(Path(__file__)),'archives':{p.name:sha(p) for p in (data/'archives').glob('*.zip')}})
 report=ROOT/'docs/experiments/planar_residual_depth/P1'/rid/'report.md'
 report.parent.mkdir(parents=True,exist_ok=True)
 report.write_text(f'# P1 Middlebury 입력 준비 — {rid}\n\n15쌍 RGB·카메라·SIFT 대응을 저장했다. GT 값은 읽지 않았다. 사람의 평면 마스크 검토와 VGGT 추론이 아직 완료되지 않았으므로 P1 완료가 아니다.\n\n결과: `{out.relative_to(ROOT)}`\n')
 with (ROOT/'docs/experiments/planar_residual_depth/index.md').open('a') as f:f.write(f'\n- P1 `{rid}`: INPUTS_PREPARED_MASKS_AND_MODEL_PENDING — [보고서](P1/{rid}/report.md)\n')
 print(rid,flush=True)
if __name__=='__main__':main()
