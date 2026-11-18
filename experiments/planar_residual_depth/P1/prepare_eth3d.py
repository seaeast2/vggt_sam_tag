"""Freeze five RGB views from supplied COLMAP calibration, never from GT quality."""
import json,sys,time,hashlib,shutil
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image,ImageDraw
import py7zr
ROOT=Path(__file__).resolve().parents[3]
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 scene=sys.argv[1];data=ROOT/'data/planar_residual_depth/eth3d';archive=data/'archives'/f'{scene}_dslr_undistorted.7z'
 if not archive.exists():raise FileNotFoundError(archive)
 source=data/'source'
 if not (source/scene/'dslr_calibration_undistorted/images.txt').exists():
  with py7zr.SevenZipFile(archive) as z:z.extractall(source)
 cal=source/scene/'dslr_calibration_undistorted';cams={}
 for l in (cal/'cameras.txt').read_text().splitlines():
  if not l.strip() or l.startswith('#'):continue
  a=l.split();cid=int(a[0]);assert a[1]=='PINHOLE',a[1]
  fx,fy,cx,cy=map(float,a[4:]);cx-=.5;cy-=.5;cams[cid]=np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])
 lines=[l for l in (cal/'images.txt').read_text().splitlines() if not l.startswith('#')]
 views=[]
 for l in lines[::2]:
  a=l.split();idx=int(a[0]);qw,qx,qy,qz=map(float,a[1:5]);R=Rotation.from_quat([qx,qy,qz,qw]).as_matrix();t=np.array(list(map(float,a[5:8])))
  views.append({'id':idx,'R':R,'C':-R.T@t,'cid':int(a[8]),'name':a[9]})
 views.sort(key=lambda x:x['id']);centers=np.array([v['C'] for v in views]);center=np.median(centers,axis=0)
 selected=[int(np.argmin(np.linalg.norm(centers-center,axis=1)))]
 while len(selected)<5:
  dist=np.linalg.norm(centers[:,None]-centers[selected][None],axis=-1).min(1);dist[selected]=-1
  selected.append(int(np.argmax(dist)))
 rid=f'eth3d-{scene}-P1-'+time.strftime('%Y%m%d')+'-';base=ROOT/'results/planar_residual_depth';seq=1
 while (base/(rid+f'{seq:03}')).exists():seq+=1
 rid+=f'{seq:03}';out=base/rid;out.mkdir(parents=True,exist_ok=False);items=[]
 sheet=Image.new('RGB',(512,5*370),'white');draw=ImageDraw.Draw(sheet)
 for j,idx in enumerate(selected):
  v=views[idx];p=source/scene/'images'/v['name'];im=Image.open(p).convert('RGB');ow,oh=im.size
  cw=min(ow,round(oh*2));ch=min(oh,round(ow*2));left=(ow-cw)//2;top=(oh-ch)//2
  if cw>=ch:tw=512;th=max(16,round(ch/cw*512/16)*16)
  else:th=512;tw=max(16,round(cw/ch*512/16)*16)
  A=np.array([[tw/cw,0,(-left+.5)*tw/cw-.5],[0,th/ch,(-top+.5)*th/ch-.5],[0,0,1.]])
  im=im.crop((left,top,left+cw,top+ch)).resize((tw,th),Image.Resampling.BICUBIC)
  dest=out/'artifacts/inputs'/f'view{j}.png';dest.parent.mkdir(parents=True,exist_ok=True);im.save(dest)
  sheet.paste(im,(0,j*370+20));draw.text((4,j*370),f'{v["id"]}: {v["name"]}',fill='black')
  items.append({'image_id':v['id'],'source':str(p.relative_to(ROOT)),'sha256':sha(p),'processed':str(dest.relative_to(ROOT)),
  'size':[tw,th],'A':A.tolist(),'K':(A@cams[v['cid']]).tolist(),'R':v['R'].tolist(),'C':v['C'].tolist()})
 dump(out/'samples.json',[{'sample':scene,'images':items,'mask_status':'AWAITING_HUMAN_RGB_REVIEW'}])
 dump(out/'manifest.json',{'run_id':rid,'stage':'P1','status':'RGB_CALIBRATION_PREPARED_MASKS_TRACKS_MODEL_PENDING','GT_values_loaded':False,'pixel_convention':'COLMAP corner origin converted to integer pixel centers by cx/cy minus 0.5 before resize',
 'selection':'nearest coordinatewise median camera center, then farthest point by distance to selected set; image-ID tie break',
 'archive_sha256':sha(archive),'source_sha256':sha(Path(__file__)),'camera_calibration_sha256':sha(cal/'cameras.txt'),'image_calibration_sha256':sha(cal/'images.txt')})
 sheet.save(out/'rgb_contact_sheet.png');(out/'source').mkdir();shutil.copy2(__file__,out/'source/prepare_eth3d.py')
 print(rid,flush=True)
if __name__=='__main__':main()
