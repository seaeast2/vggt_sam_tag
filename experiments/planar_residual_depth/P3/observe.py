"""Frozen P1c LK and P1a SIFT observation settings, new reviewed evaluation scene."""
import json,time,shutil,hashlib,argparse
from pathlib import Path
import numpy as np,cv2
from PIL import Image,ImageDraw
from lk import track_plane
from observations import tracks
ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'results/planar_residual_depth'
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--approval',required=True);args=ap.parse_args()
 parent=BASE/args.input;case=json.loads((parent/'case.json').read_text());views=case['views']
 approval=json.loads(Path(args.approval).read_text());assert approval['human_approved'] is True and approval['source_and_target_approved'] is True
 assert approval['case_sha256']==sha(parent/'case.json') and approval['review_image_sha256']==sha(parent/'RGB_plane_review.png')
 seq=1;prefix=f"kitti-{case['sample']}-P3obs-"+time.strftime('%Y%m%d')+'-'
 while (BASE/(prefix+f'{seq:03}')).exists():seq+=1
 out=BASE/(prefix+f'{seq:03}');out.mkdir();(out/'source').mkdir()
 for f in ['observe.py','lk.py','observations.py']:shutil.copy2(Path(__file__).parent/f,out/'source'/f)
 dump(out/'approval.json',approval);dump(out/'case.json',case)
 dump(out/'config_resolved.json',{'parent':parent.name,'frozen_settings':json.loads((parent/'config_resolved.json').read_text()),'GT_read':False,'raw_depth_read':False,'matching_long_side_cap':1024,'never_upsample':True,'target_polygon_filter':'apply before support gating','LK_source_sha256':sha(ROOT/'experiments/planar_residual_depth/P1c/run.py')})
 gray=[];Ts=[];images=[]
 for j,v in enumerate(views):
  im=Image.open(ROOT/v['source']).convert('RGB');w,h=im.size;cw=min(w,round(h*2));ch=min(h,round(w*2));left=(w-cw)//2;top=(h-ch)//2;factor=min(1,1024/max(cw,ch));nw=round(cw*factor);nh=round(ch*factor)
  im=im.crop((left,top,left+cw,top+ch)).resize((nw,nh),Image.Resampling.BICUBIC);dest=out/'artifacts/matching_rgb'/f'view{j}.png';dest.parent.mkdir(parents=True,exist_ok=True);im.save(dest)
  An=np.array([[nw/cw,0,(-left+.5)*nw/cw-.5],[0,nh/ch,(-top+.5)*nh/ch-.5],[0,0,1.]])
  T=np.array(v['A'])@np.linalg.inv(An);Ts.append(T);gray.append(cv2.imread(str(dest),cv2.IMREAD_GRAYSCALE))
  item=dict(v);item['processed']=str(dest.relative_to(ROOT));item['keypoint_to_calibrated']=T.tolist();images.append(item)
 dump(out/'artifacts/scale_tracks.json',tracks(images))
 records,stats=track_plane(views,gray,Ts,{'id':case['candidate'],'polygon':case['polygons'][0]})
 dump(out/'artifacts/all_LK_tracks.json',records)
 selected=[t for t in records if all(cv2.pointPolygonTest(np.float32(case['polygons'][o['view']]),tuple(o['uv']),False)>=0 for o in t['observations'])]
 dump(out/'artifacts/tracks.json',selected)
 counts={s:sum(t['split']==s for t in selected) for s in ['fit','heldout']};fit=np.float32([t['observations'][0]['uv'] for t in selected if t['split']=='fit'])
 area=float(cv2.contourArea(cv2.convexHull(fit))/cv2.contourArea(np.float32(case['polygons'][0]))) if len(fit)>=3 else 0.
 angles=[o['parallax_deg'] for t in selected if t['split']=='fit' for o in t['observations'][1:]];par=float(np.median(angles)) if angles else None
 passed=counts['fit']>=12 and counts['heldout']>=6 and area>=.1 and par is not None and par>=1
 summary={'counts':counts,'FIT_hull_area_fraction':area,'median_FIT_parallax_deg':par,'passed':passed,'per_target':stats}
 dump(out/'evaluation/support.json',summary)
 for j,v in enumerate(views):
  im=Image.open(ROOT/v['processed']).convert('RGB');d=ImageDraw.Draw(im);poly=case['polygons'][j];d.line([tuple(x) for x in poly]+[tuple(poly[0])],fill='red',width=2)
  for t in selected:
   for o in t['observations']:
    if o['view']!=j:continue
    x,y=o['uv'];d.ellipse((x-2,y-2,x+2,y+2),outline='lime' if t['split']=='fit' else 'orange')
  im.save(out/f'view{j}_tracks.png')
 dump(out/'manifest.json',{'run_id':out.name,'status':'OBSERVATION_GATES_PASSED' if passed else 'INSUFFICIENT_SUPPORT','GT_read':False,'raw_depth_read':False,'human_approved_masks':True,'parent':parent.name})
 print(json.dumps({'run_id':out.name,**summary},indent=2))
if __name__=='__main__':main()
