"""Read-only native RGB planar reprojection diagnostics; does not change any fit."""
import sys,json,hashlib
from pathlib import Path
import numpy as np,cv2
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parents[3];sys.path.insert(0,str(R/'experiments/planar_residual_depth/P3'))
from solver import rays,depths
B=R/'results/planar_residual_depth';fit=B/'P10fit-three-sequences-20261117-004/outdoor';case=json.loads((fit/'case.json').read_text());obs=json.loads((fit/'N16/observations.json').read_text())['FIT'];out=B/'P10projection-kitti-20261118-001';out.mkdir(exist_ok=False);(out/'render.py').write_bytes(Path(__file__).read_bytes());before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in fit.rglob('*') if p.is_file()};report=[]

def imageview(i):
 v=dict(case['images'][i]);A=np.array(v['keypoint_to_calibrated']);v['K']=(np.linalg.inv(A)@np.array(v['K'])).tolist();im=np.array(Image.open(R/v['matching_RGB']).convert('RGB'));poly=cv2.perspectiveTransform(np.float32(case['polygons'][i])[None],np.linalg.inv(A))[0];return v,im,poly,A

def project(p,u,vi,vj):
 d=depths(p,u,vi);X=np.array(vi['C'])+d[:,None]*rays(u,vi);z=(X-np.array(vj['C']))@(np.array(vj['K'])@np.array(vj['R'])).T;return z[:,:2]/z[:,2:],z[:,2]

def tile(im,title):
 pic=Image.fromarray(im);can=Image.new('RGB',(pic.width,pic.height+34),'white');can.paste(pic,(0,34));ImageDraw.Draw(can).text((8,9),title,fill='black');return can

for i,j in [(14,15),(7,8),(8,10)]:
 vi,src,pi,Ai=imageview(i);vj,tgt,pj,Aj=imageview(j);h,w=tgt.shape[:2];yy,xx=np.mgrid[:h,:w];u=np.c_[xx.ravel(),yy.ravel()];srcmask=np.zeros(src.shape[:2],np.uint8);cv2.fillPoly(srcmask,[np.rint(pi).astype('int32')],255);records=[];panels=[];si=src.copy();tj=tgt.copy();cv2.polylines(si,[np.rint(pi).astype('int32')],True,(255,30,30),2);cv2.polylines(tj,[np.rint(pj).astype('int32')],True,(255,30,30),2);o=next(r for r in obs if r['i']==i and r['j']==j);us=cv2.perspectiveTransform(np.float32(o['uv'])[None],np.linalg.inv(Ai))[0];ut=cv2.perspectiveTransform(np.float32(o['target_uv'])[None],np.linalg.inv(Aj))[0]
 for q in us:cv2.circle(si,tuple(np.rint(q).astype(int)),3,(255,255,0),1)
 for q in ut:cv2.circle(tj,tuple(np.rint(q).astype(int)),3,(0,255,255),1)
 panels=[tile(si,f'Source {i} (KITTI {50+2*i:06d}) | yellow: FIT points'),tile(tj,f'Target {j} (KITTI {50+2*j:06d}) | cyan: measured FIT points')];blends=[];zooms=[]
 for arm in ['B1','B2']:
  p=np.load(fit/'N16'/arm/'depths.npz')['p'];mapped,z=project(p,u,vj,vi);mx=mapped[:,0].reshape(h,w).astype('float32');my=mapped[:,1].reshape(h,w).astype('float32');war=cv2.remap(src,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT);valid=(cv2.remap(srcmask,mx,my,cv2.INTER_NEAREST)>0)&(z.reshape(h,w)>0)&(depths(p,u,vj).reshape(h,w)>0);war[~valid]=0;blend=tgt.copy();blend[valid]=np.rint(.5*tgt[valid]+.5*war[valid]).astype('uint8');cv2.polylines(blend,[np.rint(pj).astype('int32')],True,(255,255,255),1)
  pred,_=project(p,us,vi,vj)
  for a,b in zip(pred,ut):
   if np.isfinite(a).all() and np.max(abs(a))<10000:
    aa=tuple(np.rint(a).astype(int));bb=tuple(np.rint(b).astype(int));cv2.line(blend,aa,bb,(255,255,0),1);cv2.circle(blend,aa,3,(255,0,255),1);cv2.circle(blend,bb,3,(0,255,255),1)
  # Evaluation stays in processed-image pixel units, identical to solver coordinates.
  predcal=cv2.perspectiveTransform(pred.astype('float32')[None],Aj)[0];err=np.linalg.norm(predcal-np.array(o['target_uv']),axis=1);records.append({'arm':arm,'FIT_pairs':len(us),'median_reprojection_processed_px':float(np.median(err)),'mean_reprojection_processed_px':float(np.mean(err)),'visible_source_ROI_pixels_in_target':int(valid.sum())});Image.fromarray(war).save(out/f'{i:02}_to_{j:02}_{arm}_warp.png');Image.fromarray(blend).save(out/f'{i:02}_to_{j:02}_{arm}_overlay.png');panels.append(tile(war,f'{arm} source ROI warped into target | black: outside source ROI'));blends.append(tile(blend,f'{arm} overlay 50% | magenta: projected, cyan: measured, yellow: error'))
  crop=blend[int(h*.72):,0:int(w*.7)];zooms.append(tile(np.array(Image.fromarray(crop).resize((w,h),Image.Resampling.NEAREST)),f'{arm} bottom-road zoom | visual magnification, not added detail'))
 panels+=blends;canvas=Image.new('RGB',(w*2,(h+34)*3),'white')
 for n,panel in enumerate(panels):canvas.paste(panel,((n%2)*w,(n//2)*(h+34)))
 canvas.save(out/f'{i:02}_to_{j:02}_comparison.jpg',quality=95);zc=Image.new('RGB',(w*2,h+34),'white');zc.paste(zooms[0],(0,0));zc.paste(zooms[1],(w,0));zc.save(out/f'{i:02}_to_{j:02}_road_zoom.jpg',quality=95);report.append({'source':i,'target':j,'metrics':records})
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==hh for p,hh in before.items())
(out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n');(out/'manifest.json').write_text(json.dumps({'parent_fit':fit.parent.name,'camera_condition':'RGB estimated K/pose','source_scope':'planar ROI only; not full-scene 3D warp','selection':'most FIT points 14to15, next 7to8, longer supported 8to10; no GT used','GT_used':False,'fit_files_unchanged':True,'warp':'inverse plane warp using native matching_RGB K transform; bilinear RGB; no inpainting','colors':'red source/target ROI; cyan measured target point; magenta projected point; yellow residual line; black outside warped source ROI'},indent=2)+'\n');print(out);print(json.dumps(report,indent=2))
