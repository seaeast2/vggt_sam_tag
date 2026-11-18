"""P11 post-fit sparse-LiDAR evaluation and RGB/mask/warp evidence."""
import sys,json,importlib.util,shutil,hashlib
from pathlib import Path
import numpy as np,cv2
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parents[3];B=R/'results/planar_residual_depth';D=R/'experiments/planar_residual_depth';sys.path.insert(0,str(D/'P3'))
def module(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
E=module('e',D/'P10/evaluate.py');rgb=module('rg',D/'P10/evaluate_rgb.py');out=B/'P11eval-kitti-1s-20261118-001';out.mkdir(exist_ok=False);(out/'source').mkdir();shutil.copy2(__file__,out/'source/evaluate.py');fit=B/'P11fit-kitti-1s-20261118-001';fd=fit/'outdoor';hashes={str(p.relative_to(fit)):E.sha(p) for p in fit.rglob('*') if p.is_file()};c=json.loads((fd/'case.json').read_text());physical=json.loads((B/'P11input-kitti-1s-20261118-001/outdoor/case.json').read_text());physical['polygons']=c['polygons'];gt,mask=E.make_GT(physical,out);data=np.load(fd/'inputs.npz');base=data['baseline'];cal,test=rgb.split(mask[0],gt[0],'outdoor');alpha=float(np.median(gt[0][cal]/base[0][cal]));np.savez_compressed(out/'GT_evaluation_split.npz',calibration_mask=cal,evaluation_mask=test,alpha_evaluation_only=alpha);metrics=[];perframe=[]
for N in [2,4,8,16]:
 for a in ['B0','B1','B2','B3','P8']:
  z=np.load(fd/f'N{N:02}'/a/'depths.npz');cor=z['corrected_depth']*alpha;err=cor[0][test]-gt[0][test];row={'N':N,'arm':a,'source0_eval_pixels':int(test.sum()),'source0_MAE_m':float(np.mean(abs(err))),'source0_RMSE_m':float(np.sqrt(np.mean(err**2))),'status':next(x['status'] for x in json.loads((fd/'prefix_summary.json').read_text()) if x['N']==N for x in x['arms'] if x['arm']==a)};metrics.append(row)
  assert np.array_equal(z['corrected_depth'][~z['active_masks']],base[~z['active_masks']]);assert np.allclose(z['corrected_depth'],base+z['residual_depth']);assert np.isfinite(cor).all() and (cor>0).all()
for j in range(16):
 valid=mask[j]&np.isfinite(gt[j])&(gt[j]>0)
 if j==0:valid=test
 e=(base[j]*alpha-gt[j])[valid];perframe.append({'frame':j,'t_sec':c['images'][j]['t_sec'],'GT_pixels':int(valid.sum()),'raw_MAE_m':float(np.mean(abs(e))) if len(e) else None})
E.dump(out/'metrics.json',metrics);E.dump(out/'per_frame_raw_metrics.json',perframe);E.dump(out/'scale_evaluation.json',{'GT_calibration_pixels':int(cal.sum()),'GT_evaluation_pixels':int(test.sum()),'alpha_evaluation_only':alpha,'one_global_scale_all_frames_arms':True,'no_GT_used_for_fit':True})
# Movie all 16 real frames, one second per sampled frame, with per-view ROI.
h,w=mask.shape[1:];valid=mask&np.isfinite(gt);lo,hi=np.percentile(gt[valid],[1,99]);er=max(.01,float(np.percentile(abs((base*alpha-gt)[valid]),95)));writer=cv2.VideoWriter(str(out/'RGB_raw_GT_comparison.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),1,(w*4,h+40));assert writer.isOpened()
for j,v in enumerate(c['images']):
 im=Image.open(R/v['processed']).convert('RGB');ImageDraw.Draw(im).line([tuple(p) for p in c['polygons'][j]]+[tuple(c['polygons'][j][0])],fill='red',width=2);tiles=[np.array(im),rgb.color(base[j]*alpha,mask[j],lo,hi,'viridis'),rgb.color(gt[j],valid[j],lo,hi,'viridis'),rgb.color(base[j]*alpha-gt[j],valid[j],-er,er,'RdBu_r')];canvas=Image.new('RGB',(w*4,h+40),'white');dr=ImageDraw.Draw(canvas);dr.text((4,1),f'P11 1-second sampling | {j} t={v["t_sec"]:.3f}s | GT SCALE EVALUATION ONLY',fill='black')
 for k,(t,title) in enumerate(zip(tiles,['RGB / per-view road','raw depth','sparse LiDAR','raw minus reference'])):canvas.paste(Image.fromarray(t),(k*w,40));dr.text((k*w+3,21),title,fill='black')
 writer.write(cv2.cvtColor(np.array(canvas),cv2.COLOR_RGB2BGR))
writer.release()
# Warped footprint follows geometry, no stationary box substituted for correspondence.
p0=np.array(json.loads((fd/'raw_plane.json').read_text())['p0']);pairs=[]
for i,j in [(2,3),(6,7),(13,14)]:
 src=np.array(Image.open(R/c['images'][i]['processed']).convert('RGB'));tgt=np.array(Image.open(R/c['images'][j]['processed']).convert('RGB'));yy,xx=np.mgrid[:h,:w];uv=np.c_[xx.ravel(),yy.ravel()];vj=c['images'][j];vi=c['images'][i];z=E.depths(p0,uv,vj);X=np.array(vj['C'])+z[:,None]*E.rays(uv,vj);proj=(X-np.array(vi['C']))@(np.array(vi['K'])@np.array(vi['R'])).T;u=(proj[:,:2]/proj[:,2:]).reshape(h,w,2).astype('float32');war=cv2.remap(src,u[:,:,0],u[:,:,1],cv2.INTER_LINEAR);vm=(cv2.remap(mask[i].astype('uint8'),u[:,:,0],u[:,:,1],cv2.INTER_NEAREST)>0)&(z.reshape(h,w)>0)&(proj[:,2].reshape(h,w)>0);war[~vm]=0;blend=tgt.copy();blend[vm]=np.rint(.5*tgt[vm]+.5*war[vm]).astype('uint8');foot=np.zeros((h,w),np.uint8);foot[vm]=255;cont,_=cv2.findContours(foot,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(blend,cont,-1,(0,255,255),1);annot=[]
 for img,pol in [(src,c['polygons'][i]),(tgt,c['polygons'][j])]:
  cp=img.copy();cv2.polylines(cp,[np.int32(pol)],True,(255,0,0),1);annot.append(cp)
 panels=[annot[0],annot[1],war,blend];pic=Image.new('RGB',(w*2,(h+28)*2),'white');dd=ImageDraw.Draw(pic)
 for k,(im,txt) in enumerate(zip(panels,[f'Source {i} t={vi["t_sec"]:.3f}s',f'Target {j} t={vj["t_sec"]:.3f}s','B1 source ROI warped (black = no source coverage)','50% overlay; CYAN = actual projected footprint'])):
  x=(k%2)*w;y=(k//2)*(h+28);pic.paste(Image.fromarray(im),(x,y+28));dd.text((x+3,y+6),txt,fill='black')
 pic.resize((w*4,(h+28)*4),Image.Resampling.NEAREST).save(out/f'projection_{i:02}_to_{j:02}.png');pairs.append({'i':i,'j':j,'seconds':vj['t_sec']-vi['t_sec'],'projected_ROI_pixels':int(vm.sum()),'overlap_with_target_road_mask':int((vm&mask[j]).sum()),'plane':'B1 initial raw-fit only; B2 unsupported; not claimed accurate correspondences'})
E.dump(out/'projection_pairs.json',pairs);assert all(E.sha(fit/p)==h for p,h in hashes.items());E.dump(out/'fit_hashes.json',hashes);cap=cv2.VideoCapture(str(out/'RGB_raw_GT_comparison.mp4'));count=0
while True:
 ok,f=cap.read()
 if not ok:break
 count+=1
cap.release();assert count==16;E.dump(out/'manifest.json',{'status':'EVALUATED_WITH_RAW_FALLBACK','fit_unchanged':True,'raw_residual_outside_mask_checked':True,'video_decoded_frames':count,'model_input_spacing_sec':1,'no_supported_multiview_plane_refinement':True,'GT_used_only_for_evaluation':True});print('N16',[(x['arm'],x['source0_MAE_m']) for x in metrics if x['N']==16]);print('perframe',perframe);print(out)
