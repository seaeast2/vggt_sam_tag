"""RGB-camera and RGB-input-count evaluation. Sensor scale alignment is evaluation-only."""
import sys,json,time,hashlib,shutil
from pathlib import Path
import numpy as np,cv2
from PIL import Image,ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'results/planar_residual_depth'
sys.path.insert(0,str(Path(__file__).parents[1]/'P3'))
from solver import rays
import importlib.util
spec=importlib.util.spec_from_file_location('p3helper',ROOT/'experiments/planar_residual_depth/P3/fit.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);plane=mod.plane

def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def angle(a,b):return float(np.degrees(np.arccos(np.clip(abs(a@b/np.linalg.norm(a)/np.linalg.norm(b)),0,1))))
def split(mask,D,name):
 y,x=np.mgrid[:mask.shape[0],:mask.shape[1]];valid=mask&np.isfinite(D)&(D>0);key=np.stack([x//16,y//16],-1);cal=np.zeros(mask.shape,bool)
 for a,b in np.unique(key[valid],axis=0):
  bucket=int(hashlib.sha256(f'0:RGB_eval_scale:{name}:{a}:{b}'.encode()).hexdigest()[:8],16)%10
  if bucket<2:cal|=valid&(key[:,:,0]==a)&(key[:,:,1]==b)
 test=valid&~cal;assert cal.sum()>=12 and test.sum()>=12;return cal,test

def color(D,mask,lo,hi,cmap):
 c=np.rint(plt.get_cmap(cmap)(np.nan_to_num(np.clip((D-lo)/max(1e-10,hi-lo),0,1),nan=0))[:,:,:3]*255).astype('uint8');c[~mask]=150;return c

def main():
 fit=BASE/'P10fit-three-sequences-20261117-004';control=BASE/'P10eval-three-sequences-20261117-002';context=BASE/'P10context-three-sequences-20261117-001';assert json.loads((fit/'manifest.json').read_text())['provided_camera_pose_values_used'] is False;assert json.loads((context/'manifest.json').read_text())['status']=='ALL_RGB_CONTEXT_ABLATIONS_COMPLETED';prefix='three-sequences-P10RGBeval-'+time.strftime('%Y%m%d')+'-';i=1
 while (BASE/(prefix+f'{i:03}')).exists():i+=1
 out=BASE/(prefix+f'{i:03}');out.mkdir();(out/'source').mkdir();shutil.copy2(__file__,out/'source/evaluate_rgb.py');config={'parent_fit':fit.name,'sensor_reference_parent':control.name,'context_ablation_parent':context.name,'camera_condition':'model RGB-estimated K and pose, no external pose values used in correction','metric_gauge':'evaluation only, one positive scalar from source0 approximately20% spatial16px blocks; all prefix/arm same scalar, remaining GT pixels scored; not metric-scale RGB inference','network_context_metric':'each independent context has its own monocular-gauge evaluation scalar on the SAME calibration pixels; no geometric optimization uses any GT','scope':'post-hoc pilot, same masks including one post-hoc drone revision; no future pose estimation or online SLAM'};dump(out/'config_resolved.json',config);hashes={str(p.relative_to(fit)):sha(p) for p in fit.rglob('*') if p.is_file()};summary=[]
 for name in ['indoor','outdoor','drone']:
  d=out/name;d.mkdir();case=json.loads((fit/name/'case.json').read_text());truecase=json.loads((BASE/'P10fit-three-sequences-20261117-003'/name/'case.json').read_text());data=np.load(fit/name/'inputs.npz');D=np.load(control/name/'GT_arrays.npz')['GT_depth'];mask=data['masks'];cal,test=split(mask[0],D[0],name);base=data['baseline'];alpha=float(np.median(D[0][cal]/base[0][cal]));Ktrue=np.array(truecase['images'][0]['K']);h,w=mask[0].shape;y,x=np.mgrid[:h,:w];uv=np.c_[x.ravel(),y.ravel()];physview={'K':Ktrue,'R':np.eye(3),'C':np.zeros(3)};q=rays(uv,physview);pg=plane(D[0][test,None]*q[test.ravel()]);metrics=[]
  np.savez_compressed(d/'GT_evaluation_split.npz',calibration_mask=cal,evaluation_mask=test,alpha_evaluation_only=alpha);dump(d/'scale_evaluation.json',{'alpha':alpha,'GT_calibration_pixels':int(cal.sum()),'GT_evaluation_pixels':int(test.sum()),'calibration_role':'evaluation gauge only, never passed to matching/plane fitting','physical_K_for_normal_evaluation_only':Ktrue.tolist(),'GT_reference_path':str((control/name/'GT_arrays.npz').relative_to(ROOT))})
  for N in [2,4,8,16]:
   ps=json.loads((fit/name/'prefix_summary.json').read_text());pp=next(r for r in ps if r['N']==N)
   for arm in ['B0','B1','B2','B3','P8']:
    z=np.load(fit/name/f'N{N:02}'/arm/'depths.npz');cor=z['corrected_depth'][0]*alpha;err=cor[test]-D[0][test];physplane=plane(cor[mask[0],None]*q[mask[0].ravel()]);m={'N':N,'arm':arm,'GT_eval_pixels':int(test.sum()),'scale_aligned_MAE_m':float(np.mean(abs(err))),'scale_aligned_RMSE_m':float(np.sqrt(np.mean(err**2))),'physical_normal_error_deg':angle(physplane,pg),'status':next(a['status'] for a in pp['arms'] if a['arm']==arm)};metrics.append(m)
  # Camera errors are post-evaluation diagnostics in a source0-anchored gauge, not inputs.
  Cpred=np.array([v['C'] for v in case['images']])*alpha;Ctrue=np.array([v['C'] for v in truecase['images'][:16]]);rot=[]
  for a,b in zip(case['images'],truecase['images'][:16]):R=np.array(a['R'])@np.array(b['R']).T;rot.append(float(np.degrees(np.arccos(np.clip((np.trace(R)-1)/2,-1,1)))))
  pose={'source0_anchored_translation_RMSE_m':float(np.sqrt(np.mean(np.sum((Cpred-Ctrue)**2,axis=1)))),'mean_relative_camera_rotation_error_deg':float(np.mean(rot)),'scale_from_source0_depth_calibration_only':alpha,'provided_GT_pose_values_used_for_correction':False,'provided_GT_pose_values_read_for_this_evaluation':True}
  rawmetrics=[]
  for N in [2,4,8,16]:
   p=(BASE/'P10raw-three-sequences-20261117-001'/name/'predictions.npz') if N==16 else context/name/f'N{N:02}/predictions.npz';raw=np.load(p)['depth'][0,0,...,0];s=float(np.median(D[0][cal]/raw[cal]));err=(raw*s)[test]-D[0][test];pn=plane((raw*s)[mask[0],None]*q[mask[0].ravel()]);rawmetrics.append({'RGB_input_frames':N,'evaluation_gauge_scale':s,'GT_eval_pixels':int(test.sum()),'scale_aligned_raw_MAE_m':float(np.mean(abs(err))),'physical_normal_error_deg':angle(pn,pg),'raw_path':str(p.relative_to(ROOT)),'raw_sha256':sha(p)})
  dump(d/'metrics.json',metrics);dump(d/'RGB_context_raw_metrics.json',rawmetrics);dump(d/'camera_error_diagnostic.json',pose)
  # RGB-estimated-pose condition movie, all16 input frames. Fixed alpha for display only.
  lo,hi=np.percentile(D[:16][mask&np.isfinite(D[:16])],[1,99]);bnd=max(1e-5,float(np.percentile(abs((base*alpha)[mask&np.isfinite(D[:16])]-D[:16][mask&np.isfinite(D[:16])]),95)))
  for arm in ['B3','P8']:
   z=np.load(fit/name/'N16'/arm/'depths.npz');writer=cv2.VideoWriter(str(d/(arm+'_RGB_pose_comparison.mp4')),cv2.VideoWriter_fourcc(*'mp4v'),5,(w*5,h+42));assert writer.isOpened()
   for j,v in enumerate(case['images']):
    rgb=Image.open(ROOT/v['processed']).convert('RGB');po=case['polygons'][j];ImageDraw.Draw(rgb).line([tuple(a) for a in po]+[tuple(po[0])],fill='red',width=2);pred=z['corrected_depth'][j]*alpha;valid=mask[j]&np.isfinite(D[j])&(D[j]>0);tiles=[np.array(rgb),color(base[j]*alpha,mask[j],lo,hi,'viridis'),color(pred,mask[j],lo,hi,'viridis'),color(D[j],valid,lo,hi,'viridis'),color(pred-D[j],valid,-bnd,bnd,'RdBu_r')];canvas=Image.new('RGB',(w*5,h+42),'white');dr=ImageDraw.Draw(canvas);dr.text((4,2),f'{name} RGB-estimated cameras | frame{j} {v["t_sec"]:.2f}s | SENSOR SCALE FOR EVALUATION ONLY',fill='black')
    for k,(tile,title) in enumerate(zip(tiles,['RGB / ROI','raw, eval scale',arm+' corrected, eval scale','sensor reference','prediction - sensor'])):canvas.paste(Image.fromarray(tile),(k*w,42));dr.text((k*w+4,23),title,fill='black')
    writer.write(cv2.cvtColor(np.array(canvas),cv2.COLOR_RGB2BGR))
   writer.release();cap=cv2.VideoCapture(str(d/(arm+'_RGB_pose_comparison.mp4')));assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==16;cap.release()
  # Native RGB-gauge pointmaps keep metric GT scale as a separate evaluation field.
  coords={};colors=[];pixels=[];frames=[]
  for a in ['B0','B3','P8']:
   z=np.load(fit/name/'N16'/a/'depths.npz');XX=[]
   for j,v in enumerate(case['images']):
    yy,xx=np.where(mask[j]);u=np.c_[xx,yy];XX.append(np.array(v['C'])+z['corrected_depth'][j][mask[j],None]*rays(u,v))
    if a=='B0':colors.append(np.array(Image.open(ROOT/v['processed']).convert('RGB'))[mask[j]]);pixels.append(u);frames.append(np.full(len(u),j))
   coords[a]=np.concatenate(XX).astype('float32')
  np.savez_compressed(d/'RGB_native_fused_ROI_pointmaps.npz',**coords,RGB=np.concatenate(colors),pixel_uv=np.concatenate(pixels),frame_id=np.concatenate(frames),evaluation_metric_scale=alpha,coordinate_units=np.array('RGB-estimated pose gauge, not sensor metric input'))
  fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
  for a in ['B0','B2','B3','P8']:
   rr=[m for m in metrics if m['arm']==a];axs[0].plot([m['N'] for m in rr],[m['scale_aligned_MAE_m']*100 for m in rr],'.-',label=a)
  axs[1].plot([m['RGB_input_frames'] for m in rawmetrics],[m['scale_aligned_raw_MAE_m']*100 for m in rawmetrics],'.-',label='VGGT raw')
  for ax in axs:ax.set_xticks([2,4,8,16]);ax.set_ylabel('scale-aligned source0 GT-test MAE cm');ax.legend()
  axs[0].set_xlabel('correction views, fixed RGB16 raw context');axs[1].set_xlabel('VGGT RGB input frames, no correction');fig.suptitle(name+' | RGB-estimated pose; GT gauge calibration only in evaluation');fig.savefig(d/'RGB_pose_and_input_count_curves.png',dpi=140);plt.close(fig)
  row={'sample':name,'metrics':metrics,'RGB_context_raw_metrics':rawmetrics,'pose_diagnostic':pose,'evaluation_alpha':alpha};summary.append(row);dump(out/'summary.json',summary);print(name,'RGB condition',[(m['arm'],round(m['scale_aligned_MAE_m']*100,3)) for m in metrics if m['N']==16],flush=True)
 assert all(sha(fit/p)==hh for p,hh in hashes.items());dump(out/'fit_hashes.json',hashes);dump(out/'manifest.json',{'run_id':out.name,'status':'RGB_CAMERA_AND_INPUT_COUNT_EVALUATION_COMPLETED','parent_fit':fit.name,'sensor_GT_used_for_correction':False,'evaluation_GT_gauge_calibration':True,'fit_outputs_unchanged':True,'future_pose_estimation':'not performed','scope':'offline16-input context, post-hoc3-clip pilot'});print(out.name)
if __name__=='__main__':main()
