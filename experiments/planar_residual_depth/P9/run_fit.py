"""Fixed RGB supports, direct plane warping and FIT-only optimization; no GT values loaded."""
import sys,json,hashlib,shutil,traceback
from pathlib import Path
import numpy as np,cv2
from PIL import Image
from scipy.optimize import minimize,LinearConstraint
sys.path.insert(0,str(Path(__file__).parents[1]/'P5_P8'))
from common import *
from photo import *
from solver import objective

def support(item,protocol):
 views=item['views'];masks=item['masks'];h,w=masks[0].shape;source=cv2.erode(masks[0].astype('uint8'),np.ones((9,9),'uint8'),borderType=cv2.BORDER_CONSTANT,borderValue=0)>0;target=cv2.erode(masks[1].astype('uint8'),np.ones((19,19),'uint8'),borderType=cv2.BORDER_CONSTANT,borderValue=0)>0;y,x=np.where(source);uv=np.c_[x,y].astype(float);okay=np.ones(len(uv),bool)
 for p in [item['ps']['B2'],item['ps']['B3']]:
  xy,_=project(p,uv,views);roundxy=np.rint(xy).astype(int);inside=(roundxy[:,0]>=0)&(roundxy[:,0]<w)&(roundxy[:,1]>=0)&(roundxy[:,1]<h);ii=np.flatnonzero(inside);test=np.zeros(len(uv),bool);test[ii]=target[roundxy[ii,1],roundxy[ii,0]];okay&=test
 uv=uv[okay];cfg=protocol['split'];tiles=(uv//cfg['tile_pixels']).astype(int);rem=uv%cfg['tile_pixels'];guard=(rem>=cfg['guard_pixels_each_edge']).all(1)&(rem<cfg['tile_pixels']-cfg['guard_pixels_each_edge']).all(1);uv=uv[guard];tiles=tiles[guard];ds=item['row']['dataset'];sample=item['row']['sample'];buckets=np.array([int(hashlib.sha256(f'0:{ds}:{sample}:{tx}:{ty}'.encode()).hexdigest()[:8],16)%10 for tx,ty in tiles]);parts={};counts={};tilecounts={}
 for name,isheld in [('fit',False),('heldout',True)]:
  ii=np.flatnonzero((buckets<3)==isheld);cap=cfg['max_FIT_pixels'] if name=='fit' else cfg['max_heldout_pixels']
  if len(ii)>cap:ii=ii[np.linspace(0,len(ii)-1,cap).astype(int)]
  parts[name]=uv[ii];counts[name]=len(ii);tilecounts[name]=len(np.unique(tiles[ii],axis=0)) if len(ii) else 0
 area=cv2.contourArea(cv2.convexHull(parts['fit'].astype('float32')))/cv2.contourArea(np.float32(item['polygons'][0])) if len(parts['fit'])>=3 else 0;g=protocol['support_gates'];passed=counts['fit']>=g['FIT_pixels_min'] and counts['heldout']>=g['heldout_pixels_min'] and tilecounts['fit']>=g['FIT_tiles_min'] and tilecounts['heldout']>=g['heldout_tiles_min'] and area>=g['FIT_hull_fraction_min']
 return parts,{'counts':counts,'tiles':tilecounts,'FIT_hull_fraction':float(area),'passed':bool(passed),'frozen_support_before_photo_optimization':True}

def feature_control(item):
 ps=item['ps'];unpack,nd,cs,valid=constraints(item['views'],item['polygons'],ps['B2']);lin=cs[0];non=cs[1];cons=[LinearConstraint(lin.A[:,:3],lin.lb,lin.ub),{'type':'ineq','fun':lambda p:non['fun'](np.r_[p,1,0]),'jac':lambda p:non['jac'](np.r_[p,1,0])[:,:3]}];logs=[]
 for label,p in [('B2',ps['B2']),('B3',ps['B3']),('raw',ps['B1'])]:
  if not valid(p):logs.append({'seed':label,'success':False,'reason':'INITIAL_TRUST_FAILED'});continue
  fun=lambda pp:objective(pp,item['obs']['fit'],item['views']);opt=minimize(lambda p:fun(p)[0],p,jac=lambda p:fun(p)[1],method='SLSQP',constraints=cons,options={'maxiter':500,'ftol':1e-12});logs.append({'seed':label,'success':bool(opt.success and valid(opt.x)),'p':opt.x.tolist(),'FIT_feature_cost':float(opt.fun),'message':str(opt.message)})
 good=[l for l in logs if l['success']];return (np.array(min(good,key=lambda l:l['FIT_feature_cost'])['p']) if good else ps['B2']),logs

def save_arm(out,arm,p,gain,bias,item,images,parts,status,reason=None):
 baseline=item['baseline'];masks=item['masks'];cor=baseline.copy();views=item['views'];phot={}
 for j,v in enumerate(views):
  h,w=cor[j].shape;y,x=np.mgrid[:h,:w];uv=np.c_[x.ravel(),y.ravel()];m=masks[j].ravel();z=depths(p,uv[m],v);assert np.isfinite(z).all() and np.all(z>0);cor[j].ravel()[m]=z
 assert np.array_equal(cor[~masks],baseline[~masks]);delta=cor-baseline;assert np.allclose(baseline+delta,cor,rtol=0,atol=1e-12)
 # Verify full source mask, not only corner constraints, obeys the frozen trust domain.
 h,w=masks[0].shape;y,x=np.where(masks[0]);uu=np.c_[x,y].astype(float);a,_=project(p,uu,views);b,_=project(item['ps']['B2'],uu,views);maxshift=float(np.max(abs(a-b)))
 if arm not in ['C_B3']:assert maxshift<=3+1e-5,(arm,maxshift)
 folder=out/'arms';folder.mkdir(exist_ok=True);np.savez_compressed(folder/(arm+'.npz'),p=p,gain=gain,bias=bias,residual_depth=delta,corrected_depth=cor)
 for split,uv in parts.items():
  if len(uv):phot[split]=diagnostic(p,gain,bias,uv,images[0],images[1],views);phot[split]['Huber_cost']=loss_and_grad(p,gain,bias,uv,images[0],images[1],views)[0]
  else:phot[split]=None
 feature={}
 for split,oo in item['obs'].items():
  r,_=residual_jacobian(p,oo,views);rr=np.linalg.norm(r.reshape(-1,2),axis=1);feature[split+'_feature_mean_px']=float(np.mean(rr));feature[split+'_feature_median_px']=float(np.median(rr))
 return {'arm':arm,'status':status,'reason':reason,'gain':gain,'bias':bias,'max_dense_source_warp_delta_from_B2_per_coordinate_px':maxshift,'photometric':phot,**feature}

def fit_case(item,out,protocol):
 out.mkdir(parents=True);images=[np.array(Image.open(ROOT/v['processed']).convert('RGB'),dtype=float)/255 for v in item['views']];parts,sup=support(item,protocol);np.savez_compressed(out/'photo_support.npz',**parts);dump(out/'support.json',sup);np.savez_compressed(out/'inputs.npz',raw_depth=item['raw'],depth_conf=item['conf'],scale=item['scale'],baseline=item['baseline'],masks=item['masks'])
 dump(out/'case.json',{'dataset':item['row']['dataset'],'sample':item['row']['sample'],'views':item['views'],'polygons':item['polygons'],'raw_path':str(item['rawpath'].relative_to(ROOT)),'raw_sha256':sha(item['rawpath']),'parent_fit':str(item['fit'].relative_to(ROOT)),'parent_P4_evaluation':item['row']['evaluation_dir'],'mask_provenance':item['row']['mask_provenance'],'GT_read':False})
 logs={};metrics=[]
 def brightness(p):return fit_brightness(p,parts['fit'],images[0],images[1],item['views'])[:2] if len(parts['fit']) else (1.,0.)
 for arm,name in [('C_B2','B2'),('C_B3','B3')]:
  p=item['ps'][name];g,b=brightness(p);metrics.append(save_arm(out,arm,p,g,b,item,images,parts,'ORIGINAL_GEOMETRY_BRIGHTNESS_FIT'))
 p,log=feature_control(item);logs['F9C_MATCH_TRUST']=log;g,b=brightness(p);metrics.append(save_arm(out,'F9C_MATCH_TRUST',p,g,b,item,images,parts,'FEATURE_TRUST_CONTROL' if any(l['success'] for l in log) else 'FALLBACK_B2'))
 for arm,fixed in [('P9B_PHOTO_FIXED',True),('P9C_PHOTO_FREE',False)]:
  reason=None;best=None;log=[]
  if not sup['passed']:reason='INSUFFICIENT_SPATIAL_PHOTO_SUPPORT'
  else:
   d=diagnostic(item['ps']['B2'],1,0,parts['fit'],images[0],images[1],item['views']);logs[arm+'_observability']=d
   if d['source_gradient_RMS']<protocol['texture_gate']['source_FIT_RGB_gradient_RMS_min']:reason='LOW_TEXTURE'
   elif not fixed and d['photo_plane_J_singular_ratio_after_brightness']<protocol['texture_gate']['plane_Jacobian_after_gain_bias_projection_singular_ratio_min']:reason='PHOTO_PLANE_J_DEGENERATE'
   else:best,log=solve_photo(item,parts['fit'],images,fixed,protocol)
   if best is None and reason is None:reason='PHOTO_SOLVER_FAILED'
  logs[arm]=log
  if best is None:p=item['ps']['B2'];g,b=brightness(p);status='FALLBACK_B2'
  else:p=np.array(best['p']);g,b=best['gain'],best['bias'];status='PHOTO_REFINED'
  metrics.append(save_arm(out,arm,p,g,b,item,images,parts,status,reason))
 dump(out/'solver_logs.json',logs);dump(out/'metrics_noGT.json',metrics);dump(out/'manifest.json',{'status':'P9_FIT_COMPLETED_GT_NOT_READ','photo_support_passed':sup['passed'],'GT_read':False,'raw_preserved':True,'non_target_unchanged':True,'residual_roundtrip':True})
 return {'dataset':item['row']['dataset'],'sample':item['row']['sample'],'case_dir':str(out.relative_to(ROOT)),'status':'FIT_COMPLETED_GT_NOT_READ','support':sup,'arms':[{k:m[k] for k in ['arm','status','reason']} for m in metrics]}

def main():
 here=Path(__file__).parent;protocol=json.loads((here/'protocol.json').read_text());out=create('all-cases-P9fit');(out/'source').mkdir()
 for f in ['run_fit.py','photo.py','protocol.json','check_photo.py']:shutil.copy2(here/f,out/'source'/f)
 shutil.copy2(here.parent/'P5_P8/common.py',out/'source/P5_P8_common.py');shutil.copy2(here.parent/'P3/solver.py',out/'source/P3_solver.py');dump(out/'config_resolved.json',protocol);dump(out/'manifest.json',{'status':'RUNNING','GT_read':False,'run_id':out.name})
 parent=BASE/'P4-all-evaluation-20261117-001';rows=[{'dataset':'kitti','sample':f'{i:06}','evaluation_dir':str((parent/'cases'/f'kitti-{i:06}').relative_to(ROOT)),'mask_provenance':'HUMAN_REVIEWED'} for i in range(5,8)];oldbatch=BASE/'P3batch-all-evaluation-20261117-001'
 for r in json.loads((oldbatch/'summary.json').read_text()):
  if r['status']=='FIT_COMPLETED_GT_NOT_READ':rows.append({'dataset':r['dataset'],'sample':r['sample'],'evaluation_dir':str((parent/'cases'/(r['dataset']+'-'+r['sample'])).relative_to(ROOT)),'mask_provenance':r['mask_provenance']})
 result=[]
 for row in rows:
  outcase=out/'cases'/(row['dataset']+'-'+row['sample'])
  try:r=fit_case(load_case(row),outcase,protocol)
  except Exception as e:outcase.mkdir(parents=True,exist_ok=True);(outcase/'error.log').write_text(traceback.format_exc());r={**row,'status':'FIT_FAILED','reason':str(e)};dump(outcase/'manifest.json',r)
  result.append(r);dump(out/'summary.json',result);print(row['dataset'],row['sample'],r['status'],r.get('support',{}).get('counts'),r.get('reason',''),flush=True)
 dump(out/'carried_unsupported_cases.json',[r for r in json.loads((oldbatch/'summary.json').read_text()) if r['status']!='FIT_COMPLETED_GT_NOT_READ']);dump(out/'manifest.json',{'status':'ALL_P9_FITS_COMPLETED_GT_NOT_READ','run_id':out.name,'cases':len(rows),'fit_failures':sum(r['status']=='FIT_FAILED' for r in result),'GT_read':False,'scope':'previously analyzed exploratory stereo cases'});print(out.name,flush=True)
if __name__=='__main__':main()
