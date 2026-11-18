"""P5 RGB/raw-only diagnostic proposals, P6/P7/P8 FIT-only refinements. No GT I/O."""
import sys,json,shutil,traceback
from pathlib import Path
import numpy as np,cv2
from scipy.optimize import minimize,LinearConstraint
from common import *

def constrained_solve(p0,p2,p3,obs,views,polygons,limit):
 n0=p0/np.linalg.norm(p0);cos=np.cos(np.deg2rad(limit));AA=[];bb=[]
 for v,poly in zip(views,polygons):
  AA.extend(rays(np.array(poly),v));bb.extend([-1e-8]*len(poly));AA.append(-np.array(v['C']));bb.append(1-1e-8)
 A=np.array(AA);upper=np.array(bb)
 def cone(p):return float(p@n0-cos*np.linalg.norm(p))
 def grad(p):return n0-cos*p/max(1e-15,np.linalg.norm(p))
 starts=[('B2',p2),('raw_B1',p0)]
 if angle(p0,p3)<=limit and p0@p3>0:starts.append(('free_B3_inside_cone',p3))
 logs=[]
 for name,start in starts:
  if np.any(A@start>upper):logs.append({'seed':name,'success':False,'reason':'INITIAL_POSITIVE_DEPTH_FAILED'});continue
  opt=minimize(lambda p:objective(p,obs,views)[0],start,jac=lambda p:objective(p,obs,views)[1],method='SLSQP',constraints=[LinearConstraint(A,-np.inf,upper),{'type':'ineq','fun':cone,'jac':grad}],options={'maxiter':500,'ftol':1e-12})
  p=opt.x;ang=angle(p,p0);success=bool(opt.success and np.all(A@p<=upper+1e-9) and cone(p)>=-1e-10 and ang<=limit+1e-6)
  logs.append({'seed':name,'success':success,'message':str(opt.message),'iterations':int(opt.nit),'p':p.tolist(),'FIT_cost':float(objective(p,obs,views)[0]),'anchor_angle_deg':ang})
 good=[l for l in logs if l['success']];best=min(good,key=lambda l:l['FIT_cost']) if good else None
 return (np.array(best['p']) if best else None),logs

def controlled(gs,z,view,mask,protocol):
 hh,ww=z.shape;yy,xx=np.mgrid[:hh,:ww];uv=np.c_[xx.ravel(),yy.ravel()];alluv=uv[mask.ravel()];lo=alluv.min(0);span=alluv.max(0)-lo+1;cell=np.minimum(3,((uv-lo)*4/span).astype(int));keys=cell[:,1]*4+cell[:,0];records=[];rng=np.random.default_rng(0)
 for family in protocol['P5_count_control']['comparisons']:
  n=min(min(len(gs[g]) for g in family),protocol['P5_count_control']['max_points_per_group'])
  for mode in ['COUNT_MATCHED','SPATIAL_CELL_MATCHED']:
   pools={g:{c:gs[g][keys[gs[g]]==c] for c in range(16)} for g in family}
   cells=[c for c in range(16) if all(len(pools[g][c]) for g in family)];quota={c:min(protocol['P5_spatial_control']['max_per_common_cell'],min(len(pools[g][c]) for g in family)) for c in cells}
   total=sum(quota.values());minimum=protocol['P5_count_control']['min_points']
   if (mode=='COUNT_MATCHED' and n<minimum) or (mode=='SPATIAL_CELL_MATCHED' and (len(cells)<protocol['P5_spatial_control']['min_common_cells'] or total<minimum)):
    records.append({'family':family,'mode':mode,'status':'NOT_IDENTIFIABLE_SPATIAL_CONFOUNDING' if mode=='SPATIAL_CELL_MATCHED' else 'INSUFFICIENT_POINTS','common_cells':len(cells),'available_common_points':total});continue
   reps=[]
   for rep in range(protocol['P5_count_control']['replicates']):
    rr={}
    for g in family:
     chosen=rng.choice(gs[g],n,replace=False) if mode=='COUNT_MATCHED' else np.concatenate([rng.choice(pools[g][c],quota[c],replace=False) for c in cells])
     p,info=normal_info(chosen,z,view,mask,protocol);rr[g]=info
    reps.append(rr)
   records.append({'family':family,'mode':mode,'status':'CONTROL_COMPUTED','per_group_points':n if mode=='COUNT_MATCHED' else total,'common_cells':len(cells) if mode=='SPATIAL_CELL_MATCHED' else None,'replicates':reps})
 return records

def save_arm(out,arm,p,item,retention):
 baseline=item['baseline'];masks=item['masks'];views=item['views'];corrected=baseline.copy()
 for j,v in enumerate(views):
  h,w=baseline[j].shape;y,x=np.mgrid[:h,:w];uv=np.c_[x.ravel(),y.ravel()];m=masks[j].ravel();z=depths(p,uv[m],v);assert np.isfinite(z).all() and (z>0).all();corrected[j].ravel()[m]=z
 delta=corrected-baseline;assert np.array_equal(corrected[~masks],baseline[~masks]);assert np.allclose(baseline+delta,corrected,rtol=0,atol=1e-12)
 target=out/'arms';target.mkdir(exist_ok=True);np.savez_compressed(target/(arm+'.npz'),p=p,residual_depth=delta,corrected_depth=corrected)
 row={'arm':arm,**retention}
 for split,obs in item['obs'].items():
  r,_=residual_jacobian(p,obs,views);norm=np.linalg.norm(r.reshape(-1,2),axis=1);row[split+'_mean_px']=float(np.mean(norm));row[split+'_median_px']=float(np.median(norm))
 return row

def fit_case(item,out,protocol):
 out.mkdir(parents=True);views=item['views'];masks=item['masks'];z=item['baseline'];conf=item['conf'];ps=item['ps'];obs=item['obs']['fit'];polys=item['polygons'];row=item['row']
 dump(out/'case.json',{'dataset':row['dataset'],'sample':row['sample'],'views':views,'polygons':polys,'raw_path':str(item['rawpath'].relative_to(ROOT)),'raw_sha256':sha(item['rawpath']),'parent_fit':str(item['fit'].relative_to(ROOT)),'parent_P4_evaluation':row['evaluation_dir'],'mask_provenance':row['mask_provenance'],'GT_read':False})
 np.savez_compressed(out/'inputs.npz',raw_depth=item['raw'],depth_conf=conf,scale=item['scale'],baseline=z,masks=masks)
 dump(out/'observations.json',{s:[{'uv':u.tolist(),'target_uv':v.tolist(),'i':i,'j':j} for u,v,i,j in oo] for s,oo in item['obs'].items()})
 normals=[];selections={}
 for j,v in enumerate(views):
  gs=groups(z[j],conf[j],masks[j]);norm={};selected={}
  for g,ii in gs.items():
   p,info=normal_info(ii,z[j],v,masks[j],protocol);info['median_depth_conf']=float(np.median(conf[j].ravel()[ii])) if len(ii) else None
   if p is not None:
    dispersion,tiles=bootstrap_dispersion(ii,z[j],v,p,protocol);info.update(raw_normal_bootstrap_RMS_deg=dispersion,raw_normal_bootstrap_tiles=tiles)
   norm[g]=info;selections[f'view{j}_{g}']=ii
  normals.append(norm)
 np.savez_compressed(out/'selected_pixel_indices.npz',**selections);dump(out/'raw_normals.json',normals)
 dump(out/'P5_controls.json',controlled({g:selections['view0_'+g] for g in protocol['normal_groups']},z[0],views[0],masks[0],protocol))
 results=[];retained={};logs={}
 def fixed_arm(arm,pinitial,fallback,reason=None):
  if pinitial is None:p=None;log=[]
  else:p,log=solve(pinitial,obs,views,polys,fixed=True)
  logs[arm]={'direction_initial_p':pinitial.tolist() if pinitial is not None else None,'distance_solver':log,'precondition_failure':reason}
  success=p is not None
  if not success:p=retained.get(fallback,ps['B2'])
  retained[arm]=p
  results.append(save_arm(out,arm,p,item,{'status':'REFINED' if success else 'FALLBACK_RETAINED','fallback':None if success else fallback,'reason':None if success else reason or 'DISTANCE_SOLVER_OR_POSITIVE_DEPTH_FAILED'}))
 for arm,g in [('P6_NEAR','NEAR'),('P6_CONF','HIGH_CONF'),('P6_NEAR_CONF','NEAR_HIGH')]:
  info=normals[0][g];p=np.array(info['p']) if info['status']=='NORMAL_ESTIMATED' else None;fixed_arm(arm,p,'B2',info['status'] if p is None else None)
 for arm,g,fallback in [('P7_NEAR','NEAR','P6_NEAR'),('P7_NEAR_CONF','NEAR_HIGH','P6_NEAR_CONF')]:
  weighted=[];details=[]
  for j,norm in enumerate(normals):
   info=norm[g]
   if info['status']!='NORMAL_ESTIMATED':continue
   ii=selections[f'view{j}_{g}'];allc=np.sort(conf[j][masks[j]]);percentile=np.searchsorted(allc,conf[j].ravel()[ii],side='right')/len(allc);cp=float(np.median(percentile));sigma=max(info['raw_normal_bootstrap_RMS_deg'],.25);weight=info['hull_fraction']*cp/sigma**2;p=np.array(info['p']);n=p/np.linalg.norm(p);weighted.append((weight,n));details.append({'view':j,'weight':weight,'confidence_percentile_median':cp,'bootstrap_RMS_deg':info['raw_normal_bootstrap_RMS_deg']})
  p=None
  if len(weighted)>=protocol['P7_min_views']:
   S=sum(w*np.outer(n,n) for w,n in weighted);e,V=np.linalg.eigh(S);n=V[:,-1]
   if n@ps['B1']<0:n=-n
   # Match the original plane depth at the source ROI centroid for initialization only.
   h,w=z[0].shape;y,x=np.mgrid[:h,:w];uv=np.c_[x[masks[0]],y[masks[0]]];q=rays(uv.mean(0)[None],views[0])[0];p=n*(q@ps['B1'])/(q@n)
  fixed_arm(arm,p,fallback,'INSUFFICIENT_MULTI_VIEW_NORMAL_SUPPORT' if p is None else None);logs[arm]['view_weights']=details
 p,log=constrained_solve(ps['B1'],ps['B2'],ps['B3'],obs,views,polys,protocol['P8_angle_limit_degrees']);success=p is not None
 if not success:p=ps['B2']
 logs['P8_LIMITED_RAW']={'solver':log};assert angle(p,ps['B1'])<=protocol['P8_angle_limit_degrees']+1e-6
 results.append(save_arm(out,'P8_LIMITED_RAW',p,item,{'status':'REFINED' if success else 'FALLBACK_RETAINED','fallback':None if success else 'B2','anchor_angle_deg':angle(p,ps['B1'])}))
 dump(out/'solver_logs.json',logs);dump(out/'observation_metrics.json',results);dump(out/'manifest.json',{'status':'P5_PROPOSALS_P6_P7_P8_FIT_COMPLETED_GT_NOT_READ','dataset':row['dataset'],'sample':row['sample'],'GT_read':False,'arms':len(results),'raw_preserved':True,'non_target_unchanged':True,'residual_roundtrip':True})
 return {'dataset':row['dataset'],'sample':row['sample'],'case_dir':str(out.relative_to(ROOT)),'status':'FIT_COMPLETED_GT_NOT_READ','arms':results}

def main():
 here=Path(__file__).parent;protocol=json.loads((here/'protocol.json').read_text());out=create('all-cases-P5-P8fit');(out/'source').mkdir()
 for f in ['run_fit.py','common.py','protocol.json']:shutil.copy2(here/f,out/'source'/f)
 for f in ['solver.py','fit.py']:shutil.copy2(here.parent/'P3'/f,out/'source'/('P3_'+f))
 assert sha(here.parent/'P3/solver.py')==sha(BASE/'P2-kitti-000000-20261117-001/source/solver.py')
 dump(out/'config_resolved.json',protocol);dump(out/'manifest.json',{'run_id':out.name,'status':'RUNNING','GT_read':False});parent=BASE/protocol['parent_P4'];dump(out/'parent_hashes.json',{'P4_case_results_sha256':sha(parent/'case_results.json'),'P4_config_sha256':sha(parent/'config_resolved.json')})
 # Build routing solely from pre-GT P3 fit manifests. Do not parse GT-bearing P4 metrics containers.
 rows=[{'dataset':'kitti','sample':f'{i:06}','evaluation_dir':str((parent/'cases'/f'kitti-{i:06}').relative_to(ROOT)),'mask_provenance':'HUMAN_REVIEWED'} for i in range(5,8)]
 batch=BASE/'P3batch-all-evaluation-20261117-001'
 for r in json.loads((batch/'summary.json').read_text()):
  if r['status']=='FIT_COMPLETED_GT_NOT_READ':rows.append({'dataset':r['dataset'],'sample':r['sample'],'evaluation_dir':str((parent/'cases'/(r['dataset']+'-'+r['sample'])).relative_to(ROOT)),'mask_provenance':r['mask_provenance']})
 allrows=[]
 for row in rows:
  dest=out/'cases'/(row['dataset']+'-'+row['sample'])
  try:item=load_case(row);result=fit_case(item,dest,protocol)
  except Exception as e:
   dest.mkdir(parents=True,exist_ok=True);(dest/'error.log').write_text(traceback.format_exc());result={**row,'status':'FIT_FAILED','reason':str(e)};dump(dest/'manifest.json',result)
  allrows.append(result);dump(out/'summary.json',allrows);print(row['dataset'],row['sample'],result['status'],flush=True)
 dump(out/'carried_unsupported_cases.json',[r for r in json.loads((parent/'all_attempt_statuses.json').read_text()) if r['status']!='FIT_COMPLETED_GT_NOT_READ'])
 dump(out/'manifest.json',{'run_id':out.name,'status':'ALL_CASES_FITTED_GT_NOT_READ','cases':len(rows),'fit_failures':sum(r['status']=='FIT_FAILED' for r in allrows),'carried_unsupported_cases':8,'GT_read':False,'scope':'post-hoc exploratory extension; no independent validation'});print(out.name,flush=True)
if __name__=='__main__':main()
