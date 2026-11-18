"""Post-fit P5-P8 GT diagnostics and paired evaluation; never changes fitting or groups."""
import sys,json,shutil
from pathlib import Path
import numpy as np
from common import *

def metrics(p,cor,baseline,mask,gt,pg,view,obs):
 valid=mask&np.isfinite(gt)&(gt>0);h,w=mask.shape;y,x=np.mgrid[:h,:w];uv=np.c_[x.ravel(),y.ravel()];err=cor[valid]-gt[valid];n=p/np.linalg.norm(p);X=np.array(view['C'])+cor[mask,None]*rays(uv[mask.ravel()],view);res=X@n+1/np.linalg.norm(p);row={'MAE_m':float(np.mean(abs(err))),'RMSE_m':float(np.sqrt(np.mean(err**2))),'AbsRel':float(np.mean(abs(err)/gt[valid])),'normal_error_deg':angle(p,pg),'own_plane_RMSE_m':float(np.sqrt(np.mean(res**2))),'valid_GT_pixels':int(valid.sum())}
 for split,oo in obs.items():
  r,_=residual_jacobian(p,oo['observations'],oo['views']);rr=np.linalg.norm(r.reshape(-1,2),axis=1);row[split+'_mean_px']=float(np.mean(rr));row[split+'_median_px']=float(np.median(rr))
 return row

def main():
 fit=BASE/sys.argv[1];manifest=json.loads((fit/'manifest.json').read_text());assert manifest['status']=='ALL_CASES_FITTED_GT_NOT_READ' and manifest['fit_failures']==0;protocol=json.loads((fit/'config_resolved.json').read_text());parent=BASE/protocol['parent_P4'];out=create('all-cases-P5-P8eval');(out/'source').mkdir()
 for f in ['evaluate.py','common.py','protocol.json']:shutil.copy2(Path(__file__).parent/f,out/'source'/f)
 dump(out/'config_resolved.json',{'parent_fit':fit.name,'protocol_sha256':sha(fit/'config_resolved.json'),'parent_P4':parent.name,'GT_for_optimization':False,'scope':'post-hoc diagnostic and fixed-rule exploratory comparison on previously analyzed data; no independent validation'})
 hashes={str(p.relative_to(fit)):sha(p) for p in fit.rglob('*') if p.is_file()};oldrows=json.loads((parent/'case_results.json').read_text());oldmap={(r['dataset'],r['sample']):r for r in oldrows};summary=[]
 for frow in json.loads((fit/'summary.json').read_text()):
  ds,s=frow['dataset'],frow['sample'];case=ROOT/frow['case_dir'];dest=out/'cases'/(ds+'-'+s);dest.mkdir(parents=True);info=json.loads((case/'case.json').read_text());views=info['views'];view=views[0];data=np.load(case/'inputs.npz');baseline=data['baseline'];mask=data['masks'][0];normals=json.loads((case/'raw_normals.json').read_text());selections=np.load(case/'selected_pixel_indices.npz');old=oldmap[(ds,s)];gtfile=ROOT/old['evaluation_dir']/'GT_arrays.npz';gt=np.load(gtfile)['GT_depth'];pg=np.array(old['reference']['GT_plane_p']);valid=np.isfinite(gt)&(gt>0)&mask;h,w=mask.shape;y,x=np.mgrid[:h,:w];uv=np.c_[x.ravel(),y.ravel()];obs0=json.loads((case/'observations.json').read_text());obs={split:{'observations':[(np.array(o['uv']),np.array(o['target_uv']),o['i'],o['j']) for o in oo],'views':views} for split,oo in obs0.items()};diagnostics={}
  for g in protocol['normal_groups']:
   ii=selections['view0_'+g];di=dict(normals[0][g]);jj=ii[valid.ravel()[ii]];di['GT_valid_selected_pixels']=len(jj)
   if len(jj):di['raw_selected_MAE_m']=float(np.mean(abs(baseline[0].ravel()[jj]-gt.ravel()[jj])))
   if di['status']=='NORMAL_ESTIMATED':
    di['normal_error_global_GT_deg']=angle(np.array(di['p']),pg)
    if len(jj)>=32:
     X=np.array(view['C'])+gt.ravel()[jj,None]*rays(uv[jj],view);pglocal=plane(X);di['normal_error_local_GT_deg']=angle(np.array(di['p']),pglocal);di['local_GT_normal_vs_global_deg']=angle(pglocal,pg)
   diagnostics[g]=di
  control=[]
  for c in json.loads((case/'P5_controls.json').read_text()):
   d={k:v for k,v in c.items() if k!='replicates'}
   if c['status']=='CONTROL_COMPUTED':
    paired=[rep for rep in c['replicates'] if all(rep[g]['status']=='NORMAL_ESTIMATED' for g in c['family'])];d['valid_paired_replicates']=len(paired);d['mean_normal_error_deg']={g:float(np.mean([angle(np.array(rep[g]['p']),pg) for rep in paired])) for g in c['family']} if paired else None
   control.append(d)
  dump(dest/'P5_diagnostics.json',diagnostics);dump(dest/'P5_controls.json',control)
  oldmetrics={m['arm']:m for m in old['metrics']};allmetrics={a:dict(m) for a,m in oldmetrics.items()};fitold=ROOT/info['parent_fit'];oldar=fitold/'arms' if (fitold/'arms').exists() else fitold/'artifacts/arms';pall=np.array(normals[0]['ALL']['p']);primary=np.load(oldar/'B2/depths.npz');primary_p=primary['p']
  # Recompute all original control depth errors on the exact same fixed distance groups.
  for a,m in allmetrics.items():
   z=np.load(oldar/a/'depths.npz');cor=z['corrected_depth'][0]
   for g in ['NEAR','MID','FAR']:
    ii=selections['view0_'+g];ii=ii[valid.ravel()[ii]];m[g+'_MAE_m']=float(np.mean(abs(cor.ravel()[ii]-gt.ravel()[ii]))) if len(ii) else None
   m['status']='ORIGINAL_CONTROL'
  obsmetrics={r['arm']:r for r in json.loads((case/'observation_metrics.json').read_text())};technical=[]
  for a,orow in obsmetrics.items():
   z=np.load(case/'arms'/(a+'.npz'));p=z['p'];cor=z['corrected_depth'];delta=z['residual_depth'];assert np.array_equal(cor[~data['masks']],baseline[~data['masks']]);assert np.allclose(baseline+delta,cor,rtol=0,atol=1e-12);assert np.isfinite(cor[data['masks']]).all() and np.all(cor[data['masks']]>0)
   m=metrics(p,cor[0],baseline[0],mask,gt,pg,view,obs);m.update(orow)
   for g in ['NEAR','MID','FAR']:
    ii=selections['view0_'+g];ii=ii[valid.ravel()[ii]];m[g+'_MAE_m']=float(np.mean(abs(cor[0].ravel()[ii]-gt.ravel()[ii]))) if len(ii) else None
   for j,v in enumerate(views):
    hh,ww=cor[j].shape;yy,xx=np.mgrid[:hh,:ww];uvj=np.c_[xx.ravel(),yy.ravel()];mm=data['masks'][j].ravel();assert np.allclose(depths(p,uvj[mm],v),cor[j].ravel()[mm])
   if a=='P8_LIMITED_RAW':assert angle(p,pall)<=protocol['P8_angle_limit_degrees']+1e-6
   allmetrics[a]=m;technical.append({'arm':a,'positive_depth':True,'non_target_unchanged':True,'residual_roundtrip':True,'shared_plane_depths_verified':True})
  dump(dest/'metrics.json',allmetrics);dump(dest/'technical_checks.json',technical);dump(dest/'GT_provenance.json',{'parent_GT_arrays':str(gtfile.relative_to(ROOT)),'GT_arrays_sha256':sha(gtfile),'GT_plane_p':pg.tolist(),'GT_reference_role':'evaluation only; existing P4 reference reused','source_fit':str(case.relative_to(ROOT))})
  row={'dataset':ds,'sample':s,'evaluation_dir':str(dest.relative_to(ROOT)),'fit_dir':str(case.relative_to(ROOT)),'P5':diagnostics,'P5_controls':control,'metrics':allmetrics};summary.append(row);dump(out/'summary.json',summary);print(ds,s,'EVALUATED',flush=True)
 assert all(sha(fit/p)==h for p,h in hashes.items());dump(out/'fit_hashes.json',hashes);dump(out/'manifest.json',{'run_id':out.name,'status':'P5_P8_EVALUATION_COMPLETED','parent_fit':fit.name,'GT_evaluated':len(summary),'fit_outputs_unchanged':True,'carried_unsupported_cases':8,'independent_validation':False});print(out.name,flush=True)
if __name__=='__main__':main()
