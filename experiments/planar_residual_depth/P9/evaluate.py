"""P9 post-fit GT evaluation and oracle-plane image-warp diagnosis. No fit changes."""
import sys,json,shutil
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).parents[1]/'P5_P8'))
from common import *
from photo import *

def main():
 fit=BASE/sys.argv[1];mf=json.loads((fit/'manifest.json').read_text());assert mf['status']=='ALL_P9_FITS_COMPLETED_GT_NOT_READ' and mf['fit_failures']==0;assert json.loads((fit/'preGT_observability_check.json').read_text())['status']=='PASSED';out=create('all-cases-P9eval');(out/'source').mkdir()
 for name in ['evaluate.py','photo.py','protocol.json']:shutil.copy2(Path(__file__).parent/name,out/'source'/name)
 hashes={str(p.relative_to(fit)):sha(p) for p in fit.rglob('*') if p.is_file()};dump(out/'config_resolved.json',{'parent_fit':fit.name,'protocol_sha256':sha(fit/'config_resolved.json'),'GT_role':'post-fit evaluation/oracle plane brightness diagnosis only; no geometric fitting, support changes or solution selection','scope':'same already-seen exploratory stereo cases'});summary=[]
 for row in json.loads((fit/'summary.json').read_text()):
  case=ROOT/row['case_dir'];info=json.loads((case/'case.json').read_text());dest=out/'cases'/(row['dataset']+'-'+row['sample']);dest.mkdir(parents=True);data=np.load(case/'inputs.npz');masks=data['masks'];mask=masks[0];v=info['views'][0];h,w=mask.shape;y,x=np.mgrid[:h,:w];uv=np.c_[x.ravel(),y.ravel()];q=rays(uv,v);parent=ROOT/info['parent_P4_evaluation'];gdata=np.load(parent/'GT_arrays.npz');gt=gdata['GT_depth'];valid=gdata['evaluation_mask'];assert np.array_equal(valid,mask&np.isfinite(gt)&(gt>0));ref=json.loads((parent/'GT_reference.json').read_text());pg=np.array(ref['GT_plane_p']);parts=np.load(case/'photo_support.npz');images=[np.array(Image.open(ROOT/vv['processed']).convert('RGB'),float)/255 for vv in info['views']];oldmetrics=json.loads((case/'metrics_noGT.json').read_text());metrics={}
  for rr in oldmetrics:
   a=rr['arm'];z=np.load(case/'arms'/(a+'.npz'));p=z['p'];cor=z['corrected_depth'];assert np.array_equal(cor[~masks],data['baseline'][~masks]);assert np.allclose(data['baseline']+z['residual_depth'],cor,rtol=0,atol=1e-12);err=cor[0][valid]-gt[valid];Xd=np.array(v['C'])+cor[0][mask,None]*q[mask.ravel()];n=p/np.linalg.norm(p);own=Xd@n+1/np.linalg.norm(p);m={**rr,'ROI_MAE_m':float(np.mean(abs(err))),'ROI_RMSE_m':float(np.sqrt(np.mean(err**2))),'ROI_AbsRel':float(np.mean(abs(err)/gt[valid])),'normal_error_deg':angle(p,pg),'ROI_GT_valid_pixels':int(valid.sum()),'own_plane_RMSE_m':float(np.sqrt(np.mean(own**2)))}
   for split in ['fit','heldout']:
    u=parts[split];ii=u.astype(int);ok=valid[ii[:,1],ii[:,0]] if len(u) else np.array([],bool);good=ii[ok];m['photo_'+split+'_GT_valid_pixels']=len(good);m['photo_'+split+'_GT_MAE_m']=float(np.mean(abs(cor[0][good[:,1],good[:,0]]-gt[good[:,1],good[:,0]]))) if len(good) else None
    if len(u):m['photo_'+split+'_no_affine']=diagnostic(p,1,0,u,images[0],images[1],info['views'])
   metrics[a]=m
  # Oracle reference geometry is first introduced here after every fit has completed.
  oracle={};gg,bb,success=fit_brightness(pg,parts['fit'],images[0],images[1],info['views']) if len(parts['fit']) else (1.,0.,False)
  for split in ['fit','heldout']:
   u=parts[split]
   if len(u):oracle[split]=diagnostic(pg,gg,bb,u,images[0],images[1],info['views']);oracle[split]['Huber_cost']=loss_and_grad(pg,gg,bb,u,images[0],images[1],info['views'])[0]
   else:oracle[split]=None
  oracle.update(gain=gg,bias=bb,affine_FIT_success=success,GT_reference_plane_p=pg.tolist(),role='oracle diagnostic only; never used by P9 solver or support selection',GT_plane_RMSE_m=ref['GT_plane_RMSE_m']);dump(dest/'oracle_GT_plane_warp.json',oracle);dump(dest/'metrics.json',metrics);dump(dest/'GT_provenance.json',{'GT_arrays_path':str((parent/'GT_arrays.npz').relative_to(ROOT)),'GT_arrays_sha256':sha(parent/'GT_arrays.npz'),'GT_ref_path':str((parent/'GT_reference.json').relative_to(ROOT)),'GT_for_optimization':False})
  summary.append({'dataset':row['dataset'],'sample':row['sample'],'fit_dir':str(case.relative_to(ROOT)),'evaluation_dir':str(dest.relative_to(ROOT)),'support':row['support'],'metrics':metrics,'oracle':oracle});dump(out/'summary.json',summary);print(row['dataset'],row['sample'],'EVALUATED',flush=True)
 assert all(sha(fit/p)==hh for p,hh in hashes.items());dump(out/'fit_hashes.json',hashes);dump(out/'manifest.json',{'run_id':out.name,'status':'P9_POSTFIT_GT_EVALUATION_COMPLETED','parent_fit':fit.name,'cases':len(summary),'fit_outputs_unchanged':True,'GT_for_optimization':False,'independent_validation':False});print(out.name,flush=True)
if __name__=='__main__':main()
