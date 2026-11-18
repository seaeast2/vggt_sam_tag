"""Post-fit GT evaluation and P4 analysis. No optimizer calls or mask changes."""
import sys,json,time,hashlib,shutil,traceback,platform
from pathlib import Path
import numpy as np,cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'results/planar_residual_depth'
sys.path.insert(0,str(Path(__file__).parents[1]/'P3'))
from solver import rays
from fit import plane
from batch_fit import dump,sha,create_run

def pfm(path):
 with path.open('rb') as f:
  assert f.readline().strip()==b'Pf';line=f.readline()
  while line.startswith(b'#'):line=f.readline()
  w,h=map(int,line.split());scale=float(f.readline());a=np.fromfile(f,'<f4' if scale<0 else '>f4').reshape(h,w)
 return np.flipud(a) # Official Middlebury convention: magnitude does not scale disparity.

def load_gt(case,shape):
 ds,s=case['dataset'],case['sample'];v=case['views'][0];A=np.array(v['A']);Ko=np.linalg.inv(A)@np.array(v['K']);h,w=shape;y,x=np.mgrid[:h,:w];uv=np.c_[x.ravel(),y.ravel()]
 if ds=='kitti':
  path=ROOT/f'data/planar_residual_depth/kitti2015/source/training/disp_occ_0/{s}_10.png';disp=cv2.imread(str(path),cv2.IMREAD_UNCHANGED).astype(float)/256
  K1=np.linalg.inv(np.array(case['views'][1]['A']))@np.array(case['views'][1]['K']);assert np.allclose(K1,Ko)
  dc=np.array(case['views'][1]['C'])-v['C'];yy,xx=np.mgrid[:disp.shape[0],:disp.shape[1]];rx=(xx-Ko[0,2])/Ko[0,0];valid=np.isfinite(disp)&(disp>0);D=np.full_like(disp,np.nan);D[valid]=Ko[0,0]*(dc[0]-rx[valid]*dc[2])/disp[valid]+dc[2];method='exact calibrated rectified Z including Bz'
 elif ds=='middlebury':
  path=(ROOT/v['source']).parent/'disp0GT.pfm';disp=pfm(path);cal=json.loads((BASE/case['input_parent']/'artifacts/inputs'/s/'calibration.json').read_text());valid=np.isfinite(disp)&(disp>0)&(disp+cal['doffs']>0);D=np.full(disp.shape,np.nan);D[valid]=Ko[0,0]*(cal['baseline']/1000)/(disp[valid]+cal['doffs']);method='Middlebury official PFM sign-only endian; Z=fx*baseline_m/(d+doffs)'
 else:raise RuntimeError('ETH3D GT adapter not exercised: cases failed observation gates; no distorted GT treated as undistorted')
 xy=np.c_[uv,np.ones(len(uv))]@np.linalg.inv(A).T;gt=cv2.remap(D.astype(np.float32),xy[:,0].reshape(shape).astype(np.float32),xy[:,1].reshape(shape).astype(np.float32),cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=float('nan'))
 return gt,{'GT_path':str(path.relative_to(ROOT)),'GT_sha256':sha(path),'conversion':method,'sampling':'nearest, no missing value fill','GT_for_optimization':False}

def angle(a,b):return float(np.degrees(np.arccos(np.clip(abs(a@b/np.linalg.norm(a)/np.linalg.norm(b)),0,1))))
def evaluate(case,fitdir,out,gt_preloaded=None):
 out.mkdir(parents=True,exist_ok=False);inputs=json.loads((fitdir/'inputs.json').read_text()) if (fitdir/'inputs.json').exists() else json.loads((fitdir/'artifacts/inputs.json').read_text());views=case['views'];v=views[0]
 armroot=fitdir/'arms' if (fitdir/'arms').exists() else fitdir/'artifacts/arms';arms={a:np.load(armroot/a/'depths.npz') for a in ['B0','B1','B2','B3']}
 hashes={str(p.relative_to(fitdir)):sha(p) for p in fitdir.rglob('*') if p.is_file() and p.suffix=='.npz'}
 shape=arms['B0']['baseline'][0].shape;h,w=shape;y,x=np.mgrid[:h,:w];uv=np.c_[x.ravel(),y.ravel()];q=rays(uv,v);mask=arms['B0']['masks'][0]
 if gt_preloaded is None:gt,ref=load_gt(case,shape)
 else:gt=np.load(gt_preloaded)['GT_depth'];ref={'GT_arrays_path':str(gt_preloaded.relative_to(ROOT)),'sha256':sha(gt_preloaded),'conversion':'existing exact P3e source-camera GT cache'}
 valid=mask&np.isfinite(gt)&(gt>0);assert valid.sum()>=12
 X=np.array(v['C'])+gt[valid,None]*q[valid.ravel()];pg=plane(X);ng=pg/np.linalg.norm(pg);gtres=X@ng+1/np.linalg.norm(pg)
 gtarea=float(cv2.contourArea(cv2.convexHull(uv[valid.ravel()].astype(np.float32)))/cv2.contourArea(np.float32(case['polygons'][0])))
 # Spatial block bootstrap of GT reference, preserving within-tile correlations.
 coords=uv[valid.ravel()];tiles=coords//16;tilekey=tiles[:,1]*((w+15)//16)+tiles[:,0];unique=np.unique(tilekey);clusters=[np.flatnonzero(tilekey==t) for t in unique];rng=np.random.default_rng(0);boot=[]
 ps={a:(plane(np.array(v['C'])+arms[a]['corrected_depth'][0][mask,None]*q[mask.ravel()]) if a=='B0' else arms[a]['p']) for a in arms}
 for _ in range(100):
  ii=np.concatenate([clusters[k] for k in rng.integers(0,len(clusters),len(clusters))]);pb=plane(X[ii]);boot.append([angle(pb,pg),angle(ps['B2'],pb),angle(ps['B3'],pb)])
 boot=np.array(boot);deltaci=np.percentile(boot[:,2]-boot[:,1],[2.5,97.5]);ref.update({'GT_valid_ROI_pixels':int(valid.sum()),'GT_hull_fraction':gtarea,'GT_plane_p':pg.tolist(),'GT_plane_RMSE_m':float(np.sqrt(np.mean(gtres**2))),'GT_plane_residual_p95_m':float(np.percentile(abs(gtres),95)),'GT_bootstrap':'100 spatial 16x16-pixel block resamples seed0; conditional reference uncertainty, not independent sensor noise','GT_bootstrap_tiles':len(unique),'GT_normal_change_p95_deg':float(np.percentile(boot[:,0],95)),'B3_minus_B2_normal_error_bootstrap_95_deg':deltaci.tolist()})
 obsfile=fitdir/'observation_metrics.json' if (fitdir/'observation_metrics.json').exists() else fitdir/'evaluation/observation_metrics.json';obs={r['arm']:r for r in json.loads(obsfile.read_text())};rows=[]
 for a,z in arms.items():
  d=z['corrected_depth'][0];assert np.array_equal(d[~mask],z['baseline'][0][~mask]);assert np.array_equal(z['baseline'],z['raw_depth'].astype(float)*float(z['scale']));assert np.allclose(z['baseline']+z['residual_depth'],z['corrected_depth'],rtol=0,atol=1e-12)
  Xd=np.array(v['C'])+d[mask,None]*q[mask.ravel()];p=ps[a];n=p/np.linalg.norm(p);planeres=Xd@n+1/np.linalg.norm(p);err=d[valid]-gt[valid];togt=Xd@ng+1/np.linalg.norm(pg)
  center=np.mean(uv[mask.ravel()],axis=0)[None];qcenter=rays(center,v)[0];zcorr=-(1+np.array(v['C'])@p)/(qcenter@p);zgt=-(1+np.array(v['C'])@pg)/(qcenter@pg)
  row={'arm':a,'MAE_m':float(np.mean(abs(err))),'RMSE_m':float(np.sqrt(np.mean(err**2))),'AbsRel':float(np.mean(abs(err)/gt[valid])),'bias_m':float(np.mean(err)),'normal_error_deg':angle(p,pg),'own_plane_RMSE_m':float(np.sqrt(np.mean(planeres**2))),'thickness_95_m':float(np.percentile(planeres,97.5)-np.percentile(planeres,2.5)),'prediction_to_GT_plane_mean_abs_m':float(np.mean(abs(togt))),'reference_center_depth_offset_m':float(zcorr-zgt),**obs[a]};rows.append(row)
 b={r['arm']:r for r in rows};h2=(b['B3']['MAE_m']<b['B2']['MAE_m'] and b['B3']['normal_error_deg']<b['B2']['normal_error_deg'] and b['B3']['heldout_mean_px']<=b['B2']['heldout_mean_px'])
 decision={'H1_case':b['B1']['MAE_m']>b['B0']['MAE_m'] and b['B1']['thickness_95_m']<b['B0']['thickness_95_m'],'H2_nominal_joint_criterion_passed':h2,'H2_reference_direction_improvement_resolved':bool(deltaci[1]<0),'H2_verdict':'JOINT_IMPROVEMENT_REFERENCE_RESOLVED' if h2 and deltaci[1]<0 else 'NOMINAL_IMPROVEMENT_REFERENCE_UNCERTAIN' if h2 else 'JOINT_CRITERIA_NOT_MET','B3_minus_B2_MAE_m':b['B3']['MAE_m']-b['B2']['MAE_m']}
 np.savez_compressed(out/'GT_arrays.npz',GT_depth=gt,GT_valid=np.isfinite(gt)&(gt>0),ROI_mask=mask,evaluation_mask=valid,GT_reference_bootstrap=boot);dump(out/'GT_reference.json',ref);dump(out/'metrics.json',rows);dump(out/'decision.json',decision)
 # Common scale for all arms; missing GT remains gray, never interpolated.
 rgb=np.array(Image.open(ROOT/v['processed']));vals=np.concatenate([gt[valid]]+[z['corrected_depth'][0][mask] for z in arms.values()]);lo,hi=np.percentile(vals,[1,99]);bound=max(1e-6,float(np.percentile(np.concatenate([abs(z['corrected_depth'][0][valid]-gt[valid]) for z in arms.values()]),95)));cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#999999');ecmap=plt.get_cmap('RdBu_r').copy();ecmap.set_bad('#999999')
 fig,axs=plt.subplots(2,5,figsize=(16,6),layout='constrained');axs[0,0].imshow(rgb);p=case['polygons'][0];axs[0,0].plot([z[0] for z in p]+[p[0][0]],[z[1] for z in p]+[p[0][1]],'r-');axs[0,0].set_title('RGB / '+case['mask_provenance']);axs[1,0].imshow(np.where(valid,gt,np.nan),cmap=cmap,vmin=lo,vmax=hi);axs[1,0].set_title('GT; gray: not evaluated')
 for col,(a,z) in enumerate(arms.items(),1):
  d=z['corrected_depth'][0];im=axs[0,col].imshow(np.where(mask,d,np.nan),cmap=cmap,vmin=lo,vmax=hi);axs[0,col].set_title(a+' depth');ie=axs[1,col].imshow(np.where(valid,d-gt,np.nan),cmap=ecmap,vmin=-bound,vmax=bound);axs[1,col].set_title(f"{a}: MAE {b[a]['MAE_m']*100:.2f} cm\nnormal {b[a]['normal_error_deg']:.2f} deg")
 for ax in axs.ravel():ax.axis('off')
 fig.colorbar(im,ax=axs[0,1:],label='Camera-Z m',shrink=.7);fig.colorbar(ie,ax=axs[1,1:],label='Prediction - GT m',shrink=.7);fig.suptitle(case['dataset']+' '+case['sample']);fig.savefig(out/'depth_GT_errors.png',dpi=130);plt.close(fig)
 # Residual depth plus a fixed geometric row cross-section.
 fig,axs=plt.subplots(2,4,figsize=(14,6),layout='constrained');rowy=int(np.round(np.mean(uv[mask.ravel(),1])));db=max(1e-6,float(np.percentile(np.concatenate([abs(z['residual_depth'][0][mask]) for z in arms.values()]),95)))
 for j,(a,z) in enumerate(arms.items()):
  axs[0,j].imshow(np.where(mask,z['residual_depth'][0],np.nan),cmap=ecmap,vmin=-db,vmax=db);axs[0,j].set_title(a+' residual m');axs[0,j].axis('off');xx=np.arange(w);axs[1,j].plot(xx,np.where(mask[rowy],z['corrected_depth'][0][rowy],np.nan),label=a);axs[1,j].plot(xx,np.where(valid[rowy],gt[rowy],np.nan),'.',label='GT',markersize=2);axs[1,j].set_ylim(lo,hi);axs[1,j].legend();axs[1,j].set_title(f'row y={rowy}; same range')
 fig.savefig(out/'residual_cross_section.png',dpi=130);plt.close(fig)
 assert all(sha(fitdir/p)==h for p,h in hashes.items());dump(out/'manifest.json',{'status':'GT_EVALUATION_COMPLETED','fitdir':str(fitdir.relative_to(ROOT)),'fit_NPZ_hashes':hashes,'fit_outputs_unchanged':True,'GT_for_optimization':False})
 return {'dataset':case['dataset'],'sample':case['sample'],'mask_provenance':case['mask_provenance'],'evaluation_dir':str(out.relative_to(ROOT)),'metrics':rows,'reference':ref,'decision':decision}

def main():
 batch=BASE/sys.argv[1];assert json.loads((batch/'manifest.json').read_text())['status']=='ALL_REMAINING_CASES_ATTEMPTED'
 out=create_run('P4');(out/'source').mkdir()
 for name in ['evaluate.py']:shutil.copy2(Path(__file__).parent/name,out/'source'/name)
 for name in ['solver.py','fit.py','batch_fit.py']:shutil.copy2(Path(__file__).parents[1]/'P3'/name,out/'source'/name)
 config={'parent_batch':batch.name,'GT_reference_bootstrap':{'blocks_processed_pixels':16,'replicates':100,'seed':0},'scene_ID_bootstrap':{'replicates':10000,'seed':0,'warning':'nominal ID-resampling interval; sequence/physical scene independence unverified'},'primary':'B3 minus B2 per-case paired change, datasets separate','mask_groups':'human reviewed prior cases versus agent RGB designated unreviewed','H2_joint_rule':'lower MAE and normal error plus no larger heldout mean reprojection; reference-resolved only if paired reference bootstrap upper delta normal <0','GT_for_optimization':False}
 dump(out/'config_resolved.json',config);dump(out/'manifest.json',{'run_id':out.name,'stage':'P4','status':'RUNNING'});allrows=[];statuses=json.loads((batch/'summary.json').read_text())
 # Include the three previously completed predefined evaluation IDs without rerunning fitting.
 for s,candidate in [('000005','K02'),('000006','K03'),('000007','K04')]:
  fitdir=BASE/f'kitti-{s}-P3-20261117-001';obsdir=BASE/f'kitti-{s}-P3obs-20261117-001';case=json.loads((obsdir/'case.json').read_text());case.update(dataset='kitti',mask_provenance='HUMAN_REVIEWED',input_parent=case['input_parent']);dest=out/'cases'/('kitti-'+s)
  rr=evaluate(case,fitdir,dest,BASE/f'kitti-{s}-P3e-20261117-001/evaluation/GT_arrays.npz');allrows.append(rr);print('evaluated prior',s,flush=True)
 for row in statuses:
  if row['status']!='FIT_COMPLETED_GT_NOT_READ':continue
  fitdir=ROOT/row['case_dir'];case=json.loads((fitdir/'case.json').read_text());dest=out/'cases'/(case['dataset']+'-'+case['sample'])
  try:rr=evaluate(case,fitdir,dest);allrows.append(rr);row['evaluation_status']='GT_EVALUATED';print('evaluated',case['dataset'],case['sample'],flush=True)
  except Exception as e:row['evaluation_status']='GT_EVALUATION_FAILED';row['evaluation_reason']=str(e);dest.mkdir(parents=True,exist_ok=True);(dest/'error.log').write_text(traceback.format_exc());print('EVAL FAILED',case['sample'],str(e),flush=True)
  dump(out/'case_results.json',allrows);dump(out/'all_attempt_statuses.json',statuses)
 dump(out/'case_results.json',allrows);dump(out/'all_attempt_statuses.json',statuses)
 aggregate={}
 for ds in ['kitti','middlebury','eth3d']:
  rs=[r for r in allrows if r['dataset']==ds];attempted=15 if ds=='kitti' else 12 if ds=='middlebury' else 2;d={'predefined_cases':attempted,'GT_evaluated':len(rs),'non_evaluated':attempted-len(rs),'review_groups':{k:sum(r['mask_provenance']==k for r in rs) for k in sorted(set(r['mask_provenance'] for r in rs))}}
  if rs:
   mats=np.array([[next(m['MAE_m'] for m in r['metrics'] if m['arm']==a) for a in ['B0','B1','B2','B3']] for r in rs]);delta=mats[:,3]-mats[:,2];rng=np.random.default_rng(0);boot=delta[rng.integers(0,len(rs),(10000,len(rs)))].mean(1)
   d.update(mean_case_MAE_m=dict(zip(['B0','B1','B2','B3'],mats.mean(0).tolist())),median_case_MAE_m=dict(zip(['B0','B1','B2','B3'],np.median(mats,axis=0).tolist())),B3_minus_B2_mean_MAE_m=float(delta.mean()),B3_minus_B2_nominal_ID_bootstrap_95_m=np.percentile(boot,[2.5,97.5]).tolist(),B3_lower_MAE_cases=int(sum(delta<0)),B2_lower_MAE_cases=int(sum(delta>0)),H1_cases=sum(r['decision']['H1_case'] for r in rs),H2_nominal_joint_cases=sum(r['decision']['H2_nominal_joint_criterion_passed'] for r in rs),H2_reference_resolved_cases=sum(r['decision']['H2_verdict']=='JOINT_IMPROVEMENT_REFERENCE_RESOLVED' for r in rs))
  aggregate[ds]=d
 dump(out/'aggregate.json',aggregate);dump(out/'manifest.json',{'run_id':out.name,'stage':'P4','status':'GT_EVALUATION_AND_ANALYSIS_COMPLETED','parent_batch':batch.name,'GT_evaluated':len(allrows),'evaluation_failures':sum(r.get('evaluation_status')=='GT_EVALUATION_FAILED' for r in statuses),'GT_for_optimization':False});print(json.dumps({'run_id':out.name,'aggregate':aggregate},indent=2),flush=True)
if __name__=='__main__':main()
