"""Post-fit sensor-reference projection and fixed-source/future-view evaluation."""
import sys,json,time,shutil,itertools,hashlib,argparse
from pathlib import Path
import numpy as np,cv2,yaml
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'results/planar_residual_depth'
sys.path.insert(0,str(Path(__file__).parents[1]/'P3'))
from solver import rays,depths,residual_jacobian
# Import stable robust plane helper explicitly to avoid P10 fit script name collision.
import importlib.util
spec=importlib.util.spec_from_file_location('p3_fit_helpers',ROOT/'experiments/planar_residual_depth/P3/fit.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper);plane=helper.plane

def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def angle(a,b):return float(np.degrees(np.arccos(np.clip(abs(a@b/np.linalg.norm(a)/np.linalg.norm(b)),0,1))))
def masks(case):
 mm=[]
 for v,p in zip(case['images'],case['polygons']):
  im=Image.new('L',tuple(v['size']));ImageDraw.Draw(im).polygon([tuple(x) for x in p],fill=255);mm.append(np.array(im)>0)
 return np.array(mm)

def zbuffer(points,K,shape):
 h,w=shape;buf=np.full(h*w,np.inf,dtype='float32');total=0
 for start in range(0,len(points),500000):
  P=points[start:start+500000];P=P[np.isfinite(P).all(1)&(P[:,2]>1e-5)]
  if not len(P):continue
  proj=P@np.array(K,dtype='float32').T;uv=proj[:,:2]/proj[:,2:];ok=(uv[:,0]>=-.5)&(uv[:,0]<w-.5)&(uv[:,1]>=-.5)&(uv[:,1]<h-.5);P=P[ok];xy=np.rint(uv[ok]).astype('int32');idx=xy[:,1]*w+xy[:,0];np.minimum.at(buf,idx,P[:,2].astype('float32'));total+=len(P)
 buf[~np.isfinite(buf)]=np.nan;return buf.reshape(h,w),total

def load_scan(path):
 lines=[]
 with path.open('rb') as f:
  while True:
   line=f.readline().decode().strip();lines.append(line)
   if line=='end_header':break
 assert 'format ascii 1.0' in lines
 count=int(next(l for l in lines if l.startswith('element vertex ')).split()[-1]);P=np.loadtxt(path,skiprows=len(lines),usecols=(0,1,2),dtype='float32',max_rows=count);assert P.shape==(count,3);return P

def make_GT(case,out,reuse=None):
 if reuse is not None:
  previous=reuse/case['sample'];D=np.load(previous/'GT_arrays.npz')['GT_depth'];mm=masks(case);assert D.shape==mm.shape;np.savez_compressed(out/'GT_arrays.npz',GT_depth=D,valid=np.isfinite(D)&(D>0),masks=mm);dump(out/'GT_provenance.json',{'reused_projection_cache':str(previous.relative_to(ROOT)),'GT_arrays_sha256':sha(previous/'GT_arrays.npz'),'reason':'same RGB indices/poses/calibration/geometry projection; only plane masks revised','depth_point_GT_used_for_fit':False});return D,mm
 name=case['sample'];gt=[];refs=[];mask=masks(case);scan=None
 if name=='drone':
  p=ROOT/'data/planar_residual_depth/P10/source/euroc_V1_01_easy/mav0/pointcloud0/data.ply';scan=load_scan(p);config=yaml.safe_load((ROOT/case['metadata']['GT_pointcloud_sensor_yaml']).read_text());T=np.array(config['T_WR']['data'],dtype='float32').reshape(4,4);scan=scan@T[:3,:3].T+T[:3,3];scan_sha=sha(p)
 for j,v in enumerate(case['images']):
  w,h=v['size']
  if name=='indoor':
   path=ROOT/v['GT_depth_path'];raw=cv2.imread(str(path),cv2.IMREAD_UNCHANGED).astype('float32')/5000;raw[raw<=0]=np.nan;y,x=np.mgrid[:h,:w];u=np.stack([x,y,np.ones_like(x)],-1)@np.linalg.inv(v['A']).T;D=cv2.remap(raw,u[...,0].astype('float32'),u[...,1].astype('float32'),cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=float('nan'));ref={'path':str(path.relative_to(ROOT)),'sha256':sha(path),'method':'registered RGB-D /5000, nearest inverse-A, no fill','timestamp_offset_sec':v['GT_time_delta_sec']}
  elif name=='outdoor':
   path=ROOT/v['GT_velodyne_path'];P=np.fromfile(path,'float32').reshape(-1,4)[:,:3];T=np.array(case['metadata']['T_camera2_velodyne'],dtype='float32');P=P@T[:3,:3].T+T[:3,3];D,ct=zbuffer(P,v['K'],(h,w));ref={'path':str(path.relative_to(ROOT)),'sha256':sha(path),'method':'same-index calibrated Velodyne, nearest processed-pixel minimum camera-Z; no fill/motion compensation, not occlusion-perfect official depth GT','projected_points':ct}
  else:
   T=np.array(v['T_WC'],dtype='float32');P=(scan-T[:3,3])@T[:3,:3];D,ct=zbuffer(P,v['K'],(h,w));ref={'path':str(p.relative_to(ROOT)),'sha256':scan_sha,'method':'Leica room pointcloud via T_WR/world camera pose, nearest processed-pixel minimum-Z; no fill; not official dense depth benchmark','projected_points':ct}
  gt.append(D);refs.append(ref);print(name,'GT',j,int(np.sum(np.isfinite(D)&mask[j])),flush=True)
 gt=np.array(gt);np.savez_compressed(out/'GT_arrays.npz',GT_depth=gt,valid=np.isfinite(gt)&(gt>0),masks=mask);dump(out/'GT_provenance.json',{'frames':refs,'provided_pose_values_were_camera_control':True,'depth_point_GT_used_for_fit':False});return gt,mask

def mapping_observations(tracks,N):
 group={}
 for t in tracks:
  if t['split']!='heldout':continue
  oo=[o for o in t['observations'] if o['view']<N]
  for a,b in itertools.combinations(oo,2):group.setdefault((a['view'],b['view']),[]).append((a['uv'],b['uv']))
 obs=[]
 for (i,j),pp in group.items():
  u=np.array([a for a,b in pp]);v=np.array([b for a,b in pp]);obs.extend([(u,v,i,j),(v,u,j,i)])
 return obs

def projection_metrics(p,rawbase,obs,views):
 if not obs:return {'mean_px':None,'median_px':None,'pairs':0,'valid_fraction':None}
 rr=[];okay=[]
 for uv,vv,i,j in obs:
  z=depths(p,uv,views[i]) if p is not None else cv2.remap(rawbase[i].astype('float32'),uv[:,0,None].astype('float32'),uv[:,1,None].astype('float32'),cv2.INTER_LINEAR).ravel();X=views[i]['C']+z[:,None]*rays(uv,views[i]);Xj=(X-views[j]['C'])@np.array(views[j]['R']).T;H=Xj@np.array(views[j]['K']).T;pred=H[:,:2]/H[:,2:];valid=np.isfinite(pred).all(1)&(z>0)&(Xj[:,2]>0);okay.extend(valid.tolist());rr.extend(np.linalg.norm(pred-vv,axis=1).tolist())
 fraction=float(np.mean(okay));return {'mean_px':float(np.mean(rr)) if fraction==1 else None,'median_px':float(np.median(rr)) if fraction==1 else None,'pairs':len(rr),'valid_fraction':fraction}

def evaluate_case(case,fitcase,obsparent,out,reuse=None):
 out.mkdir();gt,mm=make_GT(case,out,reuse);views=case['images'];base=np.load(fitcase/'inputs.npz')['baseline'];h,w=base[0].shape;y,x=np.mgrid[:h,:w];uv=np.c_[x.ravel(),y.ravel()];mask=mm[0];valid=mask&np.isfinite(gt[0])&(gt[0]>0);n=int(valid.sum());assert n>=1;Xgt=gt[0][valid,None]*rays(uv[valid.ravel()],views[0]);area=float(cv2.contourArea(cv2.convexHull(uv[valid.ravel()].astype('float32')))/cv2.contourArea(np.float32(case['polygons'][0]))) if n>=3 else 0.;pg=plane(Xgt) if n>=12 and area>=.1 else None;ref={'source0_GT_pixels':n,'source0_GT_hull_fraction':area,'GT_plane_available':pg is not None};bootstrap=[]
 if pg is not None:
  ng=pg/np.linalg.norm(pg);ref.update(GT_plane_p=pg.tolist(),GT_plane_RMSE_m=float(np.sqrt(np.mean((Xgt@ng+1/np.linalg.norm(pg))**2))));pix=uv[valid.ravel()];key=(pix[:,1]//16)*((w+15)//16)+pix[:,0]//16;clusters=[np.flatnonzero(key==k) for k in np.unique(key)];rng=np.random.default_rng(0)
  if len(clusters)>=3:
   for _ in range(50):ii=np.concatenate([clusters[k] for k in rng.integers(0,len(clusters),len(clusters))]);bootstrap.append(plane(Xgt[ii]))
  ref.update(bootstrap_tile_pixels=16,bootstrap_tiles=len(clusters),bootstrap_replicates=len(bootstrap),GT_normal_change_p95_deg=float(np.percentile([angle(p,pg) for p in bootstrap],95)) if bootstrap else None)
 dump(out/'GT_reference_plane.json',ref);np.savez_compressed(out/'GT_plane_bootstrap.npz',p=np.array(bootstrap));tracks=json.loads((obsparent/case['sample']/'plane_LK_tracks.json').read_text());future={}
 for t in tracks:
  if t['split']!='heldout':continue
  oo={o['view']:o['uv'] for o in t['observations']}
  if 15 not in oo:continue
  for j in [16,17,18,19]:
   if j in oo:future.setdefault(j,[]).append((oo[15],oo[j]))
 futureobs=[]
 for j,pp in future.items():futureobs.append((np.array([a for a,b in pp]),np.array([b for a,b in pp]),15,j))
 metrics=[];p0=np.array(json.loads((fitcase/'raw_plane.json').read_text())['p0']);prefixsummary=json.loads((fitcase/'prefix_summary.json').read_text())
 for N in [2,4,8,16]:
  held=mapping_observations(tracks,N);pr=next(p for p in prefixsummary if p['N']==N)
  for arm in ['B0','B1','B2','B3','P8']:
   z=np.load(fitcase/f'N{N:02}'/arm/'depths.npz');p=z['p'];p=None if not np.isfinite(p).all() else p;cor=z['corrected_depth'];err=cor[0][valid]-gt[0][valid];normalp=p if p is not None else p0;futuremetrics=[]
   if p is not None:
    for j in range(16,20):
     hh,ww=mm[j].shape;yy,xx=np.where(mm[j]);uu=np.c_[xx,yy];dd=depths(p,uu,views[j]);ok=np.isfinite(dd)&(dd>0);vg=np.isfinite(gt[j][mm[j]])&(gt[j][mm[j]]>0);vv=ok&vg
     futuremetrics.append({'view':j,'GT_pixels':int(vg.sum()),'positive_depth_fraction':float(ok.mean()),'MAE_m':float(np.mean(abs(dd[vg]-gt[j][mm[j]][vg]))) if vg.sum() and ok[vg].all() else None})
   validfuture=[r['MAE_m'] for r in futuremetrics if r['MAE_m'] is not None];futuremean=float(np.mean(validfuture)) if len(validfuture)==4 else None;status=next(a for a in pr['arms'] if a['arm']==arm)['status'];row={'sequence':case['sample'],'N':N,'arm':arm,'status':status,'source0_GT_pixels':n,'source0_MAE_m':float(np.mean(abs(err))),'source0_RMSE_m':float(np.sqrt(np.mean(err**2))),'source0_AbsRel':float(np.mean(abs(err)/gt[0][valid])),'normal_error_deg':angle(normalp,pg) if pg is not None else None,'normal_reference_bootstrap_95_deg':np.percentile([angle(normalp,bp) for bp in bootstrap],[2.5,97.5]).tolist() if bootstrap else None,'mapping_track_heldout':projection_metrics(p,base,held,views),'future_track_heldout':projection_metrics(p,base,futureobs,views),'future_GT_plane_depth':futuremetrics,'future_GT_mean_MAE_m':futuremean};metrics.append(row)
 dump(out/'metrics.json',metrics);dump(out/'future_observation_summary.json',{'source_view':15,'future_views':[16,17,18,19],'track_split':'heldout only','pairs':sum(len(a) for a,b,i,j in futureobs),'scope':'future images excluded from VGGT and plane fitting; fixed16 mapping raw context already used by every prefix'});return metrics,ref

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--fit',default='P10fit-three-sequences-20261117-002');ap.add_argument('--reuse-GT');args=ap.parse_args()
 fit=BASE/args.fit;mf=json.loads((fit/'manifest.json').read_text());assert mf['failures']==0;obs=BASE/mf['observation_parent'];prefix='three-sequences-P10eval-'+time.strftime('%Y%m%d')+'-';i=1
 while (BASE/(prefix+f'{i:03}')).exists():i+=1
 out=BASE/(prefix+f'{i:03}');out.mkdir();(out/'source').mkdir();shutil.copy2(__file__,out/'source/evaluate.py');dump(out/'config_resolved.json',{'parent_fit':fit.name,'observation_parent':obs.name,'provided_pose_control':True,'depth_point_GT_role':'post-fit evaluation only','sensor_GT_policy':json.loads((fit/'config_resolved.json').read_text())['depth_GT'],'scope':'one short pilot clip per category, frozen16-frame offline RGB context; no end-to-end RGB-only or long-term SLAM claim'});hashes={str(p.relative_to(fit)):sha(p) for p in fit.rglob('*') if p.is_file()};results=[]
 for name in ['indoor','outdoor','drone']:
  case=json.loads((fit/name/'case.json').read_text());metrics,ref=evaluate_case(case,fit/name,obs,out/name,BASE/args.reuse_GT if args.reuse_GT else None);results.append({'sample':name,'metrics':metrics,'GT_reference':ref});dump(out/'summary.json',results);print(name,'evaluated',[(r['arm'],r['source0_MAE_m']) for r in metrics if r['N']==16],flush=True)
 assert all(sha(fit/p)==h for p,h in hashes.items());dump(out/'fit_hashes.json',hashes);dump(out/'manifest.json',{'run_id':out.name,'status':'THREE_TEMPORAL_SEQUENCES_EVALUATED','parent_fit':fit.name,'depth_point_GT_used_for_optimization':False,'provided_pose_control':True,'fit_outputs_unchanged':True,'independent_sequence_count_per_category':1});print(out.name,flush=True)
if __name__=='__main__':main()
