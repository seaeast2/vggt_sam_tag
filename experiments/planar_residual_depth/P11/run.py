"""Fit P11 frozen one-second RGB condition with inherited solver. No sensor depth input."""
import sys,json,shutil,importlib.util
from pathlib import Path
import numpy as np,cv2
R=Path(__file__).resolve().parents[3];B=R/'results/planar_residual_depth';D=R/'experiments/planar_residual_depth';sys.path.insert(0,str(D/'P3'))
def module(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
obs=module('obs',D/'P10/observe.py');fit=module('fit',D/'P10/fit.py');dump=obs.dump
parent=B/'P11masks-kitti-1s-20261118-002';rawparent=B/'P10raw-three-sequences-20261118-001';out=B/'P11fit-kitti-1s-20261118-001';out.mkdir(exist_ok=False);(out/'source').mkdir();shutil.copy2(__file__,out/'source/run.py');shutil.copy2(D/'P3/solver.py',out/'source/solver.py');shutil.copy2(D/'P10/fit.py',out/'source/P10_fit.py');shutil.copy2(D/'P10/observe.py',out/'source/P10_observe.py');c=json.loads((parent/'outdoor/case.json').read_text());raw=np.load(rawparent/'outdoor/predictions.npz');ext=raw['extrinsics'][0];Ks=raw['intrinsics'][0];R0=ext[0,:,:3];C0=-R0.T@ext[0,:,3];views=[]
for j,old in enumerate(c['images']):
 v={k:val for k,val in old.items() if k not in ['T_WC','GT_velodyne_path']};r=ext[j,:,:3];cc=-r.T@ext[j,:,3];v.update(K=Ks[j].tolist(),R=(r@R0.T).tolist(),C=(R0@(cc-C0)).tolist());views.append(v)
c['images']=views;c['metadata']={'dataset':c['metadata']['dataset'],'type':'outdoor','provided_pose':False};c.pop('world_reference_camera_T_WC',None);d=out/'outdoor';d.mkdir();dump(d/'case.json',c);grays=[cv2.imread(str(R/v['matching_RGB']),0) for v in views];tracks,edges=obs.sift_tracks(views,grays);dump(d/'scale_SIFT_tracks.json',tracks);dump(d/'SIFT_edges.json',edges)
# SIFT identity graph is used at 1s intervals; no synthetic points or transferred box identity.
plane=[]
for t in tracks:
 os=[o for o in t['observations'] if cv2.pointPolygonTest(np.float32(c['polygons'][o['view']]),tuple(o['uv']),False)>=0]
 if len(os)>=2:plane.append({**t,'observations':os})
dump(d/'plane_SIFT_tracks.json',plane);scale,sl=fit.scale_rgb(c,raw['depth'][0,...,0],tracks,plane);dump(d/'scale.json',{'scale':scale,**sl});base=raw['depth'][0,...,0].astype(float)*scale;masks,grids=fit.masks_for(c);np.savez_compressed(d/'inputs.npz',raw_depth=raw['depth'][0,...,0],baseline=base,scale=scale,masks=masks);p0=fit.plane(base[0][masks[0],None]*fit.rays(grids[0][masks[0].ravel()],views[0]));dump(d/'raw_plane.json',{'p0':p0.tolist()});summary=[]
for N in [2,4,8,16]:
 dest=d/f'N{N:02}';dest.mkdir();ft,ids,angles=fit.obs_for(plane,views,N,'fit');hold,hids,_=fit.obs_for(plane,views,N,'heldout');areas=[]
 for j in range(N):
  points=np.float32([o['uv'] for t in plane if t['track_id'] in set(ids) for o in t['observations'] if o['view']==j]);areas.append(float(cv2.contourArea(cv2.convexHull(points))/cv2.contourArea(np.float32(c['polygons'][j]))) if len(points)>=3 else 0.)
 par=float(np.median(angles)) if angles else None;passed=len(ids)>=12 and len(hids)>=6 and max(areas)>=.1 and par is not None and par>=1;support={'FIT_tracks':len(ids),'heldout_tracks':len(hids),'per_view_FIT_hull_fraction':areas,'median_FIT_pair_parallax_deg':par,'passed':passed};dump(dest/'support.json',support);logs={};p2=p3=p8=None
 if passed:
  p2,l2=fit.solve(p0,ft,views[:N],c['polygons'][:N],fixed=True);p3,l3=fit.solve(p0,ft,views[:N],c['polygons'][:N]);logs.update(B2=l2,B3=l3)
  if p2 is not None and p3 is not None:p8,l8=fit.constrained_solve(p0,p2,p3,ft,views[:N],c['polygons'][:N],1.);logs['P8']=l8
  if p3 is not None:
   _,J=fit.residual_jacobian(p3,ft,views);sv=np.linalg.svd(J,compute_uv=False);logs['singular_values']=sv.tolist()
   if sv[-1]/sv[0]<=1e-6:p2=p3=p8=None
 dump(dest/'solver_logs.json',logs);dump(dest/'observations.json',{'FIT_track_ids':ids,'heldout_track_ids':hids,'FIT':[{'uv':a.tolist(),'target_uv':b.tolist(),'i':i,'j':j} for a,b,i,j in ft]});arms=[]
 for arm,p in [('B0',None),('B1',p0),('B2',p2),('B3',p3),('P8',p8)]:arms.append(fit.save_arm(dest,arm,p,raw['depth'][0,...,0],scale,base,masks,grids,views,N,'RAW_BASELINE' if arm=='B0' else 'PLANE_REFINED' if p is not None else 'FALLBACK_RAW_SUPPORT_OR_SOLVER_FAILED'))
 rr={'N':N,'support':support,'arms':arms};summary.append(rr);dump(d/'prefix_summary.json',summary);print(N,support,flush=True)
cfg=json.loads((parent/'config_resolved.json').read_text());cfg.update(camera_condition='RGB-estimated K/pose, external values stripped before matching/fit',matching_change='mutual SIFT track graph within per-view road masks, replacing short-step LK; inherited ratio .75, epipolar1 processed px, hash70/30 split',raw_parent=rawparent.name,mask_parent=parent.name);dump(out/'config_resolved.json',cfg);dump(out/'manifest.json',{'status':'FIT_COMPLETED','GT_read':False,'provided_pose_used':False,'raw_parent':rawparent.name,'plane_tracks':len(plane),'plane_FIT':sum(t['split']=='fit' for t in plane),'plane_heldout':sum(t['split']=='heldout' for t in plane)});print(out)
