"""RGB-only case preparation/observations and fixed solver. Never reads GT."""
import sys,json,time,shutil,hashlib,traceback,itertools,platform
from pathlib import Path
import numpy as np,cv2
from PIL import Image,ImageDraw
from lk import track_plane
from observations import tracks
from solver import solve,rays,depths,residual_jacobian
from fit import plane
ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'results/planar_residual_depth'
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def create_run(stage):
 prefix=f'all-evaluation-{stage}-'+time.strftime('%Y%m%d')+'-';seq=1
 while (BASE/(prefix+f'{seq:03}')).exists():seq+=1
 out=BASE/(prefix+f'{seq:03}');out.mkdir();return out

def get_entry(c):
 ds,s=c['dataset'],c['sample']
 parent='P1-kitti2015-first20-20261117-001' if ds=='kitti' else 'P1-middlebury-all-20261117-001' if ds=='middlebury' else f'eth3d-{s}-P1-20261117-001'
 entry=next(e for e in json.loads((BASE/parent/'samples.json').read_text()) if e['sample']==s)
 rawparent=('P1-kitti2015-first20-raw-20261117-001' if int(s)<13 else 'P1-kitti2015-first20-raw-20261117-002') if ds=='kitti' else 'P1-middlebury-raw-20261117-001' if ds=='middlebury' else 'P1-eth3d-four-scenes-raw-20261117-001'
 rawpath=BASE/rawparent/'artifacts/raw'/s/'predictions.npz'
 return entry,parent,rawpath

def prepare(c,out):
 entry,parent,rawpath=get_entry(c);views=entry['images'];polys=c['polygons']
 if c['dataset']!='eth3d':assert entry['subset']=='fixed_evaluation'
 assert rawpath.exists()
 case={**c,'input_parent':parent,'original_views':views,'raw_path':str(rawpath.relative_to(ROOT)),'mask_provenance':'AGENT_RGB_DESIGNATED_USER_UNREVIEWED','human_approved':False,'GT_read':False,'raw_depth_read_during_mask_selection':False}
 # Express the provided cameras in source camera coordinates; does not alter geometry.
 R0=np.array(views[0]['R']);C0=np.array(views[0]['C']);normalized=[]
 for v in views:
  z=dict(v);z['R']=(np.array(v['R'])@R0.T).tolist();z['C']=(R0@(np.array(v['C'])-C0)).tolist();normalized.append(z)
 case['views']=normalized;case['world_to_reference']={'R':R0.tolist(),'C':C0.tolist()};dump(out/'case.json',case)
 sheet=Image.new('RGB',(512*len(views),max(v['size'][1] for v in views)+30),'white');d=ImageDraw.Draw(sheet)
 for j,v in enumerate(views):
  im=Image.open(ROOT/v['processed']).convert('RGB')
  if polys:ImageDraw.Draw(im).line([tuple(p) for p in polys[j]]+[tuple(polys[j][0])],fill='red',width=2)
  sheet.paste(im,(512*j,25));d.text((512*j+4,3),c['sample']+f' view{j} | AGENT RGB, unreviewed',fill='black')
 sheet.save(out/'RGB_masks.png');return case

def observe(case,out):
 views=case['views'];polys=case['polygons'];gray=[];Ts=[];images=[]
 for j,v in enumerate(views):
  im=Image.open(ROOT/v['source']).convert('RGB');w,h=im.size;cw=min(w,round(h*2));ch=min(h,round(w*2));left=(w-cw)//2;top=(h-ch)//2;factor=min(1,1024/max(cw,ch));nw=round(cw*factor);nh=round(ch*factor)
  im=im.crop((left,top,left+cw,top+ch)).resize((nw,nh),Image.Resampling.BICUBIC);dest=out/'matching_rgb'/f'view{j}.png';dest.parent.mkdir(exist_ok=True);im.save(dest)
  An=np.array([[nw/cw,0,(-left+.5)*nw/cw-.5],[0,nh/ch,(-top+.5)*nh/ch-.5],[0,0,1.]])
  T=np.array(v['A'])@np.linalg.inv(An);Ts.append(T);gray.append(cv2.imread(str(dest),cv2.IMREAD_GRAYSCALE));item=dict(v);item['processed']=str(dest.relative_to(ROOT));item['keypoint_to_calibrated']=T.tolist();images.append(item)
 sift=tracks(images);dump(out/'scale_tracks.json',sift)
 ts,stats=track_plane(views,gray,Ts,{'id':case['dataset']+':'+case['sample'],'polygon':polys[0]});dump(out/'all_LK_tracks.json',ts)
 selected=[]
 for t in ts:
  obs=[o for o in t['observations'] if cv2.pointPolygonTest(np.float32(polys[o['view']]),tuple(o['uv']),False)>=0]
  if len(obs)>=2 and obs[0]['view']==0:selected.append({**t,'observations':obs})
 dump(out/'tracks.json',selected)
 counts={s:sum(t['split']==s for t in selected) for s in ['fit','heldout']};uf=np.float32([t['observations'][0]['uv'] for t in selected if t['split']=='fit'])
 area=float(cv2.contourArea(cv2.convexHull(uf))/cv2.contourArea(np.float32(polys[0]))) if len(uf)>=3 else 0.
 angles=[o['parallax_deg'] for t in selected if t['split']=='fit' for o in t['observations'][1:]];par=float(np.median(angles)) if angles else None
 passed=counts['fit']>=12 and counts['heldout']>=6 and area>=.1 and par is not None and par>=1
 support={'counts':counts,'FIT_hull_area_fraction':area,'median_FIT_parallax_deg':par,'passed':passed,'per_target':stats};dump(out/'support.json',support)
 for j,v in enumerate(views):
  im=Image.open(ROOT/v['processed']).convert('RGB');d=ImageDraw.Draw(im);p=polys[j];d.line([tuple(x) for x in p]+[tuple(p[0])],fill='red',width=2)
  for t in selected:
   for o in t['observations']:
    if o['view']==j:
     x,y=o['uv'];d.ellipse((x-2,y-2,x+2,y+2),outline='lime' if t['split']=='fit' else 'orange')
  im.save(out/f'view{j}_tracks.png')
 return ts,selected,sift,support

def fit(case,out,ts,selected,sift,support):
 views=case['views'];polys=case['polygons'];rawpath=ROOT/case['raw_path'];raw=np.load(rawpath)['depth'][0,...,0];N=len(views)
 held={j:np.array([o['uv'] for t in ts if t['split']=='heldout' for o in t['observations'] if o['view']==j]) for j in range(N)}
 ratios=[];ids=[];excluded=[]
 for t in sift:
  if t['split']!='fit':continue
  obs={o['view']:np.array(o['uv']) for o in t['observations']}
  if 0 not in obs:continue
  if any(len(held[j]) and np.min(np.linalg.norm(held[j]-u,axis=1))<=3 for j,u in obs.items()):excluded.append(t['track_id']);continue
  for j in sorted(k for k in obs if k!=0):
   Ps=[];us=[]
   for k in [0,j]:
    v=views[k];R=np.array(v['R']);C=np.array(v['C']);Ps.append(np.c_[R,-R@C]);u=np.linalg.inv(v['K'])@np.r_[obs[k],1];us.append(u[:2,None])
   X=cv2.triangulatePoints(Ps[0],Ps[1],us[0],us[1]).ravel();X=X[:3]/X[3];Z=[(np.array(views[k]['R'])@(X-views[k]['C']))[2] for k in [0,j]]
   if not np.isfinite(Z).all() or min(Z)<=0:continue
   u=obs[0];val=cv2.getRectSubPix(raw[0].astype(np.float32),(1,1),tuple(map(float,u)))[0,0]
   if not np.isfinite(val) or val<=0:continue
   ratios.append(float(Z[0]/val));ids.append(t['track_id']);break
 assert len(ratios)>=12,('INSUFFICIENT_SCALE_SUPPORT',len(ratios));scale=float(np.median(ratios));baseline=raw.astype(float)*scale
 observations={}
 for split in ['fit','heldout']:
  pairs={}
  for t in selected:
   if t['split']!=split:continue
   o={a['view']:a['uv'] for a in t['observations']}
   for j in sorted(k for k in o if k!=0):pairs.setdefault(j,[]).append((o[0],o[j]))
  observations[split]=[]
  for j,pp in pairs.items():
   u=np.array([p[0] for p in pp]);v=np.array([p[1] for p in pp]);observations[split].extend([(u,v,0,j),(v,u,j,0)])
 masks=[];grids=[]
 for j,v in enumerate(views):
  h,w=baseline[j].shape;y,x=np.mgrid[:h,:w];grids.append(np.c_[x.ravel(),y.ravel()]);im=Image.new('L',(w,h));ImageDraw.Draw(im).polygon([tuple(p) for p in polys[j]],fill=255);masks.append(np.array(im)>0)
 masks=np.array(masks);p0=plane(np.array(views[0]['C'])+baseline[0][masks[0],None]*rays(grids[0][masks[0].ravel()],views[0]))
 fitobs=observations['fit'];p2,l2=solve(p0,fitobs,views,polys,fixed=True);p3,l3=solve(p0,fitobs,views,polys);dump(out/'solver.json',{'B1':p0.tolist(),'B2':l2,'B3':l3})
 dump(out/'scale.json',{'scale':scale,'FIT_ids':ids,'excluded_near_heldout_ids':excluded,'raw_sha256':sha(rawpath)})
 # Preserve baseline even if solver rejects a proposal.
 dump(out/'inputs.json',{'views':views,'polygons':polys,'observations':{s:[{'uv':u.tolist(),'target_uv':v.tolist(),'i':i,'j':j} for u,v,i,j in oo] for s,oo in observations.items()}})
 assert p2 is not None and p3 is not None,'SOLVER_FAILED'
 r,J=residual_jacobian(p3,fitobs,views);eps=1e-6;Jn=np.column_stack([(residual_jacobian(p3+np.eye(3)[k]*eps,fitobs,views)[0]-residual_jacobian(p3-np.eye(3)[k]*eps,fitobs,views)[0])/(2*eps) for k in range(3)])
 je=float(np.max(abs(J-Jn))/max(1,np.max(abs(J))));sv=np.linalg.svd(J,compute_uv=False);assert je<1e-6 and sv[-1]/sv[0]>1e-6,('NUMERICAL_GATE_FAILED',je,sv.tolist())
 rows=[]
 for arm,p in [('B0',None),('B1',p0),('B2',p2),('B3',p3)]:
  cor=baseline.copy()
  if p is not None:
   for j in range(N):cor[j][masks[j]]=depths(p,grids[j][masks[j].ravel()],views[j])
  delta=cor-baseline;assert np.isfinite(cor).all() and np.all(cor[masks]>0);assert np.array_equal(cor[~masks],baseline[~masks]);assert np.allclose(baseline+delta,cor,rtol=0,atol=1e-12)
  dest=out/'arms'/arm;dest.mkdir(parents=True);np.savez_compressed(dest/'depths.npz',raw_depth=raw,scale=scale,baseline=baseline,residual_depth=delta,corrected_depth=cor,masks=masks,p=np.full(3,np.nan) if p is None else p)
  row={'arm':arm}
  for split,oo in observations.items():
   if p is not None:res,_=residual_jacobian(p,oo,views)
   else:
    rr=[]
    for uv,vv,i,j in oo:
     z=cv2.remap(baseline[i].astype(np.float32),uv[:,0,None].astype(np.float32),uv[:,1,None].astype(np.float32),cv2.INTER_LINEAR).ravel();X=views[i]['C']+z[:,None]*rays(uv,views[i]);h=(X-views[j]['C'])@np.array(views[j]['R']).T@np.array(views[j]['K']).T;rr.append((h[:,:2]/h[:,2:]-vv).ravel())
    res=np.concatenate(rr)
   norm=np.linalg.norm(res.reshape(-1,2),axis=1);row[split+'_median_px']=float(np.median(norm));row[split+'_mean_px']=float(np.mean(norm))
  rows.append(row)
 dump(out/'observation_metrics.json',rows);dump(out/'technical_checks.json',{'analytic_J_relative_max_error':je,'singular_values':sv.tolist(),'positive_depth':True,'raw_preserved':True,'non_target_unchanged':True,'residual_roundtrip':True,'GT_read':False})
 return scale

def main():
 out=create_run('P3batch');(out/'source').mkdir();here=Path(__file__).parent
 for name in ['batch_fit.py','batch_cases.json','lk.py','observations.py','solver.py','fit.py']:shutil.copy2(here/name,out/'source'/name)
 assert sha(here/'solver.py')==sha(BASE/'P2-kitti-000000-20261117-001/source/solver.py')
 config={'case_definition':json.loads((here/'batch_cases.json').read_text()),'solver_sha256':sha(here/'solver.py'),'thresholds':{'fit':12,'heldout':6,'hull':.1,'parallax_deg':1,'LK_FB_px':.5,'epipolar_px':1,'SIFT_ratio':.75,'scale_exclusion_px':3},'source_coordinate_frame':'same provided cameras expressed in source-camera frame','mask_review_waived_by_user':True,'GT_read':False,'prior_results':'000005/6/7 preserved, not rerun; batch contains remaining fixed evaluation cases','user_instruction':'이제부터는 나의 영역 검토 없이 계속 진행해줘.'}
 dump(out/'config_resolved.json',config);dump(out/'environment.json',{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'opencv':cv2.__version__,'execution':'CPU sandbox; cached host GPU model predictions'})
 dump(out/'manifest.json',{'status':'RUNNING','run_id':out.name,'stage':'P3','GT_read':False});summary=[]
 # All RGB masks are frozen before any raw depth is read or any optimization runs.
 prepared=[]
 for c in config['case_definition']['cases']:
  dest=out/'cases'/(c['dataset']+'-'+c['sample']);dest.mkdir(parents=True);prepared.append((prepare(c,dest),dest))
 for case,dest in prepared:
  row={'dataset':case['dataset'],'sample':case['sample'],'mask_provenance':case['mask_provenance'],'case_dir':str(dest.relative_to(ROOT))}
  try:
   if case['polygons'] is None:row.update(status='NO_UNAMBIGUOUS_PLANE_RGB',reason=case['reason'])
   else:
    ts,selected,sift,support=observe(case,dest);row['support']=support
    if not support['passed']:row['status']='INSUFFICIENT_SUPPORT'
    else:row['scale']=fit(case,dest,ts,selected,sift,support);row['status']='FIT_COMPLETED_GT_NOT_READ'
  except Exception as e:
   row.update(status='FIT_OR_INPUT_FAILED',reason=str(e));(dest/'error.log').write_text(traceback.format_exc())
  dump(dest/'manifest.json',row);summary.append(row);dump(out/'summary.json',summary);print(case['dataset'],case['sample'],row['status'],flush=True)
 dump(out/'manifest.json',{'run_id':out.name,'stage':'P3','status':'ALL_REMAINING_CASES_ATTEMPTED','GT_read':False,'cases':len(summary),'counts':{s:sum(r['status']==s for r in summary) for s in sorted(set(r['status'] for r in summary))}});print(out.name,flush=True)
if __name__=='__main__':main()
