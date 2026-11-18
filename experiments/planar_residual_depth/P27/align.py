"""Compare wall-plane baselines and image-driven alignment with fixed cameras."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

p=argparse.ArgumentParser()
p.add_argument('run',type=Path)
p.add_argument('--input-dir',type=Path,help='Reuse an existing frozen input and SAM mask in a new output directory')
args=p.parse_args()
out=args.run
input_dir=args.input_dir or out
if args.input_dir:
    out.mkdir(parents=True)
with np.load(input_dir/'input.npz') as data:
    raw,rgb,confidence,keep,cameras=[data[k] for k in ('xyz','rgb','confidence','keep','cameras')]
mask=np.load(input_dir/'sam_mask.npy')
# The patterned board clips the right-hand panel in frame 15.
mask[15,62:,354:]=False
n,h,w=mask.shape
active=np.flatnonzero(mask.any(axis=(1,2)))
yy,xx=np.indices((h,w))
uv=np.stack([xx,yy,np.ones_like(xx)],-1)
hold=((xx//12+3*(yy//12))%5)==0
patches={i:cv2.erode(mask[i].astype('uint8'),np.ones((3,3),np.uint8)).astype(bool)&keep[i] for i in active}
applicable=np.zeros_like(mask)
for i in active:
    applicable[i]=patches[i]
center=np.array([-np.linalg.solve(P[:,:3],P[:,3]) for P in cameras])
rays=np.array([uv@np.linalg.inv(P[:,:3]).T for P in cameras])

def fit(points,ref=None):
    use=np.ones(len(points),dtype=bool)
    for _ in range(4):
        anchor=points[use].mean(0)
        _,s,v=np.linalg.svd(points[use]-anchor,full_matrices=False)
        normal=v[-1]
        d=float(anchor@normal)
        err=abs(points@normal-d)
        use=err<=max(np.quantile(err,.80),1e-7)
    assert s[1]/np.sqrt(use.sum())>.003
    if ref is not None and normal@ref<0:
        normal,d=-normal,-d
    return normal,d

samples={}
planes={}
for i in active:
    support=patches[i]&~hold
    threshold=np.median(confidence[i][support])
    support &= confidence[i]>=threshold
    pts=raw[i][support]
    assert len(pts)>=150
    samples[i]=pts[np.linspace(0,len(pts)-1,min(750,len(pts)),dtype=int)]
    normal,d=fit(samples[i],None if not planes else next(iter(planes.values()))[0])
    planes[i]=(normal,d)
# Equal influence for the four views.
normal_avg=np.mean([a for a,d in planes.values()],0)
normal_avg/=np.linalg.norm(normal_avg)
shared_normal={i:(normal_avg,float(np.median(samples[i]@normal_avg))) for i in active}
global_normal,global_d=fit(np.concatenate(list(samples.values())),normal_avg)
shared={i:(global_normal,global_d) for i in active}

def intersect(i,coords,plane):
    a,d=plane
    q=np.c_[coords,np.ones(len(coords))]@np.linalg.inv(cameras[i,:,:3]).T
    z=(d-center[i]@a)/(q@a)
    return center[i]+z[:,None]*q,z

def project(i,xyz):
    p=xyz@cameras[i,:,:3].T+cameras[i,:,3]
    return p[:,:2]/p[:,2,None],p[:,2]

def inside(i,pts,margin=3):
    x,y=np.rint(pts).astype(int).T
    good=(x>=margin)&(x<w-margin)&(y>=margin)&(y<h-margin)
    good[good] &= patches[i][y[good],x[good]]
    return good

gray=[cv2.cvtColor(a,cv2.COLOR_RGB2GRAY) for a in rgb]
pairs=[]
for i,j in zip(active[:-1],active[1:]):
    xy=cv2.goodFeaturesToTrack(gray[i],160,.008,4,mask=patches[i].astype('uint8'))
    if xy is None:
        continue
    xy=xy.reshape(-1,2)
    X,z=intersect(i,xy,shared[i])
    initial,t=project(j,X)
    eligible=inside(j,initial)&(t>0)&(z>0)
    xy,initial=xy[eligible],initial[eligible]
    if not len(xy):
        continue
    a,b=xy.astype('float32').reshape(-1,1,2),initial.astype('float32').reshape(-1,1,2)
    b,ok,_=cv2.calcOpticalFlowPyrLK(gray[i],gray[j],a,b,winSize=(13,13),maxLevel=2,flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    aa,back,_=cv2.calcOpticalFlowPyrLK(gray[j],gray[i],b,a.copy(),winSize=(13,13),maxLevel=2,flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    b,aa=b.reshape(-1,2),aa.reshape(-1,2)
    ok=ok.ravel().astype(bool)&back.ravel().astype(bool)&inside(j,b)&(np.linalg.norm(aa-xy,axis=1)<1)
    ok &= np.linalg.norm(b-initial,axis=1)<10
    for x,t in zip(xy[ok],b[ok]):
        u,v=np.rint(x).astype(int)
        pairs.append((int(i),int(j),x,t,bool(hold[v,u])))
train=[r for r in pairs if not r[4]]
test=[r for r in pairs if r[4]]
def residual(ps,rows):
    return np.concatenate([project(j,intersect(i,x[None],ps[i])[0])[0].ravel()-y
                           for i,j,x,y,_ in rows]) if rows else np.empty(0)

arms={'per_view_plane':planes,'common_normal':shared_normal,'shared_plane':shared}
rgb_fit=dict(status='INSUFFICIENT_RGB_SUPPORT',fit_pairs=len(train),heldout_pairs=len(test))
if len(train)>=8 and len({(i,j) for i,j,*_ in train})>=2:
    guide=np.eye(3)[np.argmin(abs(global_normal))]
    tangent=np.cross(global_normal,guide)
    tangent/=np.linalg.norm(tangent)
    tangent_b=np.cross(global_normal,tangent)
    pivot=np.concatenate(list(samples.values())).mean(0)
    def plane_at(v):
        n=global_normal+v[1]*tangent+v[2]*tangent_b
        n/=np.linalg.norm(n)
        d=global_d+v[0]+(n-global_normal)@pivot
        return {i:(n,d) for i in active}
    step=.10*np.median([abs(global_d-center[i]@global_normal) for i in active])
    cap=np.tan(np.deg2rad(3))/np.sqrt(2)
    for key,angle in [('rgb_distance',0.),('rgb_direction_distance',cap)]:
        def transform(x):
            return plane_at([x[0],0.,0.] if angle==0 else x)
        bounds=([-step],[step]) if angle==0 else ([-step,-angle,-angle],[step,angle,angle])
        result=least_squares(lambda x:residual(transform(x),train),np.zeros(len(bounds[0])),
                             bounds=bounds,loss='huber',f_scale=1,max_nfev=60)
        if result.success:
            arms[key]=transform(result.x)
    rgb_fit['status']='FIT_COMPLETED' if 'rgb_distance' in arms else 'SOLVER_FAILED'

def texture_features(image):
    lab=cv2.cvtColor(image.astype('float32')/255,cv2.COLOR_RGB2LAB)
    gray=lab[...,0]
    local=gray-cv2.GaussianBlur(gray,(9,9),2)
    # Include luminance detail and chroma, without matching a constant brightness offset.
    return np.stack([local/5,lab[...,1]/15,lab[...,2]/15],-1)

features=[texture_features(image) for image in rgb]
direct_rows=[]
for i in active:
    texture=np.abs(features[i][...,0])
    for j in active:
        if i==j:
            continue
        for is_test in (False,True):
            y,x=np.where(patches[i] & (hold==is_test) & (texture>.10))
            coords=np.c_[x,y].astype(float)
            X,z=intersect(i,coords,shared[i])
            xy,t=project(j,X)
            ok=inside(j,xy)&(z>0)&(t>0)
            coords,y,x=coords[ok],y[ok],x[ok]
            if len(coords)<24:
                continue
            ids=np.linspace(0,len(coords)-1,min(len(coords),300),dtype=int)
            direct_rows.append((int(i),int(j),coords[ids],features[i][y[ids],x[ids]],is_test))

def direct_error(params,rows):
    output=[]
    for i,j,xy,col,_ in rows:
        world,z=intersect(i,xy,params[i])
        uv,t=project(j,world)
        good=inside(j,uv)&(z>0)&(t>0)
        f=cv2.remap(features[j],uv[:,0].astype('float32')[None],
                    uv[:,1].astype('float32')[None],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)[0]
        err=abs(f-col).mean(1).clip(0,2)
        err[~good]=2
        output.append(err)
    return np.concatenate(output) if output else np.empty(0)

direct_train=[r for r in direct_rows if not r[-1]]
direct_test=[r for r in direct_rows if r[-1]]
direct_log=dict(status='INSUFFICIENT_SUPPORT',fit_pixels=sum(len(r[2]) for r in direct_train),
                heldout_pixels=sum(len(r[2]) for r in direct_test))
if len(direct_train)>=2 and direct_log['fit_pixels']>=100:
    guide=np.eye(3)[np.argmin(abs(global_normal))]
    tangent=np.cross(global_normal,guide)
    tangent/=np.linalg.norm(tangent)
    tangent_b=np.cross(global_normal,tangent)
    pivot=np.concatenate(list(samples.values())).mean(0)
    step=.10*np.median([abs(global_d-center[i]@global_normal) for i in active])
    angle=np.tan(np.deg2rad(3))/np.sqrt(2)
    def direct_plane(x):
        n=global_normal+x[1]*tangent+x[2]*tangent_b
        n/=np.linalg.norm(n)
        d=global_d+x[0]+(n-global_normal)@pivot
        return {i:(n,d) for i in active}
    # Derivative-free search because the raster sampler is quantized.
    from scipy.optimize import minimize, minimize_scalar
    objective=lambda x:float(direct_error(direct_plane(x),direct_train).mean())
    grid=np.linspace(-step,step,21)
    d=min(grid,key=lambda t:objective([t,0,0]))
    arms['rgb_dense_distance']=direct_plane([d,0,0])
    baseline=objective([d,0,0])
    solution=minimize(objective,[d,0,0],method='Powell',
                      bounds=[(-step,step),(-angle,angle),(-angle,angle)],
                      options=dict(maxiter=12,xtol=1e-5,ftol=1e-5))
    chosen=solution.x if np.isfinite(solution.fun) and solution.fun<baseline else [d,0,0]
    arms['rgb_dense_direction_distance']=direct_plane(chosen)
    direct_log.update(status='COMPLETED',distance=float(d),direction_distance=list(map(float,chosen)),
        pose='fixed reconstructed VGGT camera',feature='local luminance residual plus Lab chroma, validity frozen at shared-plane warp')


def measure(ps,rows):
    err=np.linalg.norm(residual(ps,rows).reshape(-1,2),axis=1)
    return dict(pairs=len(err),median_px=float(np.median(err)) if len(err) else None)

dtype=np.dtype([(k,'<f4') for k in 'xyz']+[(k,'u1') for k in ('red','green','blue')]+
    [('confidence','<f4')]+[(k,'<u2') for k in ('frame_id','pixel_u','pixel_v')])
def export(name,xyz,modified):
    xyz=xyz[keep]
    colors=rgb.copy()
    colors[modified]=[255,145,20]
    pixels=np.where(keep)
    rec=np.empty(len(xyz),dtype)
    for j,k in enumerate('xyz'):
        rec[k]=xyz[:,j]
    for j,k in enumerate(('red','green','blue')):
        rec[k]=colors[...,j][keep]
    rec['confidence']=confidence[keep]
    rec['frame_id'],rec['pixel_v'],rec['pixel_u']=pixels
    header=(f'ply\nformat binary_little_endian 1.0\nelement vertex {len(rec)}\n'
            'property float x\nproperty float y\nproperty float z\n'
            'property uchar red\nproperty uchar green\nproperty uchar blue\n'
            'property float confidence\nproperty ushort frame_id\nproperty ushort pixel_u\nproperty ushort pixel_v\nend_header\n').encode()
    path=out/f'{name}.ply'
    with path.open('xb') as stream:
        stream.write(header)
        rec.tofile(stream)
    with path.open('rb') as stream:
        assert stream.read(len(header))==header and np.array_equal(np.fromfile(stream,dtype),rec)

# Restrict all arms to pixels with a valid, bounded projection in every arm.
valid=applicable.copy()
for ps in arms.values():
    for i,(a,d) in ps.items():
        z=np.divide(d-center[i]@a,rays[i]@a,out=np.zeros((h,w)),where=abs(rays[i]@a)>1e-9)
        raw_depth=(raw[i]-center[i])@cameras[i,2,:3]
        valid[i] &=(z>0)&(abs(z-raw_depth)<=.30*raw_depth)
# Do not treat a lower appearance cost around a geometry-seeded plane as
# independent evidence that the wall has the right orientation.
normal_angles = {
    f'{i}-{j}':float(np.degrees(np.arccos(np.clip(abs(planes[i][0]@planes[j][0]),0,1))))
    for i in active for j in active if i<j
}
max_normal_angle=max(normal_angles.values(),default=180.)
reasons=[]
if max_normal_angle>15:
    reasons.append('per-view candidate normals disagree by more than 15 degrees')
if len(train)<8 or len(test)<4:
    reasons.append('insufficient spatially held-out feature correspondences')
if reasons:
    export('raw_edge3pct',raw,np.zeros_like(valid))
    export('guarded_wall',raw,np.zeros_like(valid))
    valid[:]=False
    (out/'summary.json').write_text(json.dumps(dict(status='REJECTED_WALL_PROJECTION',
        input=str(input_dir),frames=active.tolist(),reasons=reasons,normal_angles_deg=normal_angles,
        track_fit_pairs=len(train),track_holdout_pairs=len(test),points_changed=0,
        retained_points=int(keep.sum()),frozen_filter='P26 edge_3x3_3pct'),indent=2)+'\n')
    np.savez_compressed(out/'wall_support.npz',mask=mask,applied=valid)
    raise SystemExit(0)
export('raw_edge3pct',raw,np.zeros_like(valid))
report={}
for name,ps in arms.items():
    xyz=raw.copy()
    for i,(a,d) in ps.items():
        q=rays[i][valid[i]]
        xyz[i][valid[i]]=center[i]+q*((d-center[i]@a)/(q@a))[:,None]
    change=(xyz!=raw).any(axis=-1)
    assert (change<=valid).all()
    export(name,xyz,change)
    report[name]=dict(points_changed=int(change.sum()),median_shift_model_units=float(np.median(np.linalg.norm(xyz[change]-raw[change],axis=-1))),
        fit_reprojection=measure(ps,train),heldout_reprojection=measure(ps,test),
        direct_train_cost=float(direct_error(ps,direct_train).mean()) if direct_train else None,
        direct_heldout_cost=float(direct_error(ps,direct_test).mean()) if direct_test else None,
        planes={int(i):dict(normal=a.tolist(),offset=d) for i,(a,d) in ps.items()})
summary=dict(frames=active.tolist(),frozen_filter='P26 edge_3x3_3pct',kept_points=int(keep.sum()),
   wall_mask_pixels=int(mask.sum()),common_projection_pixels=int(valid.sum()),rgb=rgb_fit,direct_rgb=direct_log,
   roi_amendment='exclude small checkered board at frame 15 x>=354,y>=62; RGB inspection',
   policy='only selected wall region; shared valid projection across arms; cap 30% relative camera depth; fixed cameras',
   metric_scope='reprojection on spatially held source blocks, not independent GT; global pose can contribute to residual',
   arms=report)
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
np.savez_compressed(out/'wall_support.npz',mask=mask,applied=valid)
(out/'rgb_pairs.json').write_text(json.dumps([dict(source=i,target=j,uv=x.tolist(),target_uv=y.tolist(),heldout=t)
   for i,j,x,y,t in pairs],indent=2)+'\n')
print(dict(frames=active,points=valid.sum(),rgb=rgb_fit,arms={k:v['heldout_reprojection'] for k,v in report.items()}),flush=True)
