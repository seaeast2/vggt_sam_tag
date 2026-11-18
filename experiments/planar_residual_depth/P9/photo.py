"""Differentiable planar RGB warp and photometric solver; no GT I/O."""
import sys
from pathlib import Path
import numpy as np,cv2
from scipy.optimize import minimize,LinearConstraint
sys.path.insert(0,str(Path(__file__).parents[1]/'P3'))
from solver import rays,depths,residual_jacobian

def project(p,uv,views):
 r,J=residual_jacobian(p,[(uv,np.zeros_like(uv),0,1)],views)
 return r.reshape(-1,2),J.reshape(-1,2,3)

def bilinear(im,uv):
 # Floating-point bilinear sampling and its exact piecewise derivatives.
 h,w=im.shape[:2];x=uv[:,0];y=uv[:,1];inside=(x>=0)&(x<w-1)&(y>=0)&(y<h-1)&np.isfinite(uv).all(1);xs=np.clip(x,0,w-1.000001);ys=np.clip(y,0,h-1.000001);ix=np.floor(xs).astype(int);iy=np.floor(ys).astype(int);dx=xs-ix;dy=ys-iy
 a=im[iy,ix];b=im[iy,ix+1];c=im[iy+1,ix];d=im[iy+1,ix+1];val=a*(1-dx[:,None])*(1-dy[:,None])+b*dx[:,None]*(1-dy[:,None])+c*(1-dx[:,None])*dy[:,None]+d*dx[:,None]*dy[:,None];gx=(b-a)*(1-dy[:,None])+(d-c)*dy[:,None];gy=(c-a)*(1-dx[:,None])+(d-b)*dx[:,None]
 return val,np.stack([gx,gy],axis=-1),inside

def residual(p,gain,bias,uv,source,target,views):
 coords,J=project(p,uv,views);val,grad,inside=bilinear(target,coords);src=source[uv[:,1].astype(int),uv[:,0].astype(int)];res=gain*val+bias-src;Jp=gain*np.einsum('nci,nij->ncj',grad,J);return res,Jp,val,inside

def loss_and_grad(p,gain,bias,uv,source,target,views,delta=.03):
 r,Jp,val,inside=residual(p,gain,bias,uv,source,target,views);norm=np.linalg.norm(r,axis=1);cost=np.mean(np.where(norm<=delta,.5*norm**2,delta*(norm-.5*delta)))/delta**2;dr=r*np.minimum(1,delta/np.maximum(norm,1e-15))[:,None]/(len(r)*delta**2);gradp=np.einsum('nc,ncj->j',dr,Jp);gg=float(np.sum(dr*val));gb=float(np.sum(dr));return float(cost),gradp,gg,gb

def fit_brightness(p,uv,source,target,views):
 fun=lambda x:loss_and_grad(p,x[0],x[1],uv,source,target,views)
 opt=minimize(lambda x:fun(x)[0],[1.,0.],jac=lambda x:np.array(fun(x)[2:]),method='L-BFGS-B',bounds=[(.7,1.3),(-.15,.15)],options={'ftol':1e-12,'maxiter':100})
 return float(opt.x[0]),float(opt.x[1]),bool(opt.success)

def diagnostic(p,gain,bias,uv,source,target,views):
 r,Jp,val,inside=residual(p,gain,bias,uv,source,target,views);norm=np.linalg.norm(r,axis=1);sourceval=source[uv[:,1].astype(int),uv[:,0].astype(int)];g=np.stack([cv2.Sobel(source,cv2.CV_64F,1,0,ksize=3)/8,cv2.Sobel(source,cv2.CV_64F,0,1,ksize=3)/8],axis=-1);G=g[uv[:,1].astype(int),uv[:,0].astype(int)];A=np.c_[val.ravel(),np.ones(val.size)];J=Jp.reshape(-1,3);J=J-A@np.linalg.lstsq(A,J,rcond=None)[0];sv=np.linalg.svd(J,compute_uv=False);ratio=float(sv[-1]/sv[0]) if sv[0]>1e-10 else 0.
 return {'RGB_mean_abs_error':float(np.mean(abs(r))),'RGB_radial_RMSE':float(np.sqrt(np.mean(norm**2))),'high_residual_fraction_norm_gt_0_03':float(np.mean(norm>.03)),'sample_inside_image_fraction':float(np.mean(inside)),'source_gradient_RMS':float(np.sqrt(np.mean(G**2))),'source_RGB_std':float(np.std(sourceval)),'photo_plane_J_singular_values':sv.tolist(),'photo_plane_J_singular_ratio_after_brightness':ratio,'gain':gain,'bias':bias,'brightness_bound_hit':bool(min(abs(gain-.7),abs(gain-1.3))<1e-4 or min(abs(bias+.15),abs(bias-.15))<1e-4)}

def constraints(views,polygons,pref,fixed_base=None):
 aa=[];bb=[]
 for v,poly in zip(views,polygons):
  aa.extend(rays(np.array(poly),v));bb.extend([-1e-8]*len(poly));aa.append(-np.array(v['C']));bb.append(1-1e-8)
 A=np.array(aa);upper=np.array(bb);corners=np.array(polygons[0],float);ref,_=project(pref,corners,views)
 def unpack(x):return fixed_base*x[0] if fixed_base is not None else x[:3]
 nd=1 if fixed_base is not None else 3
 def warp(x):return project(unpack(x),corners,views)[0].ravel()
 def jac(x):
  _,J=project(unpack(x),corners,views);J=J.reshape(-1,3);geo=(J@fixed_base)[:,None] if fixed_base is not None else J;return np.c_[geo,np.zeros((len(geo),2))]
 def admissible(p):
  q,_=project(p,corners,views);return bool(np.all(A@p<=upper+1e-9) and np.max(abs(q-ref))<=3+1e-8)
 mat=(A@fixed_base)[:,None] if fixed_base is not None else A;mat=np.c_[mat,np.zeros((len(A),2))]
 cons=[LinearConstraint(mat,-np.inf,upper),{'type':'ineq','fun':lambda x:np.r_[3-(warp(x)-ref.ravel()),3+(warp(x)-ref.ravel())],'jac':lambda x:np.r_[-jac(x),jac(x)]}]
 return unpack,nd,cons,admissible

def solve_photo(item,uv,images,fixed,protocol):
 views=item['views'];polygons=item['polygons'];ps=item['ps'];base=ps['B2'] if fixed else None;unpack,nd,cons,admissible=constraints(views,polygons,ps['B2'],base);starts=[('B2',ps['B2'])]
 if not fixed:starts.append(('B3',ps['B3']))
 starts.append(('raw_B1',ps['B1']));logs=[];valid=[]
 for label,p in starts:
  if not admissible(p):logs.append({'seed':label,'success':False,'reason':'INITIAL_TRUST_OR_POSITIVE_DEPTH_FAILED'});continue
  if fixed:geo=np.array([float(p@base/(base@base))])
  else:geo=p.copy()
  x=np.r_[geo,1.,0.];history=[];success=True
  for sigma in protocol['optimizer']['Gaussian_sigma_processed_px']:
   ims=[cv2.GaussianBlur(im,(0,0),sigma) if sigma>0 else im for im in images]
   def fg(xx):
    cost,gp,gg,gb=loss_and_grad(unpack(xx),xx[nd],xx[nd+1],uv,ims[0],ims[1],views);ggeo=np.array([gp@base]) if fixed else gp;return cost,np.r_[ggeo,gg,gb]
   opt=minimize(lambda xx:fg(xx)[0],x,jac=lambda xx:fg(xx)[1],method='SLSQP',bounds=[(None,None)]*nd+[(.7,1.3),(-.15,.15)],constraints=cons,options={'maxiter':protocol['optimizer']['maxiter'],'ftol':protocol['optimizer']['ftol']})
   x=opt.x;history.append({'sigma':sigma,'success':bool(opt.success),'message':str(opt.message),'iterations':int(opt.nit),'cost':float(opt.fun)})
   if not opt.success or not admissible(unpack(x)):success=False;break
  pp=unpack(x);cost=loss_and_grad(pp,x[nd],x[nd+1],uv,images[0],images[1],views)[0];rec={'seed':label,'success':success,'history':history,'p':pp.tolist(),'gain':float(x[nd]),'bias':float(x[nd+1]),'FIT_photo_cost':cost};logs.append(rec)
  if success:valid.append(rec)
 if not valid:return None,logs
 return min(valid,key=lambda r:r['FIT_photo_cost']),logs
