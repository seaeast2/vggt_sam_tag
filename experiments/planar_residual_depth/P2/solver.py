"""Observation-only plane solver, arbitrary fixed calibrated cameras. No GT I/O."""
import numpy as np
from scipy.optimize import minimize,LinearConstraint

def rays(uv,view):
 return np.c_[uv,np.ones(len(uv))]@np.linalg.inv(np.array(view['K'])).T@np.array(view['R'])
def depths(p,uv,view):return -(1+np.array(view['C'])@p)/(rays(uv,view)@p)
def residual_jacobian(p,observations,views):
 residual=[];jac=[]
 for uv,vv,i,j in observations:
  vi,vj=views[i],views[j];Ci,Cj=np.array(vi['C']),np.array(vj['C']);q=rays(uv,vi);den=q@p;A=1+Ci@p;z=-A/den
  dz=-(Ci[None,:]*den[:,None]-A*q)/den[:,None]**2
  X=Ci+z[:,None]*q;M=np.array(vj['K'])@np.array(vj['R']);h=(X-Cj)@M.T
  pred=h[:,:2]/h[:,2:];dp=(M[None,:2,:]*h[:,2,None,None]-h[:,:2,None]*M[None,2:3,:])/h[:,2,None,None]**2
  residual.append((pred-vv).reshape(-1));jac.append(np.einsum('nij,nj,nk->nik',dp,q,dz).reshape(-1,3))
 return np.concatenate(residual),np.concatenate(jac)
def objective(p,obs,views):
 r,J=residual_jacobian(p,obs,views);rr=r.reshape(-1,2);norm=np.linalg.norm(rr,axis=1)
 cost=np.mean(np.where(norm<=1,.5*norm**2,norm-.5));gradient=J.T@(rr/np.maximum(1,norm)[:,None]).ravel()/len(norm)
 return float(cost),gradient

def linear_seed(obs,views):
 uv,vv,i,j=obs[0];vi,vj=views[i],views[j]
 if np.linalg.norm(vi['C'])>1e-10 or np.max(abs(np.array(vi['R'])-np.eye(3)))>1e-10:return None
 ra=rays(uv,vi);rb=np.c_[vv,np.ones(len(vv))]@np.linalg.inv(np.array(vj['K'])).T
 R=np.array(vj['R']);t=R@(np.array(vi['C'])-np.array(vj['C']));cross=np.cross(rb,t);b=np.cross(rb,ra@R.T)
 A=(cross[:,:,None]*ra[:,None,:]).reshape(-1,3)
 p,_,rank,_=np.linalg.lstsq(A,b.reshape(-1),rcond=None)
 return p if rank==3 else None

def solve(p0,obs,views,polygons,fixed=False):
 rows=[];upper=[]
 for view,poly in zip(views,polygons):
  rows.extend(rays(np.array(poly),view));upper.extend([-1e-8]*len(poly))
  rows.append(-np.array(view['C']));upper.append(1-1e-8)
 A=np.array(rows);upper=np.array(upper)
 starts=[('raw',p0)]
 seed=linear_seed(obs,views)
 if not fixed and seed is not None:starts.append(('FIT_linear',seed))
 results=[]
 for label,pstart in starts:
  if np.any(A@pstart>upper):
   results.append({'seed':label,'success':False,'reason':'initial_positive_depth_constraint_failed'});continue
  unpack=(lambda x:p0*x[0]) if fixed else (lambda x:x)
  fun=lambda x:objective(unpack(x),obs,views)[0]
  grad=lambda x:np.array([objective(unpack(x),obs,views)[1]@p0]) if fixed else objective(unpack(x),obs,views)[1]
  mat=(A@p0)[:,None] if fixed else A
  initial=np.array([1.]) if fixed else pstart
  result=minimize(fun,initial,jac=grad,method='SLSQP',constraints=[LinearConstraint(mat,-np.inf,upper)],
   options={'maxiter':500,'ftol':1e-12})
  p=unpack(result.x);cost,_=objective(p,obs,views)
  results.append({'seed':label,'success':bool(result.success) and bool(np.all(A@p<=upper+1e-9)),
    'message':str(result.message),'iterations':int(result.nit),'p':p.tolist(),'FIT_cost':cost})
 good=[r for r in results if r['success']]
 if not good:return None,results
 best=min(good,key=lambda r:r['FIT_cost']);return np.array(best['p']),results
