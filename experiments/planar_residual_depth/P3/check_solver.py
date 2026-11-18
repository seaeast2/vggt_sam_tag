"""Independent synthetic geometry checks before real P2 observations are used."""
import numpy as np,json
from solver import rays,depths,residual_jacobian,objective,solve
v0={'K':[[240,0,159.5],[0,240,119.5],[0,0,1]],'R':np.eye(3).tolist(),'C':[0,0,0]};v1=dict(v0,C=[.3,0,0]);vs=[v0,v1]
xx,yy=np.meshgrid(np.linspace(80,240,9),np.linspace(60,180,7));uv=np.c_[xx.ravel(),yy.ravel()]
truth=np.array([.08,.04,-1/3]);p0=truth+np.array([.08,-.03,.02])
z=depths(truth,uv,v0);X=z[:,None]*rays(uv,v0);Y=X-np.array(v1['C']);h=Y@np.array(v1['K']).T;vv=h[:,:2]/h[:,2:]
obs=[(uv,vv,0,1),(vv,uv,1,0)];r,J=residual_jacobian(p0,obs,vs);eps=1e-6
numeric=np.column_stack([(residual_jacobian(p0+np.eye(3)[k]*eps,obs,vs)[0]-residual_jacobian(p0-np.eye(3)[k]*eps,obs,vs)[0])/(2*eps) for k in range(3)])
assert np.max(abs(J-numeric))/max(1,np.max(abs(J)))<1e-6
_,g=objective(p0,obs,vs);ng=np.array([(objective(p0+np.eye(3)[k]*eps,obs,vs)[0]-objective(p0-np.eye(3)[k]*eps,obs,vs)[0])/(2*eps) for k in range(3)])
assert np.max(abs(g-ng))/max(1,np.max(abs(g)))<1e-6
polys=[[[70,50],[250,50],[250,190],[70,190]]]*2
p,log=solve(p0,obs,vs,polys);assert p is not None;assert np.linalg.norm(p-truth)<1e-6
p2,_=solve(p0,obs,vs,polys,fixed=True);assert p2 is not None;assert np.linalg.norm(np.cross(p2,p0))<1e-10
print(json.dumps({'analytic_Jacobian_passed':True,'objective_gradient_passed':True,'known_plane_recovered':True,'fixed_normal_preserved':True,'p_error':float(np.linalg.norm(p-truth))}))
