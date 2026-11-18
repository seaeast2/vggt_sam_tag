"""Independent analytic cone-gradient, exact-data recovery, and cone-boundary checks."""
import json
import numpy as np
from scipy.spatial.transform import Rotation
from common import *
from run_fit import constrained_solve

def main():
 rng=np.random.default_rng(0);n0=np.array([0,-.5,-.08]);n0/=np.linalg.norm(n0);c=np.cos(np.deg2rad(1));maxerr=0
 for _ in range(20):
  p=rng.normal(size=3);f=lambda q:q@n0-c*np.linalg.norm(q);eps=1e-6;numeric=np.array([(f(p+np.eye(3)[k]*eps)-f(p-np.eye(3)[k]*eps))/(2*eps) for k in range(3)]);analytic=n0-c*p/np.linalg.norm(p);maxerr=max(maxerr,float(np.max(abs(numeric-analytic))))
 assert maxerr<1e-8
 K=[[400,0,256],[0,400,168],[0,0,1]];views=[{'K':K,'R':np.eye(3).tolist(),'C':[0,0,0]},{'K':K,'R':np.eye(3).tolist(),'C':[.3,0,0]}];p0=np.array([0,-.5,-.08]);uv=np.array([[x,y] for x in np.linspace(100,420,8) for y in np.linspace(220,310,5)]);poly=[[80,210],[440,210],[440,320],[80,320]];records=[]
 for deg in [.5,2.]:
  ptrue=Rotation.from_rotvec([0,0,np.deg2rad(deg)]).as_matrix()@p0;X=depths(ptrue,uv,views[0])[:,None]*rays(uv,views[0]);h=(X-views[1]['C'])@np.array(K).T;vv=h[:,:2]/h[:,2:];obs=[(uv,vv,0,1),(vv,uv,1,0)];polygons=[poly,[[float(vv[:,0].min()-5),210],[440,210],[440,320],[float(vv[:,0].min()-5),320]]]
  p2,_=solve(p0,obs,views,polygons,fixed=True);p,logs=constrained_solve(p0,p2,ptrue,obs,views,polygons,1.);assert p is not None;aa=angle(p,p0);assert aa<=1+1e-6
  if deg<1:assert angle(p,ptrue)<1e-4
  else:assert aa>.99
  records.append({'true_direction_change_deg':deg,'retained_direction_change_deg':aa,'true_normal_error_deg':angle(p,ptrue)})
 print(json.dumps({'cone_gradient_max_error':maxerr,'synthetic_cone_tests':records,'status':'PASSED'},indent=2))
if __name__=='__main__':main()
