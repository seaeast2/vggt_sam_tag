"""Synthetic check of disparity inversion with full rectified-camera baseline."""
import numpy as np,json
rng=np.random.default_rng(0);f=721.5377;cx=609.5593;cy=172.854
checks=[]
for C in [np.array([.5327119,0,0]),np.array([.5327119,-.0027529,.000015979]),np.array([.53,.01,.005])]:
 z=rng.uniform(3,80,1000);rx=rng.uniform(-.5,.5,len(z));ry=rng.uniform(-.3,.3,len(z));X=np.c_[z*rx,z*ry,z]
 left=np.c_[f*rx+cx,f*ry+cy];right=np.c_[f*(X[:,0]-C[0])/(z-C[2])+cx,f*(X[:,1]-C[1])/(z-C[2])+cy]
 d=left[:,0]-right[:,0];recovered=f*(C[0]-rx*C[2])/d+C[2]
 assert np.max(abs(recovered-z))<1e-8
 checks.append({'baseline':C.tolist(),'max_depth_error_m':float(np.max(abs(recovered-z)))})
print(json.dumps({'full_baseline_inversion_passed':True,'cases':checks},indent=2))
