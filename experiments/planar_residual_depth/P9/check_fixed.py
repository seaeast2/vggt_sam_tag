"""Exercise scalar distance/brightness branch and appearance-mismatch diagnostic."""
import json
import numpy as np
from photo import *
from check_photo import image

def main():
 w,h=256,192;K=[[180,0,128],[0,180,96],[0,0,1]];views=[{'K':K,'R':np.eye(3).tolist(),'C':[0,0,0]},{'K':K,'R':np.eye(3).tolist(),'C':[.4,0,0]}];a=image(w,h);b=(image(w,h,18)-.02)/1.1;uv=np.array([[x,y] for y in range(32,161,4) for x in range(40,221,4)],float);truth=np.array([0.,0.,-.25]);p0=truth*.97;poly=[[35,25],[225,25],[225,165],[35,165]];item={'views':views,'polygons':[poly,[[10,20],[230,20],[230,170],[10,170]]],'ps':{'B1':p0,'B2':p0,'B3':p0}};protocol={'optimizer':{'Gaussian_sigma_processed_px':[1.,.5,0.],'maxiter':200,'ftol':1e-10}};best,log=solve_photo(item,uv,[a,b],True,protocol);assert best is not None,log;p=np.array(best['p']);err=float(np.max(abs(project(p,uv,views)[0]-project(truth,uv,views)[0])));assert err<.02
 clean=diagnostic(truth,1.1,.02,uv,a,b,views);occluded=b.copy();occluded[50:100,80:120]=np.clip(occluded[50:100,80:120]+.3,0,1);occ=diagnostic(truth,1.1,.02,uv,a,occluded,views);assert occ['high_residual_fraction_norm_gt_0_03']>clean['high_residual_fraction_norm_gt_0_03']+.01
 print(json.dumps({'status':'PASSED','fixed_direction_known_plane_projection_max_error_px':err,'clean_high_residual_fraction':clean['high_residual_fraction_norm_gt_0_03'],'synthetic_modified_patch_high_residual_fraction':occ['high_residual_fraction_norm_gt_0_03']},indent=2))
if __name__=='__main__':main()
