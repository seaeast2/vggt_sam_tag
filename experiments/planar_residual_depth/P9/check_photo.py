"""Bilinear/photometric analytic-gradient and known-plane synthetic recovery tests."""
import json
import numpy as np
from photo import *

def image(w,h,shift=0):
 y,x=np.mgrid[:h,:w];x=x+shift
 return np.stack([.45+.13*np.sin(x*.15)+.1*np.cos(y*.11),.45+.12*np.sin(x*.09+y*.07),.45+.12*np.cos(x*.13-y*.14)],axis=-1)

def main():
 w,h=256,192;K=[[180,0,128],[0,180,96],[0,0,1]];views=[{'K':K,'R':np.eye(3).tolist(),'C':[0,0,0]},{'K':K,'R':np.eye(3).tolist(),'C':[.4,0,0]}];a=image(w,h);b=(image(w,h,18)-.02)/1.1;uv=np.array([[x,y] for y in range(32,161,4) for x in range(40,221,4)],float);ptrue=np.array([0.,0.,-.25]);p0=ptrue+np.array([.002,-.002,.0047]);x0=np.r_[p0,1.03,.01]
 def fun(x):return loss_and_grad(x[:3],x[3],x[4],uv,a,b,views)
 f=fun(x0);analytic=np.r_[f[1],f[2:]];eps=1e-7;num=np.array([(fun(x0+np.eye(5)[k]*eps)[0]-fun(x0-np.eye(5)[k]*eps)[0])/(2*eps) for k in range(5)]);err=float(np.max(abs(num-analytic))/max(1,np.max(abs(num))));assert err<1e-5,err
 poly=[[35,25],[225,25],[225,165],[35,165]];polys=[poly,[[10,20],[230,20],[230,170],[10,170]]];protocol={'optimizer':{'Gaussian_sigma_processed_px':[1.,.5,0.],'maxiter':200,'ftol':1e-10}};item={'views':views,'polygons':polys,'ps':{'B1':p0,'B2':p0,'B3':p0+np.array([.001,0,0])}};best,logs=solve_photo(item,uv,[a,b],False,protocol);assert best is not None,logs;p=np.array(best['p']);got,_=project(p,uv,views);truth,_=project(ptrue,uv,views);warp_error=float(np.max(abs(got-truth)));assert warp_error<.02,warp_error
 constant=np.ones_like(a)*.4;diag=diagnostic(p0,1,0,uv,constant,constant,views);assert diag['photo_plane_J_singular_ratio_after_brightness']==0 and diag['source_gradient_RMS']==0
 print(json.dumps({'status':'PASSED','photometric_gradient_relative_max_error':err,'known_plane_projection_max_error_px':warp_error,'recovered_gain':best['gain'],'recovered_bias':best['bias'],'constant_texture_degeneracy_detected':True},indent=2))
if __name__=='__main__':main()
