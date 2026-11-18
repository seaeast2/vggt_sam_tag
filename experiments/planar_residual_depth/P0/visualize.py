"""Visualize stored P0 arrays; does not generate or imply RGB test inputs."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[3]
run='P0-synthetic-planes-20261117-002'
src=ROOT/'results/planar_residual_depth'/run
out=ROOT/'results/planar_residual_depth/visualizations'/f'{run}-v001'
out.mkdir(parents=True,exist_ok=True)
config=json.loads((src/'config_resolved.json').read_text()); K=np.array(config['K'])
fig,axes=plt.subplots(4,5,figsize=(15,10))
fig.subplots_adjust(left=.02,right=.89,bottom=.02,top=.92,hspace=.35,wspace=.12)
for ci,case in enumerate(['tilt','ripple']):
 folder=src/'artifacts'/f'plane1_{case}'
 t=json.loads((folder/'inputs/truth.json').read_text()); p=np.array(t['p'])
 raw=np.load(folder/'arms/B0/view0.npz'); mask=raw['mask']; h,w=mask.shape
 y,x=np.mgrid[:h,:w]; rays=np.stack([x,y,np.ones_like(x)],-1)@np.linalg.inv(K).T
 gt=-1/(rays@p)
 maps=[gt]+[np.load(folder/f'arms/{arm}/view0.npz')['corrected_depth'] for arm in ['B0','B1','B2','B3']]
 for col,(name,z) in enumerate(zip(['GT','B0 raw','B1 plane fit','B2 distance','B3 direction + distance'],maps)):
  ax=axes[ci*2,col]; im=ax.imshow(np.where(mask,z,np.nan),vmin=2.3,vmax=3.9,cmap='viridis');ax.set_title(f'{case}: {name}');ax.axis('off')
  ax=axes[ci*2+1,col]; er=abs(z-gt)*100
  im2=ax.imshow(np.where(mask,er,np.nan),vmin=0,vmax=30,cmap='magma');ax.axis('off');ax.set_title(f'Absolute error: {er[mask].mean():.3f} cm')
fig.colorbar(im,cax=fig.add_axes([.92,.56,.015,.31]),label='Camera-Z depth (m)',shrink=.7)
fig.colorbar(im2,cax=fig.add_axes([.92,.12,.015,.31]),label='Absolute depth error (cm)',shrink=.7)
fig.suptitle('Stored synthetic depth arrays (not RGB images); white = outside finite plane',fontsize=14)
fig.savefig(out/'depth_maps.png',dpi=140);plt.close(fig)
fig=plt.figure(figsize=(15,8),layout='constrained')
for row,case in enumerate(['tilt','ripple']):
 folder=src/'artifacts'/f'plane1_{case}';t=json.loads((folder/'inputs/truth.json').read_text());p=np.array(t['p'])
 for col,arm in enumerate(['B0','B1','B3']):
  arr=np.load(folder/f'arms/{arm}/view0.npz'); mask=arr['mask'];z=arr['corrected_depth'];h,w=z.shape
  y,x=np.mgrid[:h,:w];rays=np.stack([x,y,np.ones_like(x)],-1)@np.linalg.inv(K).T
  gt=-1/(rays@p); X=rays*z[...,None];G=rays*gt[...,None]
  ax=fig.add_subplot(2,3,row*3+col+1,projection='3d')
  a=X[::5,::5].copy();m=mask[::5,::5];a[~m]=np.nan
  err=abs(z-gt)[::5,::5]*100
  ax.plot_surface(a[:,:,0],a[:,:,2],a[:,:,1],facecolors=plt.cm.magma(np.clip(err/30,0,1)),rstride=1,cstride=1,shade=False)
  g=G[::15,::15].copy();g[~mask[::15,::15]]=np.nan
  ax.plot_wireframe(g[:,:,0],g[:,:,2],g[:,:,1],color='cyan',linewidth=.5,alpha=.7)
  ax.set(xlim=(-1,1),ylim=(2.3,3.9),zlim=(-1,1),xlabel='X (m)',ylabel='Z (m)',zlabel='Y (m)',title=f'{case}: {arm}')
  ax.view_init(elev=20,azim=-65)
fig.suptitle('Same viewpoint and limits; cyan wireframe = GT; surface color = depth error (0–30 cm)')
fig.savefig(out/'surfaces.png',dpi=140);plt.close(fig)
(out/'provenance.json').write_text(json.dumps({'source_run':run,'source_case':'plane1_tilt, plane1_ripple','view':0,'kind':'post-run depth/surface visualization, no RGB inputs'},indent=2))
print(out)
