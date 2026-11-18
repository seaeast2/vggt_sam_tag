"""Readable ROI closeups, preserving original full-frame evaluation figures."""
import sys,json,shutil
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'results/planar_residual_depth'
def main():
 out=BASE/sys.argv[1];shutil.copy2(__file__,out/'source/render_roi.py')
 for r in json.loads((out/'case_results.json').read_text()):
  d=ROOT/r['evaluation_dir'];fit=ROOT/json.loads((d/'manifest.json').read_text())['fitdir'];inp=json.loads((fit/('inputs.json' if (fit/'inputs.json').exists() else 'artifacts/inputs.json')).read_text());v=inp['views'][0];armroot=fit/'arms' if (fit/'arms').exists() else fit/'artifacts/arms';arms={a:np.load(armroot/a/'depths.npz') for a in ['B0','B1','B2','B3']};g=np.load(d/'GT_arrays.npz');gt=g['GT_depth'];mask=g['ROI_mask'];valid=g['evaluation_mask'];ys,xs=np.where(mask);y0=max(0,ys.min()-8);y1=min(mask.shape[0],ys.max()+9);x0=max(0,xs.min()-8);x1=min(mask.shape[1],xs.max()+9);sl=np.s_[y0:y1,x0:x1];rgb=np.array(Image.open(ROOT/v['processed']).convert('RGB'))
  lo,hi=np.percentile(np.concatenate([gt[valid]]+[a['corrected_depth'][0][mask] for a in arms.values()]),[1,99]);bound=max(1e-6,float(np.percentile(np.concatenate([abs(a['corrected_depth'][0][valid]-gt[valid]) for a in arms.values()]),95)));cm=plt.get_cmap('viridis').copy();cm.set_bad('#999999');ec=plt.get_cmap('RdBu_r').copy();ec.set_bad('#999999')
  fig,axs=plt.subplots(2,5,figsize=(15,6),layout='constrained');axs[0,0].imshow(rgb[sl]);axs[0,0].set_title('RGB crop');axs[1,0].imshow(np.where(valid,gt,np.nan)[sl],cmap=cm,vmin=lo,vmax=hi);axs[1,0].set_title('GT; gray = not evaluated')
  for col,(a,z) in enumerate(arms.items(),1):
   pred=z['corrected_depth'][0];im=axs[0,col].imshow(np.where(mask,pred,np.nan)[sl],cmap=cm,vmin=lo,vmax=hi);axs[0,col].set_title(a+' depth');ie=axs[1,col].imshow(np.where(valid,pred-gt,np.nan)[sl],cmap=ec,vmin=-bound,vmax=bound);m=next(m for m in r['metrics'] if m['arm']==a);axs[1,col].set_title(f"{a}: MAE {100*m['MAE_m']:.2f} cm\nnormal {m['normal_error_deg']:.2f} deg")
  for ax in axs.ravel():ax.axis('off')
  fig.colorbar(im,ax=axs[0,1:],label='Camera-Z m',shrink=.65);fig.colorbar(ie,ax=axs[1,1:],label='Prediction - GT m',shrink=.65);fig.suptitle(r['dataset']+' '+r['sample']+' | '+('human reviewed' if r['mask_provenance']=='HUMAN_REVIEWED' else 'agent RGB designated; user unreviewed')+f' | crop x={x0}:{x1}, y={y0}:{y1}')
  fig.savefig(d/'ROI_depth_GT_errors.png',dpi=140);plt.close(fig)
 print('Saved 21 ROI closeups; original figures retained.')
if __name__=='__main__':main()
