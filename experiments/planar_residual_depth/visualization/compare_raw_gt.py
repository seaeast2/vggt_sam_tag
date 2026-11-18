"""Display cached raw VGGT depth and GT; no inference/optimization or GT alignment."""
import json,hashlib,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'results/planar_residual_depth'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 prefix='raw-vggt-GT-v';seq=1
 while (BASE/'visualizations'/f'{prefix}{seq:03}').exists():seq+=1
 out=BASE/'visualizations'/f'{prefix}{seq:03}';out.mkdir(parents=True,exist_ok=False)
 source=BASE/'P1b-baseline-direction-20261117-004'
 cases=[('K01','KITTI 2015 / 000000 / left','P1-kitti2015-first20-20261117-001','artifacts/inputs/000000/view0.png'),
        ('M01','Middlebury / Adirondack / left','P1-middlebury-all-20261117-001','artifacts/inputs/Adirondack/view0.png')]
 metadata=[]
 cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#999999')
 ecmap=plt.get_cmap('RdBu_r').copy();ecmap.set_bad('#999999')
 for cid,title,rid,path in cases:
  arr=source/'artifacts'/cid/'arrays.npz';a=np.load(arr)
  raw=a['raw_depth'];s=float(a['scale']);pred=raw*s;gt=a['GT'];rgb=np.array(Image.open(BASE/rid/path))
  valid=np.isfinite(gt)&(gt>0);assert raw.shape==gt.shape==rgb.shape[:2]
  assert np.array_equal(pred,a['baseline']);assert np.isfinite(raw).all() and (raw>0).all()
  # Display-only bounds. No metric values are clipped and no GT-derived scale is fitted.
  lo,hi=np.percentile(gt[valid],[1,99]);bound=float(np.percentile(abs(pred[valid]-gt[valid]),95));bound=max(bound,1e-6)
  fig,axes=plt.subplots(1,3,figsize=(16,5),layout='constrained')
  axes[0].imshow(rgb);axes[0].set_title('Input RGB')
  im=axes[1].imshow(pred,cmap=cmap,vmin=lo,vmax=hi,interpolation='nearest');axes[1].set_title('VGGT raw depth x frozen RGB scale')
  axes[2].imshow(np.where(valid,gt,np.nan),cmap=cmap,vmin=lo,vmax=hi,interpolation='nearest');axes[2].set_title('GT depth (gray = no GT)')
  for ax in axes:ax.axis('off')
  fig.colorbar(im,ax=axes[1:],label='Camera-Z (m); same colors for both maps',shrink=.7,extend='both')
  fig.suptitle(title+f' | display {lo:.2f}–{hi:.2f} m; values outside range saturated',fontsize=12)
  fig.savefig(out/f'{cid}_RGB_depth_GT.png',dpi=150);plt.close(fig)
  # Model-native values: save exact arrays and an explicit model-unit figure.
  np.savez_compressed(out/f'{cid}_visualization_arrays.npz',raw_model_depth=raw,frozen_scale=s,scaled_raw_depth=pred,GT_depth=gt,GT_valid=valid)
  plt.imsave(out/f'{cid}_raw_model_depth.png',raw,cmap=cmap,vmin=lo/s,vmax=hi/s)
  plt.imsave(out/f'{cid}_scaled_raw_depth.png',pred,cmap=cmap,vmin=lo,vmax=hi)
  plt.imsave(out/f'{cid}_GT_depth.png',np.where(valid,gt,np.nan),cmap=cmap,vmin=lo,vmax=hi)
  # Common GT coverage avoids confusing unmeasured areas with reconstruction errors.
  err=np.where(valid,pred-gt,np.nan);rel=np.where(valid,100*(pred-gt)/np.where(valid,gt,1),np.nan)
  fig,axes=plt.subplots(1,3,figsize=(16,5),layout='constrained')
  im=axes[0].imshow(np.where(valid,pred,np.nan),cmap=cmap,vmin=lo,vmax=hi,interpolation='nearest');axes[0].set_title('VGGT on GT-valid pixels')
  fig.colorbar(im,ax=axes[0],label='Camera-Z (m)',shrink=.7,extend='both')
  im=axes[1].imshow(err,cmap=ecmap,vmin=-bound,vmax=bound,interpolation='nearest');axes[1].set_title('Signed error: prediction - GT')
  fig.colorbar(im,ax=axes[1],label='m: red = too far; blue = too near',shrink=.7,extend='both')
  im=axes[2].imshow(rel,cmap=ecmap,vmin=-25,vmax=25,interpolation='nearest');axes[2].set_title('Relative signed error')
  fig.colorbar(im,ax=axes[2],label='%: red = too far; blue = too near',shrink=.7,extend='both')
  for ax in axes:ax.axis('off')
  fig.suptitle(title+' | gray = GT unavailable; no filling or smoothing',fontsize=12)
  fig.savefig(out/f'{cid}_errors.png',dpi=150);plt.close(fig)
  # Zoom the reviewed book for assessing small-object boundaries and local gradients.
  if cid=='M01':
   y=slice(175,255);x=slice(55,195);fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
   zl,zh=np.percentile(np.concatenate([gt[y,x][valid[y,x]],pred[y,x].ravel()]),[1,99])
   for ax,label,z in zip(axes,['RGB crop','VGGT raw x scale','GT crop'],[rgb[y,x],pred[y,x],np.where(valid[y,x],gt[y,x],np.nan)]):
    if z.ndim==3:ax.imshow(z,interpolation='nearest')
    else:im=ax.imshow(z,cmap=cmap,vmin=zl,vmax=zh,interpolation='nearest')
    ax.set_title(label);ax.axis('off')
   fig.colorbar(im,ax=axes[1:],label='Camera-Z (m), shared scale',shrink=.7,extend='both');fig.savefig(out/'M01_book_zoom.png',dpi=150);plt.close(fig)
  metadata.append({'case':cid,'title':title,'arrays_source':str(arr.relative_to(ROOT)),'arrays_sha256':sha(arr),'frozen_scale':s,
   'GT_valid_pixels':int(valid.sum()),'pixels':int(valid.size),'display_range_m':[float(lo),float(hi)],
   'mean_signed_error_m_display_valid':float(np.mean(err[valid])),'positive_error_fraction_display_valid':float(np.mean(err[valid]>0)),
   'error_symmetric_limit_m':bound,'mask_role':'all finite positive GT for visualization only; not an independent heldout metric',
   'model_depth_corrected':False,'GT_used_for_scale':False})
 (out/'provenance.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False)+'\n');shutil.copy2(__file__,out/'source.py')
 print(json.dumps({'output':str(out),'cases':metadata},indent=2))
if __name__=='__main__':main()
