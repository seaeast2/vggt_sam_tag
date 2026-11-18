"""Export finite source ROI pointmaps; RGB/pixel provenance and plane identity retained."""
import sys,json,shutil
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'results/planar_residual_depth';sys.path.insert(0,str(Path(__file__).parents[1]/'P3'));from solver import rays

def main():
 out=BASE/sys.argv[1];rows=json.loads((out/'case_results.json').read_text());shutil.copy2(__file__,out/'source/export_pointmaps.py')
 for case in rows:
  d=ROOT/case['evaluation_dir'];mf=json.loads((d/'manifest.json').read_text());fit=ROOT/mf['fitdir'];inp=json.loads((fit/'inputs.json').read_text()) if (fit/'inputs.json').exists() else json.loads((fit/'artifacts/inputs.json').read_text());v=inp['views'][0];ar=fit/'arms' if (fit/'arms').exists() else fit/'artifacts/arms';z0=np.load(ar/'B0/depths.npz');mask=z0['masks'][0];h,w=mask.shape;yy,xx=np.mgrid[:h,:w];uv=np.c_[xx[mask],yy[mask]];q=rays(uv,v);color=np.array(Image.open(ROOT/v['processed']).convert('RGB'))[mask];dest=d/'pointmaps';dest.mkdir(exist_ok=False);points={}
  for a in ['B0','B1','B2','B3']:
   z=np.load(ar/a/'depths.npz')['corrected_depth'][0][mask];X=np.array(v['C'])+z[:,None]*q;points[a]=X
   dtype=np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1'),('pixel_u','<i4'),('pixel_v','<i4')]);data=np.empty(len(X),dtype=dtype)
   for k,key in enumerate(['x','y','z']):data[key]=X[:,k]
   for k,key in enumerate(['red','green','blue']):data[key]=color[:,k]
   data['pixel_u']=uv[:,0];data['pixel_v']=uv[:,1]
   with (dest/(a+'.ply')).open('wb') as f:
    f.write(('ply\nformat binary_little_endian 1.0\ncomment source_view_0_ROI_only_camera_reference_coordinates\ncomment plane_id '+case['dataset']+'_'+case['sample']+'\nelement vertex '+str(len(X))+'\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nproperty int pixel_u\nproperty int pixel_v\nend_header\n').encode());data.tofile(f)
  (dest/'provenance.json').write_text(json.dumps({'scope':'source view0 ROI ONLY, derived from corrected depths and provided K/pose; not network pose pointmap or whole scene','plane_id':case['dataset']+'_'+case['sample'],'mask_provenance':case['mask_provenance'],'frame_id':v['source'],'pixel_coordinates':'processed original pixel identities','coordinate_frame':'provided cameras expressed in reference source camera coordinates','views':inp['views'],'polygons':inp['polygons'],'parent_fit':mf['fitdir'],'vertices_per_arm':len(uv),'RGB_and_pixel_ids_identical_across_arms':True},indent=2)+'\n')
  if case['sample'] not in ['000005','Motorcycle']:continue
  # Predetermined first available evaluation case per stereo dataset; common axes/view.
  fig=plt.figure(figsize=(14,4),layout='constrained');allx=np.concatenate(list(points.values()));lo=np.min(allx,axis=0);hi=np.max(allx,axis=0);center=(lo+hi)/2;radius=max(hi-lo)/2
  for i,a in enumerate(['B0','B1','B2','B3']):
   ax=fig.add_subplot(1,4,i+1,projection='3d');X=points[a];step=max(1,len(X)//4000);ax.scatter(X[::step,0],X[::step,1],X[::step,2],c=color[::step]/255,s=1)
   ax.set_xlim(center[0]-radius,center[0]+radius);ax.set_ylim(center[1]-radius,center[1]+radius);ax.set_zlim(center[2]-radius,center[2]+radius);ax.set_box_aspect((1,1,1));ax.view_init(elev=-25,azim=-65);ax.set_title(a);ax.set_xlabel('X m');ax.set_ylabel('Y m');ax.set_zlabel('Z m')
  fig.suptitle(case['dataset']+' '+case['sample']+' finite ROI pointmaps; common coordinates and view');fig.savefig(d/'pointmaps_common_view.png',dpi=130);plt.close(fig)
 print('Exported 21 source ROI pointmap sets, four arms each.')
if __name__=='__main__':main()
