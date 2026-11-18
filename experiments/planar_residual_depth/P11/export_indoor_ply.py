"""Three identical full-scene point clouds from cached TUM RGB inference; colors only differ."""
import argparse,json,time,shutil,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
parser = argparse.ArgumentParser()
parser.add_argument('--raw-run', default='P10raw-three-sequences-20261117-001')
parser.add_argument('--output-prefix', default='indoor-P10ply')
args = parser.parse_args()
R=Path(__file__).resolve().parents[3];B=R/'results/planar_residual_depth';rawparent=B/args.raw_run;inputparent=B/json.loads((rawparent/'manifest.json').read_text())['parent'];rawpath=rawparent/'indoor/predictions.npz'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
prefix=args.output_prefix+'-'+time.strftime('%Y%m%d')+'-';i=1
while (B/(prefix+f'{i:03}')).exists():i+=1
out=B/(prefix+f'{i:03}');out.mkdir();shutil.copy2(__file__,out/'export_indoor_ply.py');before=sha(rawpath);parentmanifest=json.loads((rawparent/'manifest.json').read_text());entry=next(x for x in parentmanifest['sequences'] if x['sample']=='indoor');assert before==entry['output_sha256'];raw=np.load(rawpath);depth=raw['depth'][0,...,0];conf=raw['depth_conf'][0].reshape(depth.shape);ext=raw['extrinsics'][0].astype(float);K=raw['intrinsics'][0].astype(float);case=json.loads((inputparent/'indoor/case.json').read_text());n,h,w=depth.shape;assert n==16;views=case['images'][:n];DT=np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1'),('confidence','<f4'),('frame_id','<u2'),('pixel_u','<u2'),('pixel_v','<u2')]);palette=np.array([[255,80,80],[255,150,40],[255,220,40],[185,240,40],[80,210,70],[30,210,150],[30,210,220],[40,150,255],[70,85,255],[150,70,255],[225,70,255],[255,70,170],[190,130,90],[80,190,180],[170,170,255],[220,220,220]],dtype='uint8');yy,xx=np.mgrid[:h,:w];uv=np.c_[xx.ravel(),yy.ravel(),np.ones(h*w)];R0=ext[0,:,:3];t0=ext[0,:,3];blocks=[];info=[]
for j,v in enumerate(views):
 path=R/v['processed'];assert sha(path)==entry['input_sha256'][j];im=np.array(Image.open(path).convert('RGB'));assert im.shape==(h,w,3);rot=ext[j,:,:3];t=ext[j,:,3];cam=depth[j].ravel()[:,None]*(uv@np.linalg.inv(K[j]).T);world=(cam-t)@np.linalg.inv(rot).T;XYZ=world@R0.T+t0;back=(world@rot.T+t)@K[j].T;assert np.max(abs(back[:,:2]/back[:,2:]-uv[:,:2]))<1e-7;good=np.isfinite(XYZ).all(1)&np.isfinite(depth[j].ravel())&(depth[j].ravel()>0);a=np.empty(good.sum(),DT)
 for k,key in enumerate(['x','y','z']):a[key]=XYZ[good,k]
 for k,key in enumerate(['red','green','blue']):a[key]=im.reshape(-1,3)[good,k]
 a['confidence']=conf[j].ravel()[good];a['frame_id']=j;a['pixel_u']=xx.ravel()[good];a['pixel_v']=yy.ravel()[good];blocks.append(a);info.append({'frame_id':j,'timestamp':v['timestamp'],'t_sec':v['t_sec'],'vertices':len(a),'frame_RGB':palette[j].tolist(),'input':v['processed']})
a=np.concatenate(blocks);assert np.isfinite(a['confidence']).all();lo,hi=np.percentile(a['confidence'],[1,99]);anchors=np.array([[230,30,30],[255,220,0],[30,200,60],[0,200,230],[30,60,230]],float);head=('ply\nformat binary_little_endian 1.0\ncomment TUM VGGT-Omega raw predicted depth/K/poses, no correction or GT scale\ncomment frame0 predicted camera axes x right y down z forward; model gauge\nelement vertex '+str(len(a))+'\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nproperty float confidence\nproperty ushort frame_id\nproperty ushort pixel_u\nproperty ushort pixel_v\nend_header\n').encode();files=[]
for mode in ['RGB','frame','confidence']:
 b=a.copy()
 if mode=='frame':colors=palette[b['frame_id']]
 elif mode=='confidence':
  u=np.clip((b['confidence']-lo)/(hi-lo),0,1);colors=np.rint(np.stack([np.interp(u,np.linspace(0,1,len(anchors)),anchors[:,k]) for k in range(3)],1)).astype('uint8')
 if mode!='RGB':
  for k,key in enumerate(['red','green','blue']):b[key]=colors[:,k]
 path=out/f'vggt_omega_raw_{mode}.ply'
 with path.open('wb') as f:f.write(head);f.write(b.tobytes())
 with path.open('rb') as f:
  while f.readline().strip()!=b'end_header':pass
  offset=f.tell();check=np.fromfile(f,DT)
 assert path.stat().st_size==offset+len(a)*DT.itemsize and np.array_equal(check,b)
 for key in DT.names:
  if key not in ['red','green','blue']:assert np.array_equal(check[key],a[key])
 files.append({'mode':mode,'path':str(path.relative_to(R)),'vertices':len(check),'bytes':path.stat().st_size,'sha256':sha(path)});print(mode,len(check),path,flush=True)
assert sha(rawpath)==before;dump(out/'frames.json',info);dump(out/'manifest.json',{'status':'ALL_THREE_PLY_BINARY_READBACK_VERIFIED','dataset':'TUM RGB-D fr3_structure_texture_far','raw_parent':rawparent.name,'input_parent':inputparent.name,'raw_sha256_before':before,'raw_sha256_after':sha(rawpath),'model_frames':n,'resolution':[w,h],'actual_sampling_intervals_sec':np.diff([v['t_sec'] for v in views]).tolist(),'span_sec':views[-1]['t_sec'],'existing_inference_reused':True,'new_inference_performed':False,'GT_values_loaded':False,'provided_camera_values_used':False,'XYZ_confidence_frame_pixel_exactly_same_across_three':True,'confidence_filter_or_mask_or_correction_or_downsampling':False,'confidence_color_mapping':{'scope':'all16 indoor frames pooled; one common scale across frames','type':'linear p1 to p99 clipping; same rule as KITTI but scene-specific numeric bounds','low_red':float(lo),'high_blue':float(hi),'anchors_RGB':anchors.astype(int).tolist(),'raw_percentiles':{str(p):float(v) for p,v in zip([0,1,25,50,75,99,100],np.percentile(a['confidence'],[0,1,25,50,75,99,100]))},'not_calibrated_probability':True},'coordinate_system':'first predicted camera axes x right/y down/z forward; native model gauge without GT scale','files':files})
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,Normalize
fig,ax=plt.subplots(figsize=(9,1.6),layout='constrained');cmap=LinearSegmentedColormap.from_list('confidence',anchors/255,256);fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(lo,hi),cmap=cmap),cax=ax,orientation='horizontal',label='Indoor VGGT-Omega raw depth_conf | low (red) to high (blue); global indoor p1-p99');fig.savefig(out/'confidence_legend.png',dpi=160);plt.close(fig)
(out/'README.md').write_text(f'''# 실내 TUM VGGT-Ω pointmap: RGB / frame / confidence

같은 폴더의 PLY3개는 {len(a):,}개 관측의 XYZ·confidence·frame_id·pixel_u/v가 정확히 동일하고 RGB 색상만 다르다.

- vggt_omega_raw_RGB.ply: 원본 이미지 색상.
- vggt_omega_raw_frame.ply: 프레임별 구분 색상. 색상 표는 frames.json.
- vggt_omega_raw_confidence.ply: 낮음 빨강 → 노랑 → 초록 → 청록 → 높음 파랑.

기존 실내 TUM fr3_structure_texture_far의16장 추론을 재사용했다. 중앙 프레임 간격 {np.median(np.diff([v['t_sec'] for v in views])):.3f}s. span {views[-1]['t_sec']:.3f}s. 모델 입력384×288. 추론 재실행이나 학습은 하지 않았다.

점 제거, 도로/평면 마스크, 평면 보정, ICP, GT scale 정렬을 적용하지 않았다. 모델 depth와 예측 K/pose로 복원했다. 좌표는 첫 예측 카메라 기준이며 단위는 모델 gauge다. 서로 다른 프레임에서 중복 관측한 점도 합치거나 평균하지 않았다.

confidence 색상 범위는 실내16장 전체의 p1={lo:.6f}부터 p99={hi:.6f}까지 선형이며 양 끝은 포화된다. 실외와 같은 색상 규칙이지만 실내/실외 confidence 값 분포가 달라 수치 범위는 별도다. 동일한 색을 실내/실외의 같은 raw confidence나 정확도 확률로 해석하면 안 된다. 원래 confidence scalar는 PLY에 그대로 있다. confidence_legend.png 참조.

검증: 부모 모델 cache/입력 RGB hash 일치, 카메라 재투영 roundtrip, 세 PLY binary readback/파일 크기, RGB 외 모든 속성 완전 동일, 원본 cache 불변 확인.
''');print('DONE',out,'confidence range',lo,hi)
