"""Export unmodified P11 VGGT-Omega depth and model cameras to full-scene colored PLY."""
import json,hashlib,shutil,time
from pathlib import Path
import numpy as np
from PIL import Image
R=Path(__file__).resolve().parents[3];B=R/'results/planar_residual_depth';rawdir=B/'P10raw-three-sequences-20261118-001';inp=B/'P11input-kitti-1s-20261118-001';p=rawdir/'outdoor/predictions.npz'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
prefix='kitti-1s-P11ply-'+time.strftime('%Y%m%d')+'-';i=1
while (B/(prefix+f'{i:03}')).exists():i+=1
out=B/(prefix+f'{i:03}');out.mkdir();(out/'source').mkdir();shutil.copy2(__file__,out/'source/export_ply.py');before=sha(p);manifest=json.loads((rawdir/'manifest.json').read_text());assert before==manifest['sequences'][0]['output_sha256'];z=np.load(p);depth=z['depth'][0,...,0];confidence=z['depth_conf'][0];confidence=confidence.reshape(depth.shape);ext=z['extrinsics'][0].astype('float64');K=z['intrinsics'][0].astype('float64');case=json.loads((inp/'outdoor/case.json').read_text());n,h,w=depth.shape;assert n==16 and len(case['images'])==16
DT=np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1'),('confidence','<f4'),('frame_id','<u2'),('pixel_u','<u2'),('pixel_v','<u2')]);blocks=[];framecolors=[];counts=[];centers=[];palette=np.array([[255,80,80],[255,150,40],[255,220,40],[185,240,40],[80,210,70],[30,210,150],[30,210,220],[40,150,255],[70,85,255],[150,70,255],[225,70,255],[255,70,170],[190,130,90],[80,190,180],[170,170,255],[220,220,220]],dtype='uint8');yy,xx=np.mgrid[:h,:w];uv1=np.c_[xx.ravel(),yy.ravel(),np.ones(h*w)];R0=ext[0,:,:3];t0=ext[0,:,3]

def write(path,a):
 head=('ply\nformat binary_little_endian 1.0\ncomment VGGT-Omega P11 raw depth and predicted cameras; no GT scale or plane correction\ncomment frame0 camera axes; model gauge units; y down, z forward\nelement vertex '+str(len(a))+'\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nproperty float confidence\nproperty ushort frame_id\nproperty ushort pixel_u\nproperty ushort pixel_v\nend_header\n').encode();
 with path.open('wb') as f:f.write(head);f.write(a.tobytes())
 # Independent byte-level load and finite/provenance checks.
 with path.open('rb') as f:
  while f.readline().strip()!=b'end_header':pass
  offset=f.tell();b=np.fromfile(f,dtype=DT)
 assert path.stat().st_size==offset+len(a)*DT.itemsize and len(b)==len(a);assert np.array_equal(a,b);assert all(np.isfinite(b[k]).all() for k in ['x','y','z']);return {'path':str(path.relative_to(R)),'vertices':len(a),'bytes':path.stat().st_size,'sha256':sha(path)}

(out/'frames').mkdir();files=[]
for j,v in enumerate(case['images']):
 image=R/v['processed'];assert sha(image)==manifest['sequences'][0]['input_sha256'][j];rgb=np.array(Image.open(image).convert('RGB'));assert rgb.shape==(h,w,3);rot=ext[j,:,:3];t=ext[j,:,3];camera=depth[j].ravel()[:,None]*(uv1@np.linalg.inv(K[j]).T);world=(camera-t)@np.linalg.inv(rot).T;xyz=world@R0.T+t0;center=(-np.linalg.solve(rot,t))@R0.T+t0;centers.append(center);reproj=(world@rot.T+t)@K[j].T;assert np.max(abs(reproj[:,:2]/reproj[:,2:]-uv1[:,:2]))<1e-7;assert np.max(abs((world@rot.T+t)[:,2]-depth[j].ravel()))<1e-7;valid=np.isfinite(xyz).all(1)&np.isfinite(depth[j].ravel())&(depth[j].ravel()>0);a=np.empty(int(valid.sum()),DT)
 for k,col in enumerate(['x','y','z']):a[col]=xyz[valid,k]
 for k,col in enumerate(['red','green','blue']):a[col]=rgb.reshape(-1,3)[valid,k]
 a['confidence']=confidence[j].ravel()[valid];a['frame_id']=j;a['pixel_u']=xx.ravel()[valid];a['pixel_v']=yy.ravel()[valid];blocks.append(a);fc=a.copy()
 for k,col in enumerate(['red','green','blue']):fc[col]=palette[j,k]
 framecolors.append(fc);files.append(write(out/'frames'/f'frame_{j:02}_source_{v["source_frame"]:06}.ply',a));counts.append({'frame_id':j,'source_frame':v['source_frame'],'timestamp':v['timestamp'],'t_sec':v['t_sec'],'vertices':len(a),'rejected_nonfinite_or_nonpositive':h*w-len(a),'frame_color_RGB':palette[j].tolist()})
files.insert(0,write(out/'vggt_omega_raw_RGB.ply',np.concatenate(blocks)));files.insert(1,write(out/'vggt_omega_raw_frame_colors.ply',np.concatenate(framecolors)));np.savetxt(out/'camera_centers.csv',np.c_[np.arange(n),np.array(centers)],delimiter=',',header='frame_id,x,y,z',comments='');dump(out/'frames.json',counts);assert sha(p)==before;dump(out/'manifest.json',{'status':'EXPORTED_AND_BINARY_READBACK_VERIFIED','raw_parent':rawdir.name,'input_parent':inp.name,'raw_sha256_before':before,'raw_sha256_after':sha(p),'raw_unchanged':True,'input_frames':n,'resolution':[w,h],'model_depth_and_predicted_K_pose_only':True,'GT_loaded':False,'provided_cameras_loaded_for_geometry':False,'plane_correction_or_mask':False,'confidence_filter':False,'sky_filter':False,'downsample':False,'fusion':'concatenation in common predicted camera coordinates; no averaging/registration/ICP','coordinate_system':'first model camera coordinate axes, y down z forward; global rigid transform only, native model gauge not meters','files':files})
(out/'README.md').write_text('''# KITTI P11 VGGT-Ω 원본 pointmap PLY

`vggt_omega_raw_RGB.ply`: 1초 간격16장 전체 픽셀을 모델 depth/K/pose로 복원하고 RGB 색을 입힌 공통 좌표 점군.

`vggt_omega_raw_frame_colors.ply`: 정확히 같은 XYZ를 프레임마다 다른 색으로 표시. 중첩·이중 표면·정합 차이를 관찰할 때 사용한다. 색상 표는 frames.json.

`frames/`: 각 프레임의 PLY16개. 모두 같은 좌표계로 저장해 여러 개를 동시에 열어도 된다.

GT, 평면 보정, 도로 마스크, confidence/sky 필터, 다운샘플링, ICP를 적용하지 않았다. 양의 유한 depth만 저장했다. 하늘/동적 물체/저신뢰점도 포함되어 있다. raw 배열을 변경하지 않았다. 단위는 모델 고유 gauge이며 미터 scale 정렬은 없다. 좌표는 첫 예측 카메라 기준 x오른쪽/y아래/z전방이다. Mesh가 아닌 vertex point cloud다.

PLY의 추가 속성 confidence, frame_id, pixel_u, pixel_v로 원래 관측을 추적할 수 있다. CloudCompare에서 RGB 또는 frame_id scalar field로 표시하면 된다. 모든 파일을 binary readback하고 byte 수·XYZ 유한성·원본 투영 roundtrip·RGB 입력 hash를 검증했다.

겹쳐진 모양은 정합 오차의 정성적 근거이며, GT 대비 절대 깊이 오차를 이 파일만으로 확정할 수 없다.
''');print(json.dumps({'out':str(out),'merged':files[:2],'frames':len(counts)},indent=2))
