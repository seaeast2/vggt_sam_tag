"""Temporal pilot report, real RGB/depth comparison movies and finite fused ROI pointmaps."""
import sys,json,csv,shutil
from pathlib import Path
import numpy as np,cv2
from PIL import Image,ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'results/planar_residual_depth'
sys.path.insert(0,str(Path(__file__).parents[1]/'P3'))
from solver import rays,depths
import importlib.util
spec=importlib.util.spec_from_file_location('p3_helpers',ROOT/'experiments/planar_residual_depth/P3/fit.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper);plane=helper.plane

def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def angle(a,b):return float(np.degrees(np.arccos(np.clip(abs(a@b/np.linalg.norm(a)/np.linalg.norm(b)),0,1))))
def colormap(data,valid,lo,hi,name):
 cm=plt.get_cmap(name);normalized=np.clip((data-lo)/max(1e-12,hi-lo),0,1);rgb=np.rint(cm(np.nan_to_num(normalized,nan=0))[:,:,:3]*255).astype('uint8');rgb[~valid]=[150,150,150];return rgb

def global_reference(case,D,mask):
 points=[];per=[]
 for j in [0,4,8,12,15,19]:
  valid=mask[j]&np.isfinite(D[j])&(D[j]>0);y,x=np.where(valid);uv=np.c_[x,y];X=np.array(case['images'][j]['C'])+D[j][valid,None]*rays(uv,case['images'][j]);p=plane(X);n=p/np.linalg.norm(p);r=X@n+1/np.linalg.norm(p);per.append({'frame':j,'GT_pixels':len(X),'world_reference_plane_p':p.tolist(),'local_GT_plane_RMSE_m':float(np.sqrt(np.mean(r**2)))});step=max(1,len(X)//1000);points.append(X[::step])
 X=np.concatenate(points);p=plane(X);n=p/np.linalg.norm(p);r=X@n+1/np.linalg.norm(p)
 return {'role':'post-fit assumption diagnostic; cannot causally separate mask errors, real nonplanarity, sensor projection and camera errors','frames':[0,4,8,12,15,19],'global_plane_p':p.tolist(),'global_GT_plane_RMSE_m':float(np.sqrt(np.mean(r**2))),'per_frame':per}

def write_fused(fitcase,case,dest):
 data=np.load(fitcase/'inputs.npz');mask=data['masks'];views=case['images'];alluv=[];RGB=[];frame=[];Q=[]
 for j,v in enumerate(views[:16]):
  yy,xx=np.where(mask[j]);uv=np.c_[xx,yy];alluv.append(uv);RGB.append(np.array(Image.open(ROOT/v['processed']).convert('RGB'))[mask[j]]);frame.append(np.full(len(uv),j));Q.append(rays(uv,v))
 uv=np.concatenate(alluv);color=np.concatenate(RGB);frames=np.concatenate(frame);folder=dest/'fused_ROI_pointmaps';folder.mkdir(exist_ok=False);dtype=np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1'),('pixel_u','<i4'),('pixel_v','<i4'),('frame_id','<i4')])
 for arm in ['B0','B1','B2','B3','P8']:
  z=np.load(fitcase/'N16'/arm/'depths.npz');X=np.concatenate([np.array(views[j]['C'])+z['corrected_depth'][j][mask[j],None]*Q[j] for j in range(16)]);data=np.empty(len(X),dtype=dtype)
  for k,name in enumerate(['x','y','z']):data[name]=X[:,k]
  for k,name in enumerate(['red','green','blue']):data[name]=color[:,k]
  data['pixel_u']=uv[:,0];data['pixel_v']=uv[:,1];data['frame_id']=frames
  with (folder/(arm+'.ply')).open('wb') as f:
   f.write(('ply\nformat binary_little_endian 1.0\ncomment finite_ROIs_only_16_mapping_frames_reference_camera0_coordinates\nelement vertex '+str(len(X))+'\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nproperty int pixel_u\nproperty int pixel_v\nproperty int frame_id\nend_header\n').encode());data.tofile(f)
 dump(folder/'provenance.json',{'plane_id':case['plane_id'],'scope':'finite16 mapping-frame ROIs, no point merging or mesh, common provided-pose reference-camera coordinates','pixel_frame_identifiers_preserved':True,'frame_sources':[v['source'] for v in views[:16]],'polygons':case['polygons'][:16],'views':views[:16],'raw_cache':'P10raw-three-sequences-20261117-001','vertex_count_per_arm':len(uv),'mask_revision':case.get('mask_revision')})


def render_sequence(case,fitcase,out,metrics):
 name=case['sample'];gd=np.load(out/'GT_arrays.npz');D=gd['GT_depth'];mm=gd['masks'];views=case['images'];baseline=np.load(fitcase/'inputs.npz')['baseline'];h,w=baseline[0].shape;arms={a:np.load(fitcase/'N16'/a/'depths.npz') for a in ['B0','B1','B2','B3','P8']};valid0=mm[0]&np.isfinite(D[0]);gdvals=D[np.isfinite(D)&mm];predvals=np.concatenate([z['corrected_depth'][mm[:16]] for z in arms.values()]);lo,hi=np.percentile(np.r_[gdvals,predvals],[1,99]);allerrors=[]
 for a,z in arms.items():
  v=mm[:16]&np.isfinite(D[:16]);allerrors.append(abs(z['corrected_depth'][v]-D[:16][v]))
 bound=max(1e-5,float(np.percentile(np.concatenate(allerrors),95)))
 # Fixed ROI source0 comparisons, same colors and evaluation mask across all methods.
 fig,axs=plt.subplots(2,6,figsize=(17,6),layout='constrained');rgb=np.array(Image.open(ROOT/views[0]['processed']).convert('RGB'));axs[0,0].imshow(rgb);p=case['polygons'][0];axs[0,0].plot([x[0] for x in p]+[p[0][0]],[x[1] for x in p]+[p[0][1]],'r-');axs[0,0].set_title('Source RGB / finite ROI');dc=plt.get_cmap('viridis').copy();dc.set_bad('#999999');ec=plt.get_cmap('RdBu_r').copy();ec.set_bad('#999999');axs[1,0].imshow(np.where(valid0,D[0],np.nan),cmap=dc,vmin=lo,vmax=hi);axs[1,0].set_title('Sensor reference; gray=missing')
 for col,(a,z) in enumerate(arms.items(),1):
  d=z['corrected_depth'][0];im=axs[0,col].imshow(np.where(mm[0],d,np.nan),cmap=dc,vmin=lo,vmax=hi);axs[0,col].set_title(a);ie=axs[1,col].imshow(np.where(valid0,d-D[0],np.nan),cmap=ec,vmin=-bound,vmax=bound);m=next(m for m in metrics if m['N']==16 and m['arm']==a);axs[1,col].set_title(f"MAE {m['source0_MAE_m']*100:.2f} cm\nnormal {m['normal_error_deg']:.2f} deg")
 for ax in axs.ravel():ax.axis('off')
 fig.colorbar(im,ax=axs[0,1:],label='camera-Z m',shrink=.65);fig.colorbar(ie,ax=axs[1,1:],label='prediction - sensor m',shrink=.65);fig.suptitle(name+' N16 | provided poses, fixed16 RGB context, finite floor ROI');fig.savefig(out/'source0_depth_GT_errors.png',dpi=140);plt.close(fig)
 # Movies expose every sampled RGB frame, including the four unseen-by-VGGT future frames.
 for arm in ['B3','P8']:
  z=arms[arm];p=z['p'];hasplane=np.isfinite(p).all();writer=cv2.VideoWriter(str(out/(arm+'_N16_comparison.mp4')),cv2.VideoWriter_fourcc(*'mp4v'),5,(5*w,h+42));assert writer.isOpened();gif=[]
  for j,v in enumerate(views):
   raw=np.array(Image.open(ROOT/v['processed']).convert('RGB'));im=Image.fromarray(raw.copy());poly=case['polygons'][j];ImageDraw.Draw(im).line([tuple(x) for x in poly]+[tuple(poly[0])],fill='red',width=2);rgb=np.array(im);valid=mm[j]&np.isfinite(D[j])&(D[j]>0)
   if j<16:base=baseline[j];cor=z['corrected_depth'][j];rawvis=colormap(base,mm[j],lo,hi,'viridis')
   else:
    rawvis=np.full((h,w,3),150,'uint8');base=np.full((h,w),np.nan);cor=np.full((h,w),np.nan);yy,xx=np.where(mm[j])
    if hasplane:cor[mm[j]]=depths(p,np.c_[xx,yy],v)
   cv=mm[j]&np.isfinite(cor)&(cor>0);predvis=colormap(cor,cv,lo,hi,'viridis');gtvis=colormap(D[j],valid,lo,hi,'viridis');errvis=colormap(cor-D[j],valid&cv,-bound,bound,'RdBu_r');canvas=Image.new('RGB',(5*w,h+42),'white');d=ImageDraw.Draw(canvas)
   for k,tile in enumerate([rgb,rawvis,predvis,gtvis,errvis]):canvas.paste(Image.fromarray(tile),(k*w,42))
   titles=['RGB / red ROI','raw x fixed scale' if j<16 else 'future: no raw inference',arm+' N16' if j<16 else arm+' plane extrapolation','sensor reference','prediction - sensor']
   for k,t in enumerate(titles):d.text((k*w+4,23),t,fill='black')
   d.text((4,3),f'{name} frame{j:02} t={v["t_sec"]:.3f}s '+v['role']+' | provided camera poses, GT evaluation only',fill='black');writer.write(cv2.cvtColor(np.array(canvas),cv2.COLOR_RGB2BGR))
   if arm=='P8':
    small=Image.fromarray(np.concatenate([rgb,predvis,gtvis],axis=1));small.thumbnail((960,240));gif.append(small)
  writer.release();cap=cv2.VideoCapture(str(out/(arm+'_N16_comparison.mp4')));frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));cap.release();assert frames==20,(name,arm,frames)
  if gif:gif[0].save(out/'P8_RGB_corrected_sensor.gif',save_all=True,append_images=gif[1:],duration=200,loop=0)
 # Cumulative curves: same source0 pixels, never choose the best prefix using GT.
 fig,axs=plt.subplots(1,3,figsize=(14,4),layout='constrained')
 for a in ['B0','B2','B3','P8']:
  rr=[m for m in metrics if m['arm']==a];axs[0].plot([m['N'] for m in rr],[100*m['source0_MAE_m'] for m in rr],'.-',label=a);axs[1].plot([m['N'] for m in rr],[m['normal_error_deg'] for m in rr],'.-',label=a);axs[2].plot([m['N'] for m in rr],[100*m['future_GT_mean_MAE_m'] if m['future_GT_mean_MAE_m'] is not None else np.nan for m in rr],'.-',label=a)
 for ax in axs:ax.set_xticks([2,4,8,16]);ax.set_xlabel('mapping views in plane optimization');ax.legend()
 axs[0].set_ylabel('fixed source0 ROI MAE cm');axs[1].set_ylabel('normal error deg');axs[2].set_ylabel('future4 ROI mean depth MAE cm');fig.suptitle(name+' | offline fixed raw context16; failures retain raw');fig.savefig(out/'prefix_curves.png',dpi=140);plt.close(fig);write_fused(fitcase,case,out)


def main():
 out=BASE/sys.argv[1];mf=json.loads((out/'manifest.json').read_text());fit=BASE/mf['parent_fit'];obs=BASE/json.loads((fit/'manifest.json').read_text())['observation_parent'];summary=json.loads((out/'summary.json').read_text());shutil.copy2(__file__,out/'source/report.py');flat=[]
 for r in summary:
  name=r['sample'];case=json.loads((fit/name/'case.json').read_text());data=np.load(out/name/'GT_arrays.npz');diag=global_reference(case,data['GT_depth'],data['masks']);dump(out/name/'global_GT_plane_diagnostic.json',diag);r['global_GT_plane_diagnostic']=diag;render_sequence(case,fit/name,out/name,r['metrics'])
  for m in r['metrics']:flat.append({k:v for k,v in m.items() if not isinstance(v,(list,dict))})
 dump(out/'report_summary.json',summary)
 with (out/'per_prefix_metrics.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
 report=ROOT/'docs/experiments/planar_residual_depth/P10'/out.name/'report.md';report.parent.mkdir(parents=True,exist_ok=True);lines=['# P10 실내·실외·드론 연속 영상: 다중시점 관측 누적 파일럿','', '**실내/실외/드론 각1개 실제 연속 시퀀스의 짧은 구간에서 VGGT-Ω 추론·평면 보정·센서 기준 평가·영상 출력을 완료했다. 총60개 RGB 샘플(시퀀스당20개),48개 mapping 이미지(각16),12개 미래 보류 이미지(각4)다. 카메라 자세는 제공 궤적으로 고정한 통제 조건이며 RGB만으로 자세까지 추정하는 배포/장기 SLAM 검증은 아니다.**','',
 '## 데이터와 시간 분할','',
 '- 실내: TUM RGB-D `fr3_structure_texture_far`, 첫 RGB 이후2s에서5Hz 근처20장, 약3.8s. 등록된 Kinect depth를 사후 기준으로 사용했다.',
 '- 실외: KITTI Raw `2011_09_26_drive_0005_sync`, image02의 frame50–88 step2, 약3.9s. GPS/IMU OXTS·공식 외부보정으로 카메라를 구성하고 동기 Velodyne을 사후 투영했다.',
 '- 드론: EuRoC MAV `V1_01_easy`의 cam0, 카메라/pose 공통 시작 이후10s에서5Hz20장. 실내 MAV 촬영·흑백 카메라를 RGB3채널로 복제했다. radial-tangential 왜곡을 제공 K로 제거했고 Leica MS50 room scan을 사후 투영했다. 야외 드론·자율비행 성능을 입증한 샘플로 표현하지 않는다.',
 '', 'EuRoC 기존 공식 서버 이동/접속 실패로 공개 mirror revision `d05bd6a92a5f456562153f83d65805fea27cb57d`를 사용했고 archive SHA256 `a920fe5b5e69a6ad19b32f1cfaf90ac2f59d45dfd1ca18cc9722e07684ba45fa`를 확인했다. 모든 원본 archive/source hash는 P10input manifest에 있다.',
 '', '## 통제 조건과 실행 이력','',
 'VGGT-Ω 원본 checkpoint로 mapping16장만 공동 추론하고 future4장은 입력하지 않았다. 384 long-side, 공식 max_size 전처리, 실제 host GPU에서 수행했다. peak allocated VRAM은 실내5310002176B, 실외5071418368B, 드론5200589824B로 최대4.95GiB였다. 각 prefix의 raw는 같은16-frame cache이므로 N2도 이미16개 mapping RGB 문맥을 가진 오프라인 보정 비교다. 실시간/인과적 네트워크 추론 누적 실험으로 부르지 않는다.',
 '', '제공 K/pose를 camera-error control로 사용했다. 모델 입력은 RGB뿐이지만 보정은 외부 제공 자세를 쓰므로 전체 파이프라인이 RGB-only SLAM이라고 주장하지 않는다. depth PNG/Velodyne/room scan 값은 첫 fit 이후에만 평가에 사용했다. 각 clip scale은 mapping0–15의 FIT SIFT 삼각측량과 raw depth 비율 중앙값 하나로 고정했다. future16–19나 depth GT로 scale을 정하지 않았다.',
 '', '관측은 원본 undistorted RGB crop을 긴변768 이하로 두고, global SIFT mutual ratio0.75·epipolar1px로 scale을 구했다. 평면 보정에는 temporal Shi–Tomasi/LK와 FB0.5px·epipolar1px를 사용했다. 매핑 프레임에서만 재검출하고 future에서는 새 track을 만들지 않았다. Track seed ID hash로 FIT/보류를 고정했다. Support는 FIT12/보류6 track, 면적10%, 중앙 시차1°, Jacobian rank를 유지했다. 같은 plane의 모든 관측쌍에 기존 양방향 Huber1px 목적함수를 적용했다. 같은 track 내 시간 관측은 상관되며 독립 표본으로 세지 않는다.',
 '', '1. `P10fit-001`: 첫 프레임의 FIT hull을 요구한 원래 gate. 실외 source0 추적 전달이 없어 전 prefix 미지원으로 보존했다.',
 '2. `P10fit-002`: 깊이 GT를 읽기 전에, 같은 물리 평면의 다른 시점에서도 면적10%를 충족할 수 있게 최대 per-view hull gate로 바꿨다. 숫자 기준·frames·mask·raw·solver는 유지했고 원래 실패도 보존했다. 실외 N16에서 처음 support를 확보했다. 첫 GT 평가는 `P10eval-001`이다.',
 '3. 첫 GT/영상 감사에서 드론 후반 fixed-screen ROI가 보이는 계단형 비평면 물체를 포함함을 확인했다. 원래 마스크/결과를 보존하고 드론 frame1–19 polygon만 RGB에서 물체를 피해 움직이도록1회 수정했다. source0 polygon은 동일하다. `P10masks-002`→`P10obs-002`→`P10fit-003`→현재 `P10eval-002`가 mask-audited canonical이다. 이 수정은 GT를 본 뒤의 후행 수정이므로 blind/independent test로 표현하지 않는다. 재검출 track과 scale의3px 보류 제외가 함께 달라질 수 있어 파생 scale값도 별도 기록했다. 수정 후 GT 결과를 보고 다시 영역을 조절하지 않았다.',
 '', '이번 단계는 관측 누적 효과를 기존 B2/B3/P8 feature 목적함수로 통제한 P10이다. P9의 direct RGB 광도 목적함수를20-view로 확장하거나 카메라 자세를 최적화한 실험은 아니다.',
 '', '## N16 고정 source0 ROI 결과','', '| 시퀀스 | 유효 센서 기준 픽셀 | B0 원본×scale MAE(cm) | B1 | B2 거리만 | B3 방향·거리 | P8 방향1° 제한 |','|---|---:|---:|---:|---:|---:|---:|']
 for r in summary:
  d={m['arm']:m for m in r['metrics'] if m['N']==16};lines.append('| '+r['sample']+' | '+str(r['GT_reference']['source0_GT_pixels'])+' | '+' | '.join(f"{100*d[a]['source0_MAE_m']:.3f}" for a in ['B0','B1','B2','B3','P8'])+' |')
 lines+=['','세 데이터셋의 오차를 하나로 평균하지 않았다. GT는 시퀀스별 센서 기준이며, KITTI/EuRoC는 camera pixel에 최소Z로 투영한 근사 참조다. 원래 카메라 시점의 occlusion-perfect dense depth나 공식 벤치마크 점수로 표현하지 않는다. KITTI motion compensation은 수행하지 않았고 GT 결측은 채우지 않았다.','',
 '## 관측2→4→8→16장 변화','', '| 시퀀스 | N | B2 MAE(cm) | B3 MAE(cm) | P8 MAE(cm) | support | FIT/보류 track | 중앙 시차(°) |','|---|---:|---:|---:|---:|---|---:|---:|']
 for r in summary:
  name=r['sample'];prefix=json.loads((fit/name/'prefix_summary.json').read_text())
  for N in [2,4,8,16]:
   dd={m['arm']:m for m in r['metrics'] if m['N']==N};p=next(p for p in prefix if p['N']==N);s=p['support'];par=s['median_FIT_pair_parallax_deg'];lines.append(f"| {name} | {N} | {dd['B2']['source0_MAE_m']*100:.3f} | {dd['B3']['source0_MAE_m']*100:.3f} | {dd['P8']['source0_MAE_m']*100:.3f} | {'passed' if s['passed'] else 'raw retained'} | {s['FIT_tracks']}/{s['heldout_tracks']} | {par if par is not None else '—'} |")
 lines+=['','**사진/관측 수가 증가할수록 오차가 단조롭게 줄지는 않았다.** 실외2/4/8장은 미지원이라 raw 유지이며, 그 수치를 작은 prefix의 보정 성공으로 세지 않는다. 실내/드론에서도 prefix 변화가 비단조적이다. 시간 간격·시차·같은 물리점의 생존·재검출·조명·제공 자세 오차·실제 평면 가정이 함께 작용하므로 개수만의 인과 효과를 확정하지 않는다.','',
 '## 미래4개 시점 평가','', '| 시퀀스 | 방법 N16 | 미래4 ROI 평균 MAE(cm) | 미래 track-heldout 쌍 수 | 미래 재투영 median(px) |','|---|---|---:|---:|---:|']
 for r in summary:
  for m in r['metrics']:
   if m['N']!=16 or m['arm'] not in ['B1','B2','B3','P8']:continue
   fp=m['future_track_heldout'];mae=m['future_GT_mean_MAE_m'];lines.append(f"| {r['sample']} | {m['arm']} | {mae*100 if mae is not None else '—'} | {fp['pairs']} | {fp['median_px'] if fp['median_px'] is not None else '—'} |")
 lines+=['','future RGB4장은 VGGT와 plane fit에 쓰지 않았다. 보정한 공유 평면을 제공 future camera에 투영해 sensor reference와 비교했다. future raw inference는 없으므로 B0 future depth 비교는 없다. 미래 RGB track-heldout은 mapping15→future로 고정했다. 실내는0쌍, 실외는1쌍으로 이 재투영만으로 일반화를 주장할 수 없다. 센서 depth 비교와 RGB 대응 일치를 구분했다.','',
 '## GT 평면 가정·방향 불확실성','', '| 시퀀스 | source0 GT plane RMS(cm) | GT법선 block bootstrap p95(°) | sampled 전체 world GT plane RMS(cm) |','|---|---:|---:|---:|']
 for r in summary:
  q=r['GT_reference'];g=r['global_GT_plane_diagnostic'];lines.append(f"| {r['sample']} | {100*q.get('GT_plane_RMSE_m',0):.3f} | {q.get('GT_normal_change_p95_deg')} | {100*g['global_GT_plane_RMSE_m']:.3f} |")
 lines+=['','Source0 GT 기준 법선은 같은 ROI에 robust plane을 적합하고16px 공간 block bootstrap50회로 변동을 기록했다. 전체 plane 진단은 frames0/4/8/12/15/19의 센서 기준점을 제공 자세로 묶은 후 적합했다. 이 RMS는 실제 비평면성·마스크 가림·센서 투영·자세 오차를 분리하지 못한다. GT 평면에 맞지 않는 사례나 점을 사후 삭제하지 않았다. 최초 잘못 포함된 드론 객체의 진단은 이전 run에 남아 있다.','',
 '## 입력 영상·결과 영상과 pointmap','']
 for r in summary:
  name=r['sample'];rel=str((out/name).relative_to(ROOT));lines+=['',f'- **{name}**: source RGB `results/planar_residual_depth/P10input-three-sequences-20261117-001/{name}/source_clip.mp4`.',f'  비교영상: `{rel}/B3_N16_comparison.mp4`, `{rel}/P8_N16_comparison.mp4`; GIF `{rel}/P8_RGB_corrected_sensor.gif`.',f'  그림: `{rel}/source0_depth_GT_errors.png`, `{rel}/prefix_curves.png`.']
 lines+=['','비교영상은 실제 sampled RGB, 원본×scale ROI depth, 보정/외삽 depth, 센서 기준 depth, signed error를 공통 색 범위로 보여준다. 회색은 ROI 밖/결측/원본 미추론이다. 마지막4장은 raw 미추론을 명시하고 plane extrapolation만 표시한다. `fused_ROI_pointmaps/*.ply`는16장 유한 ROI의 파생점이며 RGB·pixel_u/v·frame_id·공유plane/카메라 계보를 보존했다. 전체 지도 정확도·메시·SLAM 성공을 뜻하지 않는다.','',
 '## 결론과 남은 한계','',
 '실제 연속 영상3개를 처리했고, 초기/후반 support 전환과 관측 누적의 비단조성을 확인했다. 실내 source0는 거리만 보정한 B2가 가장 작고, 드론은 원본 대비 보정 이득이 있으나 관측을 더 넣을수록 항상 좋아지지는 않는다. 실외는 관측 support가 생긴 N16에서도 source0 깊이가 악화해 현재 규칙의 적용 한계를 드러냈다. GT로 scene별 최적 arm이나 최적 prefix를 골라 실제 성능으로 제시하지 않는다.',
 '', '카메라 제공 자세를 RGB 추정 자세로 바꾸는 조건, 긴 시간/루프 폐쇄, 야외 드론, 동적 객체 분할, plane ID 자동 유지, P9 광도 loss의 시간 확장, 독립 시퀀스 반복은 아직 수행하지 않았다. 이번 결과는 카메라 오차를 통제한 약4초 pilot이며 end-to-end RGB-only 배포 검증이 아니다.']
 report.write_text('\n'.join(lines)+'\n');dump(out/'report_manifest.json',{'report':str(report.relative_to(ROOT)),'status':'THREE_PILOT_SEQUENCES_REPORTED','canonical_fit':fit.name,'posthoc_RGB_mask_revision':True})
 with (ROOT/'docs/experiments/planar_residual_depth/index.md').open('a') as f:f.write(f'\n- P10 실내/실외/드론 연속영상 pilot 완료 — [보고서](P10/{out.name}/report.md), 제공자세 통제, 2/4/8/16 관측·후행 RGB mask 수정 이력 보존.\n')
 print(report,flush=True)
if __name__=='__main__':main()
