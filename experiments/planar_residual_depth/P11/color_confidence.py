"""Recolor existing PLY only. Keep exact XYZ, confidence and observation identities."""
from pathlib import Path
import json,hashlib,time,shutil
import numpy as np
R=Path(__file__).resolve().parents[3];B=R/'results/planar_residual_depth';parent=B/'P11ply-kitti-1s-20261118-002';src=parent/'vggt_omega_raw_RGB.ply'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
prefix='kitti-1s-P11confidence-'+time.strftime('%Y%m%d')+'-';i=1
while (B/(prefix+f'{i:03}')).exists():i+=1
out=B/(prefix+f'{i:03}');out.mkdir();shutil.copy2(__file__,out/'color_confidence.py');before=sha(src);DT=np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1'),('confidence','<f4'),('frame_id','<u2'),('pixel_u','<u2'),('pixel_v','<u2')])
with src.open('rb') as f:
 lines=[]
 while True:
  l=f.readline();assert l;lines.append(l)
  if l.strip()==b'end_header':break
 header=b''.join(lines);a=np.fromfile(f,DT)
assert len(a)==int(next(l for l in lines if l.startswith(b'element vertex ')).split()[-1]);conf=a['confidence'];assert np.isfinite(conf).all();lo,hi=np.percentile(conf,[1,99]);t=np.clip((conf-lo)/(hi-lo),0,1);anchors=np.array([[230,30,30],[255,220,0],[30,200,60],[0,200,230],[30,60,230]],float);colors=np.stack([np.interp(t,np.linspace(0,1,len(anchors)),anchors[:,k]) for k in range(3)],axis=1);b=a.copy()
for k,col in enumerate(['red','green','blue']):b[col]=np.rint(colors[:,k]).astype('uint8')
newheader=header.replace(b'end_header\n',(f'comment confidence RGB global p01={lo:.9g} red to p99={hi:.9g} blue, linear with clipping; no points removed\nend_header\n').encode());dest=out/'vggt_omega_raw_confidence.ply'
with dest.open('wb') as f:f.write(newheader);f.write(b.tobytes())
with dest.open('rb') as f:
 while f.readline().strip()!=b'end_header':pass
 read=np.fromfile(f,DT)
assert np.array_equal(read,b)
for key in DT.names:
 if key not in ['red','green','blue']:assert np.array_equal(read[key],a[key])
assert sha(src)==before
stats={'status':'CONFIDENCE_COLORED_AND_READBACK_VERIFIED','parent':parent.name,'parent_file':str(src.relative_to(R)),'parent_sha256':before,'output':str(dest.relative_to(R)),'output_sha256':sha(dest),'vertices':len(b),'XYZ_and_confidence_and_provenance_exactly_unchanged':True,'point_filtering':False,'color_normalization':'global linear p1 to p99, clipped, same mapping all16 frames','confidence_low_red':float(lo),'confidence_high_blue':float(hi),'raw_confidence_percentiles':{str(p):float(v) for p,v in zip([0,1,5,25,50,75,95,99,100],np.percentile(conf,[0,1,5,25,50,75,95,99,100]))},'color_anchors':anchors.astype(int).tolist(),'not_calibrated_accuracy_probability':True};(out/'manifest.json').write_text(json.dumps(stats,indent=2)+'\n');(out/'README.md').write_text(f'''# VGGT-Ω raw depth confidence 색상

동일 KITTI P11 원본 점군 {len(b):,}개. XYZ·confidence·frame_id·pixel_u/v는 기존 RGB PLY와 byte 값이 같다. RGB 색상만 바꿨다. 점 제거/다운샘플/GT/보정은 없다.

색상: 낮음 빨강 → 노랑 → 초록 → 청록 → 높음 파랑. 16장 전체의 raw depth_conf를 하나의 선형 scale에 적용했다. 하한 p1={lo:.6f}, 상한 p99={hi:.6f}이며 바깥은 endpoint 색으로 포화시켰다. confidence가 확률이나 실제 GT 오차를 뜻하지는 않는다. 원래 raw confidence scalar도 PLY에 보존했다.

CloudCompare에서 RGB로 보면 사전 지정 색상, confidence scalar field로 보면 원래 값을 확인할 수 있다. 색상 범례는 confidence_legend.png.
''')
# Scientific color legend, using the same numeric mapping as the PLY.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,Normalize
cm=LinearSegmentedColormap.from_list('confidence',anchors/255,256);fig,ax=plt.subplots(figsize=(9,1.6),layout='constrained');fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(lo,hi),cmap=cm),cax=ax,orientation='horizontal',label='VGGT-Omega raw depth_conf | low (red) -> high (blue); global p1-p99 clipping');fig.savefig(out/'confidence_legend.png',dpi=160);plt.close(fig);print(json.dumps(stats,indent=2))
