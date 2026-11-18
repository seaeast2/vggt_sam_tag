"""Move mixed-depth background-side edge pixels to a local background depth."""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'results/planar_residual_depth'
OUT = BASE/'P13-indoor-20261118-001'
PARENT = BASE/'P10raw-three-sequences-20261118-003/indoor/predictions.npz'
VIEWS = json.loads((BASE/'P10-indoor-2s-input-20261118-001/indoor/case.json').read_text())['images']
RAW_PLY = BASE/'P10-indoor-2s-ply-20261118-001/vggt_omega_raw_RGB.ply'

def dump(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2)+'\n')

def detect(rgb, depth, confidence, protection, relaxed=False):
    lab = cv2.cvtColor(rgb.astype(np.float32)/255, cv2.COLOR_RGB2LAB)
    smooth = cv2.GaussianBlur(lab,(3,3),.65)
    gx = cv2.Sobel(smooth,-1,1,0,ksize=3,scale=.125)
    gy = cv2.Sobel(smooth,-1,0,1,ksize=3,scale=.125)
    strength = np.sqrt(np.sum(gx**2+gy**2,axis=-1))
    y,x = np.where(strength >= (2.5 if relaxed else 4))
    direction = np.argmax(gx[y,x]**2+gy[y,x]**2,axis=1)
    dx,dy = gx[y,x,direction],gy[y,x,direction]
    length = np.maximum(np.hypot(dx,dy),1e-5)
    dx,dy = dx/length,dy/length
    h,w = depth.shape
    def sampling(distances, tangent):
        u = np.rint(x[:,None]+dx[:,None]*distances-dy[:,None]*tangent).astype(int)
        v = np.rint(y[:,None]+dy[:,None]*distances+dx[:,None]*tangent).astype(int)
        valid = ((u>=0)&(u<w)&(v>=0)&(v<h)).all(1)
        return np.clip(v,0,h-1), np.clip(u,0,w-1),valid
    distance = np.tile(np.arange(7,12) if relaxed else np.arange(5,10),3)
    tangent = np.repeat([-1,0,1],5)
    a,b,valid1 = sampling(-distance,tangent)
    c,d,valid2 = sampling(distance,tangent)
    left,right = depth[a,b],depth[c,d]
    zl,zr = np.median(left,1),np.median(right,1)
    cl,cr = np.median(lab[a,b],1),np.median(lab[c,d],1)
    far_right = zr>zl
    far,near = np.maximum(zr,zl),np.minimum(zr,zl)
    gap = far-near
    base = valid1 & valid2 & (near>0) & (gap >= (.10 if relaxed else .15)*near)
    base &= (np.percentile(abs(left-zl[:,None]),80,axis=1)<= (.055 if relaxed else .035)*zl)
    base &= (np.percentile(abs(right-zr[:,None]),80,axis=1)<= (.055 if relaxed else .035)*zr)
    base &= np.linalg.norm(cl-cr,axis=1) >= (10 if relaxed else 15)
    bg = np.where(far_right[:,None],cr,cl)
    fg = np.where(far_right[:,None],cl,cr)
    corrected = depth.copy()
    chosen = np.zeros_like(depth,dtype=bool)
    score_map = np.full_like(depth,-np.inf)
    for step in range(7 if relaxed else 5):
        v,u,valid = sampling((step*np.where(far_right,1,-1))[:,None],0)
        v,u = v[:,0],u[:,0]
        color = lab[v,u]
        bg_error = np.linalg.norm(color-bg,axis=1)
        fg_error = np.linalg.norm(color-fg,axis=1)
        z = depth[v,u]
        ok = base & valid & (bg_error <= (22 if relaxed else 16)) & (bg_error + (3 if relaxed else 6) <= fg_error)
        margin = .08 if relaxed else .12
        ok &= (z>=near+margin*gap)&(z<=far-margin*gap)&(confidence[v,u]<protection)
        score = strength[y,x]-2*step-bg_error
        ids = np.flatnonzero(ok)
        for j in ids[np.argsort(score[ids])]:
            yy,xx = v[j],u[j]
            if score[j]>score_map[yy,xx]:
                score_map[yy,xx]=score[j]
                corrected[yy,xx]=far[j]
                chosen[yy,xx]=True
    return corrected,chosen

def fit():
    OUT.mkdir()
    pred = np.load(PARENT)
    raw = pred['depth'][0,...,0]
    conf = pred['depth_conf'][0].reshape(raw.shape)
    cap = float(np.quantile(conf,.70))
    dump('protocol.json',dict(experiment='P13', confidence_protection_p70=cap, raw_sha256=hashlib.sha256(PARENT.read_bytes()).hexdigest(),
        color='Lab, background similarity and foreground margin', reference_strips=('7–11' if ARGS.relaxed else '5–9')+' pixels along normal, ±1 tangent',
        stable_side_p80_relative_deviation=.055 if ARGS.relaxed else .035, gap_relative_min=.10 if ARGS.relaxed else .15, middle_depth_fraction=[.08,.92] if ARGS.relaxed else [.12,.88],
        correction_band_pixels=6 if ARGS.relaxed else 4, geometry='single-view, no plane fitting', GT_used_for_fit=False))
    corrected, selected = zip(*(detect(np.array(Image.open(ROOT/v['processed']).convert('RGB')),z,c,cap,ARGS.relaxed)
                                for v,z,c in zip(VIEWS,raw,conf)))
    corrected,selected = np.stack(corrected),np.stack(selected)
    if ARGS.relaxed:
        previous = np.load(BASE/'P13-indoor-20261118-001/depths.npz')
        assert np.array_equal(previous['raw'],raw)
        original = previous['selected']
        corrected[original] = previous['corrected'][original]
        selected |= original
        dump('relaxed_selection.json', dict(parent='P13-indoor-20261118-001',
              gradient_min=2.5, side_color_difference_min=10, background_color_error_max=22,
              background_color_margin=3, gap_min=.10, strip_deviation=.055,
              stripe_offsets=[7,11], band=6, middle_fraction=[.08,.92],
              original_selected=int(original.sum()), added=int((selected&~original).sum())))
    assert np.array_equal(raw[~selected],corrected[~selected])
    assert np.all(corrected[selected]>raw[selected]) and np.all(conf[selected]<cap)
    np.savez_compressed(OUT/'depths.npz',raw=raw,corrected=corrected,residual=corrected-raw,selected=selected)
    print('selected',int(selected.sum()),flush=True)

def evaluate():
    data=np.load(OUT/'depths.npz')
    raw,corrected,selected=[data[k] for k in ['raw','corrected','selected']]
    source=ROOT/'data/planar_residual_depth/P10/source/rgbd_dataset_freiburg3_structure_texture_far'
    index=[line.split() for line in (source/'depth.txt').read_text().splitlines() if line and not line.startswith('#')]
    gt=[]
    h,w=raw.shape[1:]
    for v in VIEWS:
        stamp,file=min(index,key=lambda a:abs(float(a[0])-float(v['timestamp'])))
        assert abs(float(stamp)-float(v['timestamp']))<.03
        gt.append(np.asarray(Image.open(source/file).resize((w,h),Image.Resampling.NEAREST))/5000.)
    gt=np.stack(gt)
    y,x=np.indices((h,w))
    cal=((x//16+3*(y//16))%5)==1
    valid=gt>0
    calibration=valid&cal&~selected
    scale=float(np.median(gt[calibration]/raw[calibration]))
    a,b=[abs(v*scale-gt) for v in [raw,corrected]]
    def stats(mask, a=a, b=b):
        return dict(pixels=int(mask.sum()),raw_MAE_cm=float(100*a[mask].mean()) if mask.any() else None,
                    corrected_MAE_cm=float(100*b[mask].mean()) if mask.any() else None,
                    improved_fraction=float((b[mask]<a[mask]).mean()) if mask.any() else None)
    mask=valid&~cal&selected
    result=dict(scale=scale,selected=int(selected.sum()),changed_test=stats(mask),all_test=stats(valid&~cal),
                frames=[stats(mask[i],a[i],b[i]) for i in range(len(raw))])
    if ARGS.relaxed:
        original = np.load(BASE/'P13-indoor-20261118-001/depths.npz')['selected']
        result['added_test'] = stats(mask & ~original)
    dump('evaluation.json',result)
    print(result['changed_test'],result['all_test'],flush=True)

def export():
    z=np.load(OUT/'depths.npz')
    raw,corrected,selected=[z[k] for k in ['raw','corrected','selected']]
    n,h,w=raw.shape
    sheet=Image.new('RGB',(4*w,4*(h+24)),'white')
    draw=ImageDraw.Draw(sheet)
    for i,v in enumerate(VIEWS):
        rgb=np.array(Image.open(ROOT/v['processed']).convert('RGB'))
        rgb[selected[i]]=(255,150,20)
        x,y=i%4*w,i//4*(h+24)
        sheet.paste(Image.fromarray(rgb),(x,y+24))
        draw.text((x+5,y+5),f'{i:02} | {selected[i].sum()} pixels',fill='black')
    sheet.save(OUT/'selected_regions.jpg',quality=94)
    dtype=np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1'),
                    ('confidence','<f4'),('frame_id','<u2'),('pixel_u','<u2'),('pixel_v','<u2')])
    with RAW_PLY.open('rb') as stream:
        header=[]
        for line in stream:
            if not line.startswith(b'comment'):
                header.append(line)
            if line.strip()==b'end_header':
                break
        points=np.fromfile(stream,dtype)
    f,y,x=points['frame_id'],points['pixel_v'],points['pixel_u']
    change=selected[f,y,x]
    assert change.sum()==selected.sum()
    ex=np.load(PARENT)['extrinsics'][0].astype(float)
    centers=np.array([-np.linalg.solve(e[:,:3],e[:,3]) for e in ex])@ex[0,:,:3].T+ex[0,:,3]
    ratio=corrected[f[change],y[change],x[change]]/raw[f[change],y[change],x[change]]
    original=points.copy()
    for i,k in enumerate('xyz'):
        center=centers[f[change],i]
        points[k][change]=center+(points[k][change]-center)*ratio
    assert np.array_equal(original[~change],points[~change])
    for name in ['corrected_rgb','corrected_marked']:
        path=OUT/f'p13_{name}.ply'
        if name.endswith('marked'):
            points['red'][change],points['green'][change],points['blue'][change]=255,150,20
        with path.open('wb') as stream:
            stream.write(b''.join(header))
            points.tofile(stream)
        with path.open('rb') as stream:
            stream.seek(sum(map(len,header)))
            assert np.array_equal(np.fromfile(stream,dtype),points)

parser=argparse.ArgumentParser()
parser.add_argument('stage',choices=['fit','evaluate','export'])
parser.add_argument('--relaxed',action='store_true')
ARGS=parser.parse_args()
if ARGS.relaxed:
    OUT=BASE/'P13-indoor-20261118-002'
{'fit':fit,'evaluate':evaluate,'export':export}[ARGS.stage]()
