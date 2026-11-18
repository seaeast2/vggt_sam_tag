import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

root = Path(__file__).resolve().parents[3]
base = root/'results/planar_residual_depth'
out = base/'P12-indoor-20261118-001'
case = json.loads((base/'P10-indoor-2s-input-20261118-001/indoor/case.json').read_text())
early, full = [json.loads((out/f'{name}.json').read_text()) for name in ('early','full')]
later = {(r['frame'],r['mask']):r for r in full['regions']}
matched = [(r,later.get((r['frame'],r['mask']))) for r in early['regions'] if r['mode']=='provisional']
transition = [dict(early=a,late=b) for a,b in matched]
summary = dict(provisional_initial=len(matched),
    same_mask_full=sum(b is not None for _,b in matched),
    full_context_high_support=sum(b is not None and b['mode']=='high_support' for _,b in matched),
    scope='independent 8/16-frame VGGT runs, matched same early frame and same SAM mask; contextual diagnostic only; no direct new-frame wall track or temporal replacement')
(out/'context_diagnostic.json').write_text(json.dumps(dict(summary=summary,regions=transition),indent=2))
z = np.load(out/'full.npz')
h,w = z['raw'].shape[1:]
sheet = Image.new('RGB',(4*w,4*(h+23)),'white')
draw = ImageDraw.Draw(sheet)
for i,v in enumerate(case['images']):
    rgb = np.array(Image.open(root/v['processed']).convert('RGB'))
    m = z['selected'][i]
    provisional = np.zeros_like(m)
    for row in full['regions']:
        if row['frame']==i and row['mode']=='provisional':
            provisional |= z['mask_ids'][i]==row['mask']
    for flag,color in [(m&provisional,[255,150,20]),(m&~provisional,[30,220,220])]:
        rgb[flag] = np.rint(.45*rgb[flag]+.55*np.array(color)).astype('uint8')
    x=i%4*w
    y=i//4*(h+23)
    sheet.paste(Image.fromarray(rgb),(x,y+23))
    draw.text((x+4,y+4),f'{i:02} | {v["t_sec"]:.2f}s | {m.sum()} selected pixels',fill='black')
sheet.save(out/'selected_regions.jpg',quality=94)
r=full['regions']
print(summary)
print('region median plane distance medians',np.median([a['plane_distance_median_raw'] for a in r]),np.median([a['plane_distance_median_corrected'] for a in r]))
print('selected total',int(z['selected'].sum()),int(z['selected'].size))

parent = base/'P10-indoor-2s-ply-20261118-001'
source = parent/'vggt_omega_raw_RGB.ply'
dtype = np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1'),
                  ('confidence','<f4'),('frame_id','<u2'),('pixel_u','<u2'),('pixel_v','<u2')])
with source.open('rb') as stream:
    lines = []
    for line in stream:
        lines.append(line)
        if line.strip()==b'end_header':
            break
    raw = np.fromfile(stream,dtype)
centers = np.loadtxt(parent/'camera_centers.csv',delimiter=',',skiprows=1)[:,1:] if (parent/'camera_centers.csv').exists() else None
if centers is None:
    pred = np.load(base/'P10raw-three-sequences-20261118-003/indoor/predictions.npz')
    e = pred['extrinsics'][0].astype(float)
    centers = np.array([-np.linalg.solve(v[:,:3],v[:,3]) for v in e])@e[0,:,:3].T+e[0,:,3]
frame,y,x = raw['frame_id'],raw['pixel_v'],raw['pixel_u']
selected = z['selected'][frame,y,x]
changed = raw.copy()
ratio = (z['corrected'][frame[selected],y[selected],x[selected]]/
         z['raw'][frame[selected],y[selected],x[selected]])
for k,key in enumerate('xyz'):
    origin = centers[frame[selected],k]
    changed[key][selected] = origin+(raw[key][selected]-origin)*ratio
assert np.array_equal(changed[~selected],raw[~selected])
marked = changed.copy()
marked['red'][selected],marked['green'][selected],marked['blue'][selected] = 255,155,35
for name,points in [('p12_corrected_rgb',changed),('p12_selected_orange',marked)]:
    path=out/f'{name}.ply'
    header=b''.join(line for line in lines if not line.startswith(b'comment'))
    with path.open('wb') as stream:
        stream.write(header)
        stream.write(points.tobytes())
    with path.open('rb') as stream:
        stream.seek(len(header))
        assert np.array_equal(np.fromfile(stream,dtype),points)
