"""Fit a common cylindrical or tapered bollard to prompted KITTI masks."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import least_squares

FIT = (0, 2, 3, 5)
TEST = (1, 4)
VERTEX = np.dtype([(k,'<f4') for k in 'xyz'] + [(k,'u1') for k in ('red','green','blue')]
                  + [('confidence','<f4')] + [(k,'<u2') for k in ('frame_id','pixel_u','pixel_v')])


def ray_terms(parameters, centers, rays):
    axis = np.r_[parameters[3], -1., parameters[4]]
    axis /= np.linalg.norm(axis)
    q = centers - parameters[:3]
    h0, dh = q @ axis, rays @ axis
    q = q - h0[..., None] * axis
    ray = rays - dh[..., None] * axis
    radius = parameters[5] - parameters[6] * h0
    dr = -parameters[6] * dh
    a = np.sum(ray * ray, axis=-1) - dr**2
    b = np.sum(q * ray, axis=-1) - radius * dr
    c = np.sum(q * q, axis=-1) - radius**2
    return a, b, c, radius, dr, h0, dh


def intersections(terms):
    a, b, c, radius, dr, _, _ = terms
    discriminant = b*b - a*c
    z = (-b - np.sqrt(np.maximum(0., discriminant))) / np.maximum(a, 1e-10)
    return z, (a > 0) & (discriminant >= 0) & (z > 0) & (radius + dr*z > 0)


def export(path, source, changed, depth, xyz, scale):
    with source.open('rb') as f:
        header = []
        for line in f:
            header.append(line)
            if line == b'end_header\n':
                break
        points = np.fromfile(f, VERTEX)
    frame, y, x = (points[k] for k in ('frame_id','pixel_v','pixel_u'))
    use = changed[frame,y,x]
    assert use.sum() == changed.sum()
    indices = frame[use],y[use],x[use]
    corrected = xyz(indices, depth[indices]) * scale
    for j, k in enumerate('xyz'):
        points[k][use] = corrected[:,j]
    for k, color in zip(('red','green','blue'), (255,145,20)):
        points[k][use] = color
    with path.open('xb') as f:
        f.writelines(header)
        points.tofile(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('source',type=Path,help='P25 run directory')
    p.add_argument('out',type=Path)
    args = p.parse_args()
    folder = args.source/'R640'
    with np.load(folder/'prediction.npz') as data:
        raw, conf, rgb, intrinsics, poses = (data[k] for k in
            ('raw','confidence','rgb','model_K','model_pose'))
    mask = np.load(args.out/'sam_mask.npy')
    height,width = raw.shape[1:]
    y,x = np.indices((height,width))
    rotations = poses[:,:3,:3].astype(float)
    inverse = np.linalg.inv(rotations)
    centers = -np.einsum('nij,nj->ni',inverse,poses[:,:3,3])
    directions = np.array([np.stack([x,y,np.ones_like(x)],-1) @ np.linalg.inv(k).T @ r.T
                           for k,r in zip(intrinsics,inverse)])
    k3 = np.ones((3,3),np.uint8)
    keep = np.stack([(cv2.dilate(z,k3)-cv2.erode(z,k3))/z <= .03 for z in raw])
    interior = np.stack([cv2.erode(m.astype('uint8'),np.ones((1,3),np.uint8)).astype(bool)
                         for m in mask]) & keep
    anchors = np.zeros_like(mask)
    for i in FIT:
        if interior[i].sum() < 5:
            raise ValueError(f'Insufficient interior depth anchors in frame {i}')
        anchors[i] = interior[i] & (conf[i] >= np.median(conf[i][interior[i]]))
    unit = float(np.median(raw[anchors]))
    centers /= unit
    normalized = raw/unit
    anchor_f = np.where(anchors)[0]
    anchor_points = centers[anchor_f]+directions[anchors]*normalized[anchors,None]
    origin = np.median(anchor_points,axis=0)
    contours = {}
    for i,m in enumerate(mask):
        rows = np.where(m.sum(axis=1)>=2)[0]
        rows = rows[(rows>=np.quantile(rows,.15)) & (rows<=np.quantile(rows,.85))]
        uv = []
        for row in rows:
            columns=np.where(m[row])[0]
            if columns[-1]-columns[0]+1 == len(columns):
                uv.extend([(columns[0]-.5,row,1),(columns[-1]+.5,row,1)])
        if len(uv)<20:
            raise ValueError(f'Insufficient silhouette in frame {i}')
        contours[i]=np.asarray(uv) @ np.linalg.inv(intrinsics[i]).T @ inverse[i].T

    def silhouette(params, ids):
        values=[]
        for i in ids:
            terms=ray_terms(params,centers[i],contours[i])
            a,b,c,r,dr,_,_=terms
            closest = -b/np.maximum(a,1e-8)
            radius = r+dr*closest
            miss = (c-b*b/np.maximum(a,1e-8))/np.maximum(2*radius,.002)
            values.append(miss*intrinsics[i,0,0]/np.maximum(closest,.1))
        return values

    def residual(params):
        outline=np.concatenate(silhouette(params,FIT))
        terms=ray_terms(params,centers[anchor_f],directions[anchors])
        z,hit=intersections(terms)
        # A weak surface-depth anchor fixes scale/placement without sensor depths.
        depth=np.where(hit,z,-terms[1]/terms[0])
        anchor_error=(depth-normalized[anchors])/.04
        return np.r_[outline, anchor_error*np.sqrt(len(outline)/len(depth))*.35,
                     params[3:5]*3]

    bounds=([*(origin-.45),-.25,-.25,.001,0.],
            [*(origin+.45),.25,.25,.06,.20])
    models={}
    predictions={}
    valid={}
    stats={}
    frame_index = np.where(mask)[0]
    for name in ('cylinder','tapered'):
        start=np.r_[origin,0.,0.,.007,.015]
        if name=='cylinder':
            fit=least_squares(lambda v: residual(np.r_[v,0.]),start[:-1],
                              bounds=(bounds[0][:-1],bounds[1][:-1]),loss='soft_l1',max_nfev=300)
            params=np.r_[fit.x,0.]
        else:
            fit=least_squares(residual,np.r_[models['cylinder'][:-1],.015],
                              bounds=bounds,loss='soft_l1',max_nfev=300)
            params=fit.x
        if not fit.success:
            raise RuntimeError(f'{name} fit failed to converge')
        models[name]=params
        z,hit=intersections(ray_terms(params,centers[frame_index],directions[mask]))
        prediction=raw.copy()
        prediction[mask]=z*unit
        valid[name]=np.zeros_like(mask)
        valid[name][mask]=hit & (abs(z*unit-raw[mask])<=.3*raw[mask])
        predictions[name]=prediction
        stats[name]=dict(parameters=params.tolist(),
                         fit_mean_abs_tangency_px=float(np.mean(np.abs(np.concatenate(silhouette(params,FIT))))),
                         heldout_mean_abs_tangency_px=float(np.mean(np.abs(np.concatenate(silhouette(params,TEST))))),
                         per_frame_mean_abs_tangency_px=[float(np.mean(np.abs(r))) for r in silhouette(params,range(len(raw)))])
    low=mask.copy()
    for i in range(len(raw)):
        low[i] &= conf[i]<np.median(conf[i][mask[i]])
    selected=mask&keep&low&valid['cylinder']&valid['tapered']
    for name in predictions:
        predictions[name][~selected]=raw[~selected]
    np.savez_compressed(args.out/'depths.npz',raw=raw,selected=selected,keep=keep,
                        **predictions,**{k+'_residual':v-raw for k,v in predictions.items()})

    manifest=json.loads((args.source/'manifest.json').read_text())
    scale=manifest['resolution']['640']['first_frame_scale_to_R384']
    def xyz(indices,depth):
        frame=indices[0]
        world=centers[frame]*unit+directions[indices]*depth[:,None]
        return world@rotations[0].T+poses[0,:3,3]
    source=folder/'edge_3x3_3pct.ply'
    for name,depth in [('raw',raw),*predictions.items()]:
        export(args.out/f'{name}.ply',source,depth!=raw,depth,xyz,scale)
    sheet=Image.new('RGB',(600,344))
    for i,(m,a) in enumerate(zip(mask,rgb)):
        ys,xs=np.where(m)
        panel=a.copy()
        panel[m]=np.rint(.6*panel[m]+.4*np.array([50,170,255])).astype('uint8')
        panel[selected[i]]=[255,145,20]
        crop=Image.fromarray(panel[ys.min()-5:ys.max()+6,xs.min()-8:xs.max()+9])
        crop.thumbnail((94,302))
        crop=crop.resize((min(94,crop.width*3),min(302,crop.height*3)))
        sheet.paste(crop,(i*100,28))
        ImageDraw.Draw(sheet).text((i*100+2,3),f'{78+2*i} '+('test' if i in TEST else 'fit'),fill='white')
    sheet.save(args.out/'mask_review.png')
    summary=dict(fit_frames=list(FIT),heldout_frames=list(TEST),source_frames=manifest['frames'],
                 parameters=['axis_point_x','axis_point_y','axis_point_z','axis_tilt_x','axis_tilt_z',
                             'radius_at_axis_point','radius_decrease_per_height'],
                 model_unit_divisor=unit,fit=stats,sam_pixels=mask.sum((1,2)).tolist(),
                 anchor_pixels=anchors.sum((1,2)).tolist(),selected_pixels=selected.sum((1,2)).tolist(),
                 assumed_approximately_vertical_in_first_camera=True,
                 filter='P25 edge_3x3_3pct; lower half of mask confidence; common positive hit and 30% cap')
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':
    main()
