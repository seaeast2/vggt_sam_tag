"""Learn a keep/near/far action from paired boundary neighborhoods."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'P16'))
from data import load_manifest, calibrated

CLASSES=('keep','near','far')
RNG=np.random.default_rng(20)
cv2.setNumThreads(1)
FEATURES=['rgb_edge','depth_edge','confidence_edge','sam_distance','log_confidence',
          'log_depth_normalized','near_lab_distance','far_lab_distance','side_lab_distance',
          'near_relative_change','far_relative_change','relative_gap','near_depth_mad',
          'far_depth_mad','near_confidence','far_confidence','near_pca_thickness',
          'far_pca_thickness','abs_normal_dot']


def dump(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def gradients(a):
    return np.stack([cv2.Sobel(a,cv2.CV_32F,1,0,ksize=3,scale=.125),
                     cv2.Sobel(a,cv2.CV_32F,0,1,ksize=3,scale=.125)],axis=-1)


def sides(rgb,depth,confidence,sam,k,protect,normalizer):
    lab=cv2.cvtColor(rgb.astype('float32')/255,cv2.COLOR_RGB2LAB)
    lab_grad=gradients(cv2.GaussianBlur(lab,(3,3),.65))
    log_z=np.log(depth).astype('float32')
    log_c=np.log1p(confidence).astype('float32')
    zgrad,cgrad=[gradients(cv2.GaussianBlur(q,(3,3),.65)) for q in (log_z,log_c)]
    rgb_strength=np.sqrt((lab_grad**2).sum(axis=(-1,-2)))
    z_strength=np.linalg.norm(zgrad,axis=-1)
    c_strength=np.linalg.norm(cgrad,axis=-1)
    y,x=np.where((confidence<protect)&((rgb_strength>=4)|(z_strength>=.018)|(sam<=.25)))
    h,w=depth.shape
    channel=np.argmax((lab_grad[y,x]**2).sum(-1),axis=-1)
    color_normal=lab_grad[y,x,channel]/7
    depth_normal=zgrad[y,x]/.035
    vectors=np.where((np.linalg.norm(color_normal,axis=-1)>=np.linalg.norm(depth_normal,axis=-1))[:,None],
                     color_normal,depth_normal)
    magnitude=np.linalg.norm(vectors,axis=-1)
    keep=magnitude>1e-5
    x,y,vectors,magnitude=x[keep],y[keep],vectors[keep],magnitude[keep]
    ux,uy=(vectors/magnitude[:,None]).T
    offsets=np.tile(np.arange(4,9),3)
    tangent=np.repeat([-2,0,2],5)
    inv=np.linalg.inv(k)
    def sample(sign):
        u=np.rint(x[:,None]+sign*ux[:,None]*offsets-uy[:,None]*tangent).astype(int)
        v=np.rint(y[:,None]+sign*uy[:,None]*offsets+ux[:,None]*tangent).astype(int)
        inside=((u>=0)&(u<w)&(v>=0)&(v<h)).all(1)
        u,v=np.clip(u,0,w-1),np.clip(v,0,h-1)
        depths=depth[v,u]
        z=np.median(depths,axis=1)
        mad=np.median(abs(depths-z[:,None]),axis=1)/z
        rays=np.stack([u,v,np.ones_like(u)],-1)@inv.T
        xyz=rays*depths[...,None]
        centered=xyz-xyz.mean(1,keepdims=True)
        eig,vec=np.linalg.eigh(centered.transpose(0,2,1)@centered)
        thickness=np.clip(eig[:,0],0,None)/np.maximum(eig.sum(1),1e-12)
        return (z,mad,np.median(lab[v,u],axis=1),np.median(log_c[v,u],axis=1),
                thickness,vec[:,:,0],inside)
    a,b=sample(-1),sample(1)
    swap=a[0]>b[0]
    near,far=zip(*[(np.where(swap.reshape((-1,)+(1,)*(u.ndim-1)),v,u),
                    np.where(swap.reshape((-1,)+(1,)*(u.ndim-1)),u,v)) for u,v in zip(a,b)])
    center=depth[y,x]
    gap=(far[0]-near[0])/np.maximum(near[0],1e-8)
    valid=a[-1]&b[-1]&(near[1]<.08)&(far[1]<.08)&(gap>=.04)
    colors=lab[y,x]
    feature=np.column_stack([rgb_strength[y,x],z_strength[y,x],c_strength[y,x],sam[y,x],log_c[y,x],
        np.log(center/normalizer),np.linalg.norm(colors-near[2],axis=1),np.linalg.norm(colors-far[2],axis=1),
        np.linalg.norm(near[2]-far[2],axis=1),(near[0]-center)/center,(far[0]-center)/center,
        gap,near[1],far[1],near[3],far[3],near[4],far[4],abs((near[5]*far[5]).sum(1))])
    return y[valid],x[valid],feature[valid].astype('float32'),np.stack([center,near[0],far[0]],axis=1)[valid]


def candidates(clip):
    with np.load(clip['cache'],allow_pickle=False) as cache:
        scale,test=calibrated(cache['gt'],cache['raw'])
        protect=float(np.quantile(cache['confidence'],.70))
        normalizer=float(np.median(cache['raw']))
        result=[]
        for i,z in enumerate(cache['raw']):
            y,x,f,proposals=sides(cache['rgb'][i],z,cache['confidence'][i],cache['sam'][i],
                                  cache['model_K'][i],protect,normalizer)
            valid=test[i,y,x]
            gt=cache['gt'][i,y,x]
            error=abs(proposals*scale-gt[:,None])
            oracle=np.argmin(error,axis=1)
            improvement=error[:,0]-error[np.arange(len(oracle)),oracle]
            oracle[improvement<np.maximum(.002,.01*gt)]=0
            result.append(dict(y=y,x=x,features=f,depths=proposals,test=valid,
                               oracle=oracle,error=error,frame=i))
    return result,scale,protect


def counts(a):
    return {name:int((a==i).sum()) for i,name in enumerate(CLASSES)}


def fit(training):
    rows=[]
    for clip in training:
        frames,_,_=candidates(clip)
        for q in frames:
            ids=np.flatnonzero(q['test'])
            if len(ids)>700:
                ids=RNG.choice(ids,700,replace=False)
            rows.append((q['features'][ids],q['oracle'][ids]))
    x,y=[np.concatenate(v) for v in zip(*rows)]
    ceiling=min(2500,min((y==k).sum() for k in range(3)))
    if ceiling<20:
        raise ValueError(f'Insufficient examples for one action: {counts(y)}')
    idx=np.concatenate([RNG.choice(np.flatnonzero(y==k),ceiling,replace=False) for k in range(3)])
    mean=x.mean(0)
    std=x.std(0).clip(1e-6)
    svm=cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_RBF)
    svm.setC(2.)
    svm.setGamma(1/x.shape[1])
    assert svm.train(np.ascontiguousarray(((x[idx]-mean)/std).astype('float32')),
                     cv2.ml.ROW_SAMPLE,y[idx].astype('int32'))
    svm.save(str(OUT/'svm.xml'))
    return svm,mean,std,counts(y),int(ceiling)


def score_rows(rows):
    errors=np.concatenate([r['error'] for r in rows])
    actual=np.concatenate([r['oracle'] for r in rows])
    predicted=np.concatenate([r['predicted'] for r in rows])
    ids=np.arange(len(actual))
    arms={'raw':errors[:,0],'always_far':errors[:,2],'svm':errors[ids,predicted],'oracle':errors[ids,actual]}
    return dict(pixels=len(ids),mae_cm={k:float(100*v.mean()) if len(v) else None for k,v in arms.items()},
        target_actions=counts(actual),predicted_actions=counts(predicted),
        action_accuracy=float((predicted==actual).mean()) if len(ids) else None,
        confusion=np.bincount(3*actual+predicted,minlength=9).reshape(3,3).tolist(),
        improved_svm=int((arms['svm']<arms['raw']).sum()),worsened_svm=int((arms['svm']>arms['raw']).sum()))


def evaluate(clip,svm,mean,std):
    frames,scale,protect=candidates(clip)
    folder=OUT/Path(clip['cache']).parent.name
    folder.mkdir()
    with np.load(clip['cache']) as cache:
        raw=cache['raw']
        action=np.zeros_like(raw,dtype=np.uint8)
        candidate=np.zeros_like(raw,dtype=bool)
        corrected=raw.copy()
        rows=[]
        for q in frames:
            i,y,x=q['frame'],q['y'],q['x']
            if len(x):
                f=np.ascontiguousarray((q['features']-mean)/std,dtype=np.float32)
                predicted=svm.predict(f)[1].ravel().astype(int)
            else:
                predicted=np.empty(0,dtype=int)
            corrected[i,y,x]=q['depths'][np.arange(len(x)),predicted]
            action[i,y,x]=predicted
            candidate[i,y,x]=True
            mask=q['test']
            rows.append(dict(error=q['error'][mask],oracle=q['oracle'][mask],predicted=predicted[mask]))
        assert np.array_equal(corrected[action==0],raw[action==0])
        assert not np.any((action!=0)&(cache['confidence']>=protect))
        np.savez_compressed(folder/'depths.npz',raw=raw,corrected=corrected,residual=corrected-raw,
                            action=action,candidate=candidate)
        h,w=raw.shape[1:]
        sheet=Image.new('RGB',(w*2,(h+24)*((len(raw)+1)//2)),'white')
        draw=ImageDraw.Draw(sheet)
        for i,rgb in enumerate(cache['rgb']):
            image=rgb.copy()
            for a,color in [(1,[0,205,255]),(2,[255,145,20])]:
                image[action[i]==a]=color
            x,y=i%2*w,i//2*(h+24)
            sheet.paste(Image.fromarray(image),(x,y+24))
            draw.text((x+4,y+4),f'{i:02}  Near {int((action[i]==1).sum())} / Far {int((action[i]==2).sum())}',fill='black')
        sheet.save(folder/'actions.jpg',quality=94)
        if clip['domain']=='indoor':
            write_ply(folder,cache,corrected,action)
    result=score_rows(rows)
    result.update(scene=clip['scene'],split=clip['split'],domain=clip['domain'],
                  candidate_pixels=int(candidate.sum()),full_predicted_actions=counts(action[candidate]),
                  scale=scale,confidence_p70=protect)
    dump(folder/'metrics.json',result)
    return result,rows


def write_ply(folder,cache,corrected,action):
    colors=cache['rgb']
    h,w=corrected.shape[1:]
    y,x=np.indices((h,w))
    uv=np.stack([x,y,np.ones_like(x)],-1)
    dtype=np.dtype([(k,'<f4') for k in 'xyz']+[(k,'u1') for k in ('red','green','blue')]+
                   [('confidence','<f4')]+[(k,'<u2') for k in ('frame_id','pixel_u','pixel_v')])
    blocks=[]
    for i,z in enumerate(corrected):
        pose=cache['model_pose'][i]
        view=(uv@np.linalg.inv(cache['model_K'][i]).T)*z[...,None]
        world=(view-pose[:3,3])@np.linalg.inv(pose[:3,:3]).T
        rec=np.empty(h*w,dtype=dtype)
        for k,name in enumerate('xyz'):
            rec[name]=world[...,k].ravel()
        for k,name in enumerate(('red','green','blue')):
            rec[name]=colors[i,...,k].ravel()
        rec['confidence']=cache['confidence'][i].ravel()
        rec['frame_id'],rec['pixel_u'],rec['pixel_v']=i,x.ravel(),y.ravel()
        blocks.append(rec)
    records=np.concatenate(blocks)
    head=(f'ply\nformat binary_little_endian 1.0\nelement vertex {len(records)}\n'
          'property float x\nproperty float y\nproperty float z\n'
          'property uchar red\nproperty uchar green\nproperty uchar blue\n'
          'property float confidence\nproperty ushort frame_id\nproperty ushort pixel_u\nproperty ushort pixel_v\nend_header\n').encode()
    for name in ('corrected_rgb','corrected_actions'):
        if name.endswith('actions'):
            for a,color in [(1,[0,205,255]),(2,[255,145,20])]:
                for k,key in enumerate(('red','green','blue')):
                    records[key][action.ravel()==a]=color[k]
        path=folder/f'{name}.ply'
        with path.open('xb') as f:
            f.write(head)
            records.tofile(f)
        with path.open('rb') as f:
            assert f.read(len(head))==head and np.array_equal(np.fromfile(f,dtype),records)


def main():
    global MANIFEST, OUT
    MANIFEST, OUT = map(Path,sys.argv[1:3])
    OUT.mkdir(parents=True)
    clips=load_manifest(MANIFEST)
    dump(OUT/'protocol.json',dict(manifest=str(MANIFEST),manifest_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        features=FEATURES,classes=CLASSES,confidence='clip p70',side_radius_px=[4,8],min_relative_gap=.04,
        min_oracle_improvement='max(2mm, 1% of GT depth)',model='RBF SVM, C=2, gamma=1/19, balanced train',
        gate='RGB or depth or SAM edge; confidence below p70',scope='same P17 scene-separated split'))
    svm,mean,std,labels,per_class=fit([c for c in clips if c['split']=='train'])
    np.savez(OUT/'normalization.npz',mean=mean,std=std)
    by_group={}
    details=[]
    for clip in clips:
        if clip['split']=='train':
            continue
        metrics,rows=evaluate(clip,svm,mean,std)
        details.append(metrics)
        by_group.setdefault((clip['split'],clip['domain']),[]).extend(rows)
        print(clip['scene'],metrics['pixels'],metrics['mae_cm'],flush=True)
    summary={f'{s}_{d}':score_rows(rows) for (s,d),rows in by_group.items()}
    dump(OUT/'summary.json',dict(train_available_labels=labels,train_per_class=per_class,groups=summary,clips=details))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
