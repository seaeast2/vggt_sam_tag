"""P16 train, evaluate, and RGB-only inference entry points."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from data import Clips, load_manifest, stats
from model import Refiner, evidence


def dump(path, payload):
    path.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')


def sha(path):
    with open(path,'rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def settings(args):
    return dict(features=args.features, camera=args.camera, volume=args.volume,
                lambda_rgb=args.lambda_rgb, lambda_keep=args.lambda_keep)


def channels(config):
    return {'rgb-depth':4,'confidence':5,'sam':6}[config['features']]+9*config['volume']


def forward(model, batch, config, normalization, amp=False):
    raw=batch['raw']
    conf=batch['confidence']
    editable=torch.ones_like(raw,dtype=torch.bool)
    features=[batch['rgb'],raw/batch['normalizer'][:,None,None,None]]
    if config['features']!='rgb-depth':
        low,high=normalization['low'],normalization['high']
        features.append(((conf-low)/max(high-low,1e-6)).clamp(0,1))
        editable=conf<normalization['protect']
    if config['features']=='sam':
        features.append(batch['sam'])
    if config['volume']:
        features.append(evidence(batch,config['camera'],volume=True))
    with torch.autocast(device_type=raw.device.type,enabled=amp,dtype=torch.float16):
        return model(torch.cat(features,1),raw,editable)


def masked(value, mask):
    return torch.where(mask,value,0).sum()/mask.sum().clamp_min(1)


def summarize(model, loader, config, norm, device, mode, out):
    counts={}
    def record(group,name,err,mask):
        key=(group,name)
        total=counts.setdefault(key,[0.,0])
        total[0]+=torch.where(mask,err,0).sum().item()
        total[1]+=int(mask.sum())
    with torch.no_grad():
        for sample in loader:
            b={k:v.to(device) for k,v in sample.items()}
            pred,gate=forward(model,b,config,norm)
            if mode=='evaluate':
                valid=b['test']
                gt=b['gt'].nan_to_num()
                scale=b['scale'][:,None,None,None]
                high=b['confidence']>=norm['protect']
                physical=(b['gt']>0)&b['gt'].isfinite()
                edge=torch.zeros_like(valid)
                for dim in (-1,-2):
                    left,right=[b['gt'].narrow(dim,i,b['gt'].size(dim)-1) for i in (0,1)]
                    ok=physical.narrow(dim,0,left.size(dim))&physical.narrow(dim,1,left.size(dim))
                    step=ok&((left-right).abs()>.1*torch.minimum(left,right))
                    edge.narrow(dim,0,left.size(dim)).logical_or_(step)
                    edge.narrow(dim,1,left.size(dim)).logical_or_(step)
                edge=F.max_pool2d(edge.float(),5,1,2)>0
                for group, mask in {'all':valid,'low':valid&~high,'high':valid&high,
                                    'gt_edge':valid&edge,'low_gt_edge':valid&edge&~high,
                                    'wall_side':valid&(b['wall_floor']==1),
                                    'floor_side':valid&(b['wall_floor']==2)}.items():
                    for arm,d in [('raw',b['raw']),('corrected',pred),('rule',b['rule_depth'])]:
                        used=mask if arm!='rule' else mask&b['has_rule'][:,None,None,None]
                        error=(d*scale-gt).abs()
                        record(group,f'{arm}_MAE_m',error,used)
                        record(group,f'{arm}_AbsRel',error/gt.clamp_min(1e-6),used)
                record('all','outside_residual_cap',((gt-b['raw']*scale).abs()>.3*b['raw']*scale).float(),valid)
                record('all','changed_fraction',(pred!=b['raw']).float(),valid)
                for name,d in [('raw',b['raw']),('corrected',pred)]:
                    pair_batch = {**b, **{k[5:]: v for k,v in b.items() if k.startswith('eval_')}}
                    cost,support,inside=evidence(pair_batch,config['camera'],d)
                    record('reprojection',f'{name}_cost',cost,support)
                    record('reprojection',f'{name}_coverage',inside.float(),torch.ones_like(inside))
            if out is not None:
                save_predictions(out,loader.dataset,b,pred,gate)
    return {group:{name:dict(mean=total/n if n else None,pixels=n)
                   for (g,name),(total,n) in counts.items() if g==group}
            for group in sorted({g for g,_ in counts})}


def save_predictions(out,dataset,b,pred,gate):
    for k,(idx,i) in enumerate(zip(b['clip'].tolist(),b['frame'].tolist())):
        directory=out/f'clip_{idx:04}'
        directory.mkdir(exist_ok=True)
        clip=dataset.clips[idx]['clip']
        dump(directory/'source.json',clip)
        raw=b['raw'][k,0].cpu().numpy()
        depth=pred[k,0].cpu().numpy()
        K=b['model_K'][k].cpu().numpy()
        pose=b['model_pose'][k].cpu().numpy()
        np.savez_compressed(directory/f'{i:04}.npz',raw=raw,corrected=depth,residual=depth-raw,
                            gate=gate[k,0].cpu().numpy(),applied=depth!=raw,K=K,pose=pose)
        y,x=np.indices(depth.shape)
        xyz=(np.stack([x,y,np.ones_like(x)],-1)@np.linalg.inv(K).T)*depth[...,None]
        xyz=(xyz-pose[:3,3])@np.linalg.inv(pose[:3,:3]).T
        rgb=(b['rgb'][k].permute(1,2,0).cpu().numpy()*255).round().astype('uint8')
        points=np.empty(depth.size,dtype=[('xyz','<f4',(3,)),('rgb','u1',(3,))])
        points['xyz']=xyz.reshape(-1,3)
        points['rgb']=rgb.reshape(-1,3)
        header=(f'ply\nformat binary_little_endian 1.0\nelement vertex {len(points)}\n'
                'property float x\nproperty float y\nproperty float z\n'
                'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
        with (directory/f'{i:04}.ply').open('wb') as f:
            f.write(header.encode())
            points.tofile(f)


def train(args, clips, device):
    config=settings(args)
    norm=stats(clips)
    training=Clips(clips,'train',True,args.camera)
    validation=Clips(clips,'val',True,args.camera,'eval_neighbors')
    if not training or not validation:
        raise ValueError('Separate nonempty train and val scene splits required')
    if (args.lambda_rgb or args.volume) and not any((c['neighbors']>=0).any() for c in training.clips):
        raise ValueError('Multi-view training requires neighbor pairs')
    loader=DataLoader(training,batch_size=args.batch,shuffle=True,generator=torch.Generator().manual_seed(args.seed))
    val_loader=DataLoader(validation,batch_size=1)
    model=Refiner(channels(config)).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr)
    scaler=torch.amp.GradScaler('cuda',enabled=args.amp)
    history=[]
    best=float('inf')
    stale=0
    stopped=False
    for epoch in range(args.epochs):
        model.train()
        sums=np.zeros(4)
        for sample in loader:
            b={k:v.to(device) for k,v in sample.items()}
            opt.zero_grad(set_to_none=True)
            pred,_=forward(model,b,config,norm,args.amp)
            scale=b['scale'][:,None,None,None]
            target=b['gt'].nan_to_num()/scale
            sensor=masked(F.smooth_l1_loss(pred,target,reduction='none',beta=.02),b['test'])
            keep=((pred-b['raw'])/b['raw']).abs().mean()
            photo=pred.new_zeros(())
            coverage=pred.new_zeros(())
            if args.lambda_rgb:
                cost,support,inside=evidence(b,args.camera,pred)
                photo=masked(cost,support)
                coverage=support.float().mean()
            loss=sensor+args.lambda_rgb*photo+args.lambda_keep*keep
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sums += [sensor.item(),photo.item(),keep.item(),coverage.item()]
        if args.lambda_rgb and sums[3]==0:
            raise ValueError('No valid training reprojection pixels; inspect pose, parallax and visibility')
        model.eval()
        metrics=summarize(model,val_loader,config,norm,device,'evaluate',None)
        score=metrics[args.selection_group]['corrected_MAE_m']['mean']
        if score is None:
            raise ValueError(f'No valid validation pixels for {args.selection_group}')
        history.append(dict(epoch=epoch+1,train=(sums/len(loader)).tolist(),validation=metrics))
        dump(args.out/'history.json',history)
        if score<best-args.min_delta:
            best=score
            stale=0
            torch.save(dict(weights=model.state_dict(),config=config,norm=norm,epoch=epoch+1),args.out/'best.pt')
        else:
            stale+=1
        print(f'epoch {epoch+1} val {args.selection_group} MAE {score}',flush=True)
        if args.patience and stale>=args.patience:
            stopped=True
            break
    dump(args.out/'training_summary.json',dict(epochs_run=len(history),early_stopped=stopped,
         selection_group=args.selection_group,best_score_m=best))
    if not (args.out/'best.pt').is_file():
        raise ValueError('No valid validation pixels')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['train','evaluate','infer'])
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--split',choices=['train','val','test'],default='test')
    p.add_argument('--features',choices=['rgb-depth','confidence','sam'],default='sam')
    p.add_argument('--camera',choices=['model','provided'],default='model')
    p.add_argument('--volume',action='store_true')
    p.add_argument('--lambda-rgb',type=float,default=0)
    p.add_argument('--lambda-keep',type=float,default=.02)
    p.add_argument('--epochs',type=int,default=5)
    p.add_argument('--patience',type=int,default=0,help='Stop after this many epochs without improvement')
    p.add_argument('--min-delta',type=float,default=0,help='Minimum improvement in validation MAE, meters')
    p.add_argument('--selection-group',choices=['all','low','low_gt_edge'],default='all')
    p.add_argument('--batch',type=int,default=1)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--amp',action='store_true')
    args=p.parse_args()
    if args.amp and args.device!='cuda':
        p.error('--amp requires CUDA')
    if args.device=='cuda' and os.environ.get('P10_HOST_GPU')!='1':
        p.error('Run CUDA outside the sandbox with P10_HOST_GPU=1')
    if args.stage!='train' and args.checkpoint is None:
        p.error('--checkpoint required')
    if min(args.epochs,args.batch)<=0 or min(args.lambda_rgb,args.lambda_keep,args.patience,args.min_delta)<0:
        p.error('Invalid training settings')
    clips=load_manifest(args.manifest)
    torch.manual_seed(args.seed)
    device=torch.device(args.device)
    args.out.mkdir(parents=True)
    start=time.monotonic()
    if args.device=='cuda':
        torch.cuda.reset_peak_memory_stats()
    dump(args.out/'manifest.json',dict(status='RUNNING',arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
         inputs=[dict(c,sha256=sha(c['cache'])) for c in clips]))
    if args.stage=='train':
        train(args,clips,device)
    else:
        ckpt=torch.load(args.checkpoint,map_location=device,weights_only=True)
        config=ckpt['config']
        # A single-view refiner does not use cameras as network inputs.
        camera=config['camera'] if args.stage=='evaluate' or config['volume'] else 'model'
        model=Refiner(channels(config)).to(device)
        model.load_state_dict(ckpt['weights'])
        model.eval()
        ds=Clips(clips,args.split,args.stage=='evaluate',camera,
                 'eval_neighbors' if args.stage=='evaluate' else 'neighbors')
        if not ds:
            raise ValueError(f'Empty {args.split} split')
        result=summarize(model,DataLoader(ds,batch_size=1),config,ckpt['norm'],device,args.stage,args.out)
        if result:
            dump(args.out/'metrics.json',result)
    manifest=json.loads((args.out/'manifest.json').read_text())
    manifest.update(status='COMPLETED',seconds=time.monotonic()-start,
                    peak_cuda_bytes=torch.cuda.max_memory_allocated() if args.device=='cuda' else None)
    dump(args.out/'manifest.json',manifest)


if __name__=='__main__':
    main()
