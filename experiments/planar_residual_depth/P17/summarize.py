"""Audit split counts and report paired test metrics for the two training arms."""
import argparse
import json
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('run',type=Path)
args=p.parse_args()
def load(path):
    return json.loads(path.read_text())
clips=load(args.run/'dataset.json')['clips']
report={'splits':{},'training':{},'evaluation':{}}
for split in ('train','val','test'):
    rows=[c for c in clips if c['split']==split]
    report['splits'][split]=dict(scenes=sorted({c['scene'] for c in rows}),
                               clips=len(rows),frames=sum(len(c['source_frame_ids']) for c in rows))
for a,b in [('train','val'),('train','test'),('val','test')]:
    assert not set(report['splits'][a]['scenes'])&set(report['splits'][b]['scenes'])
for arm in ('sensor','multiview'):
    folder=args.run/arm
    manifest=load(folder/'manifest.json')
    assert manifest['status']=='COMPLETED'
    summary=load(folder/'training_summary.json')
    history=load(folder/'history.json')
    best=min(history,key=lambda row:row['validation'][summary['selection_group']]['corrected_MAE_m']['mean'])
    report['training'][arm]=dict(summary,best_epoch=best['epoch'],seconds=manifest['seconds'],peak_cuda_bytes=manifest['peak_cuda_bytes'])
    for test in ('stereo','tum'):
        folder=args.run/f'eval-{arm}-{test}'
        assert load(folder/'manifest.json')['status']=='COMPLETED'
        metrics=load(folder/'metrics.json')
        assert metrics['high']['raw_MAE_m']==metrics['high']['corrected_MAE_m']
        report['evaluation'].setdefault(test,{})[arm]={group:{
            key:metrics[group][key] for key in ['raw_MAE_m','corrected_MAE_m']}
            for group in ['all','low','low_gt_edge','high']}
for arms in report['evaluation'].values():
    for key in arms['sensor']:
        assert arms['sensor'][key]['raw_MAE_m']==arms['multiview'][key]['raw_MAE_m']
(args.run/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
for test,arms in report['evaluation'].items():
    for group in ['all','low','low_gt_edge']:
        a,b=[arms[arm][group] for arm in ('sensor','multiview')]
        print(test,group,a['raw_MAE_m']['pixels'],
              [None if v['mean'] is None else round(v['mean']*100,3)
               for v in (a['raw_MAE_m'],a['corrected_MAE_m'],b['corrected_MAE_m'])])
print(report['training'])
