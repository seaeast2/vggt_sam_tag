"""Prepare predefined evaluation KITTI sample; RGB-only plane proposal."""
import json,time,hashlib,shutil,sys,argparse
from pathlib import Path
from PIL import Image,ImageDraw
import numpy as np
ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'results/planar_residual_depth'
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--sample',required=True);ap.add_argument('--candidate',required=True);ap.add_argument('--polygons',required=True,help='JSON pair of RGB-only polygons in processed pixel coordinates');args=ap.parse_args()
 prefix=f'kitti-{args.sample}-P3input-' +time.strftime('%Y%m%d')+'-';seq=1
 while (BASE/(prefix+f'{seq:03}')).exists():seq+=1
 rid=prefix+f'{seq:03}';out=BASE/rid;out.mkdir(parents=True,exist_ok=False)
 inputparent=BASE/'P1-kitti2015-first20-20261117-001';entry=next(s for s in json.loads((inputparent/'samples.json').read_text()) if s['sample']==args.sample)
 polys=json.loads(args.polygons);assert len(polys)==2 and all(len(p)>=3 for p in polys)
 assert entry['subset']=='fixed_evaluation'
 case={'sample':args.sample,'candidate':args.candidate,'input_parent':inputparent.name,'views':entry['images'],'polygons':polys,'raw_parent':'P1-kitti2015-first20-raw-20261117-001',
 'source_mask_status':'PENDING_HUMAN_REVIEW','target_mask_status':'PENDING_HUMAN_REVIEW','GT_read':False,'raw_depth_read':False}
 dump(out/'case.json',case)
 settings={'arms':['B0','B1','B2','B3'],'solver_parent':'P2-kitti-000000-20261117-001/source/solver.py',
 'solver_sha256':sha(BASE/'P2-kitti-000000-20261117-001/source/solver.py'),'LK_parent_config_sha256':sha(BASE/'P1c-approved-planes-20261117-001/config_resolved.json'),
 'same_solver_matching_support_scale_policy_as_P2':True,'sample_selection':f'predefined evaluation ID{args.sample} in ascending order; no GT/model quality used to choose scene',
 'no_tuning_on_this_sample':True,'limitations':'sample-ID split; not certified disjoint capture sequence or independent physical scene'}
 dump(out/'config_resolved.json',settings)
 (out/'source').mkdir();shutil.copy2(__file__,out/'source/prepare.py')
 sheet=Image.new('RGB',(1024,300),'white');draw=ImageDraw.Draw(sheet)
 for j,v in enumerate(entry['images']):
  p=ROOT/v['processed'];im=Image.open(p).convert('RGB');ImageDraw.Draw(im).line([tuple(x) for x in polys[j]]+[tuple(polys[j][0])],fill='red',width=2)
  sheet.paste(im,(j*512,30));draw.text((j*512+5,5),args.candidate+' '+('left/source' if j==0 else 'right/target')+' | road plane proposal',fill='black')
 sheet.save(out/'RGB_plane_review.png')
 dump(out/'manifest.json',{'run_id':rid,'stage':'P3input','status':'RGB_PROPOSAL_PREPARED_HUMAN_REVIEW_PENDING','GT_read':False,'depth_correction_executed':False})
 print(rid)
if __name__=='__main__':main()
