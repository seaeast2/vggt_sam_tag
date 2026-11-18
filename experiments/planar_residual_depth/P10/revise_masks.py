"""One post-hoc RGB occluder correction, preserving original masks and all first results."""
import json,time,shutil
from pathlib import Path
from PIL import Image,ImageDraw
import numpy as np
ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'results/planar_residual_depth'
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def main():
 parent=BASE/'P10masks-three-sequences-20261117-001';out=BASE/('three-sequences-P10masks-'+time.strftime('%Y%m%d')+'-002');out.mkdir(exist_ok=False);(out/'source').mkdir();shutil.copy2(__file__,out/'source/revise_masks.py');cases=json.loads((parent/'samples.json').read_text());first=np.array(cases[2]['polygons'][0],float);last=np.array([[145,190],[225,150],[245,160],[195,230],[145,230]],float)
 for c in cases:
  if c['sample']=='drone':
   c['polygons']=[((1-j/19)*first+j/19*last).tolist() for j in range(20)];c['mask_revision']='post-hoc RGB occluder exclusion; original frame0 unchanged; no GT error threshold/plane fitting used to define revised coordinates; GT had been inspected in first-run diagnosis'
  dump(out/c['sample']/'case.json',c);sheet=Image.new('RGB',(960,656),'white');d=ImageDraw.Draw(sheet)
  for j,v in enumerate(c['images']):
   im=Image.open(ROOT/v['processed']).convert('RGB');p=c['polygons'][j];ImageDraw.Draw(im).line([tuple(x) for x in p]+[tuple(p[0])],fill='red',width=2);im.thumbnail((192,144));x=j%5*192;y=j//5*164;sheet.paste(im,(x,y+18));d.text((x+2,y+2),str(j),fill='black')
  sheet.save(out/c['sample']/'RGB_mask_contact.png')
 cfg=json.loads((ROOT/'experiments/planar_residual_depth/P10/protocol.json').read_text());cfg['posthoc_mask_revision']={'parent':parent.name,'reason':'fixed drone screen ROI included a visible non-floor stair/board object as viewpoint moved','changes':'only drone frames1..19 finite RGB polygons; source0/method thresholds/raw/calibration/scale policy unchanged','limitation':'GT was seen during prior diagnosis; not blinded or independent validation; no further revisions after evaluating this correction'};dump(out/'config_resolved.json',cfg);dump(out/'samples.json',cases);dump(out/'manifest.json',{'run_id':out.name,'status':'POSTHOC_RGB_MASK_REVISION_FROZEN','parent':parent.name,'original_results_preserved':True,'GT_previously_seen':True,'GT_used_for_numeric_mask_optimization':False,'revision_count':1});print(out.name)
if __name__=='__main__':main()
