"""Agent RGB-only common floor footprints; no depth/point GT or raw depth loading."""
import json,time,shutil
from pathlib import Path
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'results/planar_residual_depth'
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def main():
 parent=BASE/'P10input-three-sequences-20261117-001';prefix='three-sequences-P10masks-'+time.strftime('%Y%m%d')+'-';i=1
 while (BASE/(prefix+f'{i:03}')).exists():i+=1
 out=BASE/(prefix+f'{i:03}');out.mkdir();(out/'source').mkdir();shutil.copy2(__file__,out/'source/freeze_masks.py');polys={'indoor':[[8,230],[115,215],[128,280],[8,280]],'outdoor':[[12,172],[35,158],[145,157],[170,187],[12,187]],'drone':[[205,195],[290,165],[345,175],[345,230],[205,230]]};cases=[]
 for c in json.loads((parent/'samples.json').read_text()):
  name=c['sample'];c['polygons']=[polys[name] for _ in c['images']];c['plane_id']=name+'_continuous_floor';c['mask_provenance']='AGENT_RGB_DESIGNATED_USER_UNREVIEWED';c['plane_notes']='same physical floor/road, papers/road markings treated as negligible surface relief; report sensor-reference nonplanarity, never remove a case based on GT';cases.append(c);dump(out/name/'case.json',c);sheet=Image.new('RGB',(5*192,4*164),'white');d=ImageDraw.Draw(sheet)
  for j,v in enumerate(c['images']):
   im=Image.open(ROOT/v['processed']).convert('RGB');dr=ImageDraw.Draw(im);dr.line([tuple(p) for p in polys[name]]+[tuple(polys[name][0])],fill='red',width=2);im.thumbnail((192,144));x=(j%5)*192;y=(j//5)*164;sheet.paste(im,(x,y+18));d.text((x+2,y+2),f'{j:02} {v["t_sec"]:.2f}s',fill='black')
  sheet.save(out/name/'RGB_mask_contact.png')
 dump(out/'samples.json',cases);dump(out/'config_resolved.json',json.loads((Path(__file__).parent/'protocol.json').read_text()));dump(out/'manifest.json',{'run_id':out.name,'input_parent':parent.name,'raw_parent':'P10raw-three-sequences-20261117-001','status':'MASKS_FROZEN_RGB_ONLY','raw_depth_values_loaded_during_mask_selection':False,'depth_point_GT_values_loaded':False,'provided_pose_control':True,'user_review_waived':True});print(out.name)
if __name__=='__main__':main()
