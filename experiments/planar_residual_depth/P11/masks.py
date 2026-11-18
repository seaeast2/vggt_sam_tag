import json,shutil
from pathlib import Path
import numpy as np,cv2
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parents[3];B=R/'results/planar_residual_depth';parent=B/'P11input-kitti-1s-20261118-001';out=B/'P11masks-kitti-1s-20261118-002';out.mkdir(exist_ok=False);(out/'source').mkdir();shutil.copy2(__file__,out/'source/masks.py');c=json.loads((parent/'outdoor/case.json').read_text())
# RGB-only per-view safe asphalt polygons, distinct from correspondence footprints.
polys=[[[12,157],[180,151],[214,186],[12,186]],[[12,153],[170,150],[205,186],[12,186]],[[12,149],[164,147],[191,186],[12,186]],[[12,153],[157,148],[184,186],[12,186]],[[12,151],[132,148],[170,186],[12,186]],[[12,163],[116,154],[145,186],[12,186]],[[12,160],[171,151],[201,186],[12,186]],[[12,157],[237,150],[262,186],[12,186]],[[12,160],[242,149],[264,186],[12,186]],[[12,160],[236,144],[257,186],[12,186]],[[12,167],[178,151],[230,186],[12,186]],[[88,163],[217,158],[242,186],[65,186]],[[96,161],[218,158],[240,186],[74,186]],[[93,161],[174,157],[194,186],[70,186]],[[89,164],[166,156],[185,186],[62,186]],[[68,165],[154,153],[179,186],[45,186]]]
polys[8]=[[70,172],[242,149],[264,186],[45,186]]
polys[9]=[[100,172],[236,151],[257,186],[70,186]]
polys[10]=[[150,176],[230,171],[245,186],[140,186]]
polys[11]=[[135,169],[220,164],[242,186],[110,186]]
c.update(polygons=polys,plane_id='road_candidate_per_view',mask_provenance='agent RGB-only per-frame safe asphalt polygons frozen before new raw depth inspection or depth GT; not one fixed box and not identical physical patch. Same-plane road hypothesis only; correspondence identity verified by matches.');(out/'outdoor').mkdir();(out/'outdoor/case.json').write_text(json.dumps(c,indent=2));(out/'samples.json').write_text(json.dumps([c],indent=2));cfg=json.loads((parent/'config_resolved.json').read_text());cfg['mask_policy']=c['mask_provenance'];(out/'config_resolved.json').write_text(json.dumps(cfg,indent=2));sheet=Image.new('RGB',(4*500,4*275),'white')
for i,v in enumerate(c['images']):
 im=Image.open(R/v['processed']).convert('RGB');ImageDraw.Draw(im).line([tuple(p) for p in polys[i]]+[tuple(polys[i][0])],fill='red',width=2);im=im.resize((500,250));sheet.paste(im,((i%4)*500,(i//4)*275+25));ImageDraw.Draw(sheet).text(((i%4)*500+5,(i//4)*275+5),f'{i:02} source{v["source_frame"]:03} {v["t_sec"]:.3f}s',fill='black')
sheet.save(out/'outdoor/mask_contact.png');(out/'manifest.json').write_text(json.dumps({'parent':parent.name,'status':'RGB_MASKS_FROZEN','depth_GT_read':False,'new_raw_depth_read':False},indent=2));print(out)
