"""RGB-only provisional polygons for human review; never accepted automatically."""
import json
from pathlib import Path
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[3]
items=[
 ('K01','P1-kitti2015-first20-20261117-001','artifacts/inputs/000000/view0.png','road surface (rails need review)',[(96,250),(204,147),(244,147),(420,250)]),
 ('P01','P1-eth3d-pipes-20261117-003','artifacts/inputs/view0.png','orange flat panel',[(347,82),(503,77),(503,325),(348,325)]),
 ('P02','P1-eth3d-pipes-20261117-003','artifacts/inputs/view0.png','right gray cabinet door',[(72,104),(143,101),(143,284),(72,272)]),
 ('M01','P1-middlebury-all-20261117-001','artifacts/inputs/Adirondack/view0.png','red book top cover',[(83,207),(128,191),(172,211),(131,231)]),
]
out=ROOT/'results/planar_residual_depth/review/P1-candidates-v002'
out.mkdir(parents=True,exist_ok=False);sheet=Image.new('RGB',(1024,len(items)*390),'white');draw=ImageDraw.Draw(sheet);data=[]
for row,(pid,run,path,label,poly) in enumerate(items):
 src=ROOT/'results/planar_residual_depth'/run/path;im=Image.open(src).convert('RGB');overlay=im.copy();d=ImageDraw.Draw(overlay)
 d.line(poly+[poly[0]],fill='red',width=3);d.text((poly[0][0],max(0,poly[0][1]-15)),pid,fill='red')
 sheet.paste(im,(0,row*390+30));sheet.paste(overlay,(512,row*390+30));draw.text((5,row*390+5),f'{pid}: {label} | left=RGB, right=proposed polygon',fill='black')
 overlay.save(out/f'{pid}.png');data.append({'id':pid,'source_run':run,'image':path,'view':0,'label':label,'polygon':poly,'status':'PENDING_HUMAN_REVIEW','cross_view_masks':'PENDING','GT_or_raw_depth_used':False})
sheet.save(out/'candidate_review.png');(out/'candidates.json').write_text(json.dumps(data,indent=2)+'\n');print(out)
