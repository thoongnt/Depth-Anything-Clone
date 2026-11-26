import sys, json
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import cv2,numpy as np,torch
import torch.nn.functional as F
from depth_anything.dpt import load_depth_anything
from evaluate_nyuv2 import make_transform
from evaluate_diode_relative import score_diode,aggregate
root=Path.cwd();out=root/'results/diode-relative/audit'
probe=json.loads((root/'results/diode-relative/research/sample-probe.json').read_text())
model=load_depth_anything('vitl',str(root/'checkpoints'),'cuda').eval()
transform=make_transform()
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
rows=[]
with torch.inference_mode():
 for previous in probe['rows']:
  name=previous['image'].replace(chr(92),'/');p=root/'data/diode/val'/name
  rgb=cv2.cvtColor(cv2.imread(str(p)),cv2.COLOR_BGR2RGB).astype(np.float32)/255
  x=torch.from_numpy(transform({'image':rgb})['image'])[None].cuda()
  prediction=(model(x)+model(x.flip(-1)).flip(-1))*.5
  prediction=F.interpolate(prediction[:,None],size=rgb.shape[:2],mode='bilinear',align_corners=False)[0,0].cpu().numpy()
  depth=np.load(p.with_name(p.stem+'_depth.npy')).squeeze()
  mask=np.load(p.with_name(p.stem+'_depth_mask.npy')).squeeze()
  rows.append({'image':name,'domain':previous['domain'],**score_diode(prediction,depth,mask)})
  print('Flip diagnostic',len(rows),'/16',flush=True)
report={'scope':'16 preselected probe images only, not a full TTA benchmark',
        'selection':probe['selection'],'baseline_no_flip':probe['by_domain'],
        'flip':{d:aggregate([r for r in rows if r['domain']==d]) for d in ['indoors','outdoor']},'rows':rows}
(out/'flip-probe.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
