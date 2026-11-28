from pathlib import Path
import sys,json
ROOT=Path.cwd();sys.path.insert(0,str(ROOT))
import numpy as np,torch,cv2
import torch.nn.functional as F
from evaluate_nyuv2 import make_transform,align_disparity,sha256
from depth_anything.dpt import load_depth_anything
out=ROOT/'.cache/diode-research';cache=out/'sample-predictions';cache.mkdir(exist_ok=True)
model=load_depth_anything('vitl',str(ROOT/'checkpoints'),'cuda').eval()
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
transform=make_transform();rows=[]
with torch.inference_mode():
 for domain in ('indoors','outdoor'):
  allfiles=sorted((ROOT/'data/diode/val'/domain).rglob('*.png'))
  selected=[allfiles[i] for i in np.linspace(0,len(allfiles)-1,8,dtype=int)]
  for file in selected:
   rgb=cv2.cvtColor(cv2.imread(str(file)),cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
   tensor=torch.from_numpy(transform({'image':rgb})['image'])[None].cuda()
   raw=model(tensor)
   pred=F.interpolate(raw[:,None],size=rgb.shape[:2],mode='bilinear',align_corners=False)[0,0].cpu().numpy()
   d=np.load(file.with_name(file.stem+'_depth.npy')).squeeze()
   m=np.load(file.with_name(file.stem+'_depth_mask.npy')).squeeze()
   mask=(m>0)&np.isfinite(d)&(d>0)
   gt=np.zeros_like(d,dtype=np.float64);np.divide(1,d,out=gt,where=mask)
   aligned,scale,shift=align_disparity(pred,gt,mask)
   a,b=aligned[mask],gt[mask]
   ratio=np.full(a.shape,np.inf);positive=a>0
   ratio[positive]=np.maximum(a[positive]/b[positive],b[positive]/a[positive])
   row={'image':str(file.relative_to(ROOT/'data/diode/val')),'domain':domain,'valid_pixels':int(mask.sum()),'nonpositive_aligned_prediction_pixels':int(np.count_nonzero(~positive)),'abs_rel_disparity':float(np.mean(np.abs(a-b)/b)),'delta1':float(np.mean(ratio<1.25)),'scale':scale,'shift':shift,'valid_depth_range_m':[float(d[mask].min()),float(d[mask].max())]}
   rows.append(row);np.save(cache/(file.stem+'.npy'),pred)
   print('Probe',len(rows),'/16',domain,'AbsRel',round(row['abs_rel_disparity'],4),'delta1',round(row['delta1'],4),flush=True)
result={'scope':'Research probe only, not a full benchmark or claimed paper reproduction','num_images':len(rows),'selection':'8 evenly spaced images per domain in sorted official val file layout; selected before scoring','encoder':'vitl','checkpoint_sha256':sha256(ROOT/'checkpoints/depth_anything_vitl14.pth'),'protocol':{'input':'Official 518 lower-bound, aspect-preserving, multiple 14, OpenCV cubic, ImageNet norm','output':'bilinear align_corners=False at native 768x1024','flip':False,'valid_mask':'provided mask > 0 AND finite depth > 0','alignment':'float64 centered affine least squares to reciprocal ground truth on valid mask','score':'AbsRel in disparity without clipping; delta1 counts nonpositive aligned predictions as failures','depth_crop_or_cap':'none; exact paper cap/filter unspecified'},'by_domain':{domain:{'num_images':sum(r['domain']==domain for r in rows),**{k:float(np.mean([r[k] for r in rows if r['domain']==domain])) for k in ('abs_rel_disparity','delta1')}} for domain in ('indoors','outdoor')},'rows':rows}
(out/'sample-probe.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
