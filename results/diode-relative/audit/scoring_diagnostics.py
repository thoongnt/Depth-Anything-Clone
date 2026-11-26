import json, hashlib
from pathlib import Path
import sys
sys.path.insert(0,str(Path.cwd()))
import numpy as np
from evaluate_nyuv2 import align_disparity
from evaluate_diode_relative import aggregate
root=Path.cwd();out=root/'results/diode-relative/audit'
s=json.loads((root/'results/diode-relative/initial-full-run-summary.json').read_text())
m=json.loads((root/'results/diode-relative/vitl-native-all-manifest.json').read_text())
cache=Path(s['prediction_cache']);rows={k:[] for k in ['native-disparity','gt-range-clamped-disparity','gt-range-clamped-depth-diagnostic']}
for i,sample in enumerate(m):
 rgb=sample['files'][0];key=hashlib.sha256(json.dumps(rgb,sort_keys=True).encode()).hexdigest()[:16]
 p=np.load(cache/(Path(sample['image']).stem+'-'+key+'.npy'))
 z=np.load(root/'data/diode/val'/sample['files'][1]['path']).squeeze()
 valid=np.load(root/'data/diode/val'/sample['files'][2]['path']).squeeze()>0
 valid &= np.isfinite(z)&(z>0)
 q=np.zeros(z.shape,dtype=np.float64);np.divide(1,z,out=q,where=valid)
 aligned,scale,shift=align_disparity(p,q,valid);a=aligned[valid];b=q[valid]
 clamped=np.maximum(a,1/float(z[valid].max()))
 for name,pred,gt in [('native-disparity',a,b),('gt-range-clamped-disparity',clamped,b),
                      ('gt-range-clamped-depth-diagnostic',1/clamped,z[valid])]:
  ratio=np.full(pred.shape,np.inf);positive=pred>0
  ratio[positive]=np.maximum(pred[positive]/gt[positive],gt[positive]/pred[positive])
  rows[name].append({'image':sample['image'],'domain':sample['domain'],
       'abs_rel':float(np.mean(np.abs(pred-gt)/gt)),'delta1':float(np.mean(ratio<1.25)),
       'valid_pixels':int(valid.sum()),'nonpositive_predictions':int((pred<=0).sum())})
 if (i+1)%100==0: print('Scoring diagnostics',i+1,'/771',flush=True)
result={'scope':'Full 771 diagnostic comparison. Clipping and depth-space scoring are not confirmed author Table2 protocol.',
        'gt_mask':'provided mask > 0 & finite GT & depth > 0; identical for every protocol',
        'variants':{name:{'all':aggregate(a),'by_domain':{d:aggregate([x for x in a if x['domain']==d])
                                     for d in ['indoors','outdoor']}} for name,a in rows.items()}}
(out/'scoring-diagnostics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
