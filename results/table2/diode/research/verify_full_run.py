import csv, json, hashlib
from pathlib import Path
import numpy as np

root=Path.cwd()
out=root/'results/table2/diode'
summary=json.loads((out/'vitl-native-all-summary.json').read_text())
manifest=json.loads((out/'vitl-native-all-manifest.json').read_text())
with (out/'vitl-native-all-per-image.csv').open(newline='',encoding='utf-8') as stream:
 rows=list(csv.DictReader(stream))
assert summary['full_validation_split'] and len(rows)==len(manifest)==771
assert len({r['image'] for r in rows})==771
assert {r['image'] for r in rows}=={m['image'] for m in manifest}
assert summary['by_domain']['indoors']['num_images']==325
assert summary['by_domain']['outdoor']['num_images']==446
for name in ['all','indoors','outdoor']:
 selected=rows if name=='all' else [r for r in rows if r['domain']==name]
 expected=summary if name=='all' else summary['by_domain'][name]
 for metric in ['abs_rel','delta1']:
  assert abs(np.mean([float(r[metric]) for r in selected])-expected['metrics'][metric])<1e-12
 assert sum(int(r['valid_pixels']) for r in selected)==expected['valid_pixels']
audit=json.loads((out/'research/data-audit.json').read_text())
for domain in audit:
 assert summary['by_domain'][domain]['valid_pixels']==audit[domain]['valid_pixels']
cache=Path(summary['prediction_cache'])
identity=json.loads((cache/'manifest.json').read_text())
assert hashlib.sha256((root/'evaluate_diode_relative.py').read_bytes()).hexdigest()==identity['evaluator_source']
probe=json.loads((out/'research/sample-probe.json').read_text())
index={m['image']:m for m in manifest}
diagnostics=[]
for previous in probe['rows']:
 name=previous['image'].replace(chr(92),'/')
 sample=index[name]
 rgb=sample['files'][0]
 image_key=hashlib.sha256(json.dumps(rgb,sort_keys=True).encode()).hexdigest()[:16]
 pred=np.load(cache/(Path(name).stem+'-'+image_key+'.npy'))
 old=np.load(root/'.cache/diode-research/sample-predictions'/(Path(name).stem+'.npy'))
 assert np.allclose(pred,old,rtol=1e-5,atol=1e-5)
 depth=np.load(root/'data/diode/val'/sample['files'][1]['path']).squeeze()
 valid=np.load(root/'data/diode/val'/sample['files'][2]['path']).squeeze()>0
 valid &= np.isfinite(depth)&(depth>0)
 x=pred[valid].astype(np.float64);q=1/depth[valid].astype(np.float64)
 design=np.column_stack((x,np.ones_like(x)))
 scale,shift=np.linalg.lstsq(design,q,rcond=None)[0]
 aligned=scale*x+shift
 score=float(np.mean(np.abs(aligned-q)/q))
 actual=next(r for r in rows if r['image']==name)
 assert abs(score-float(actual['abs_rel']))<1e-6
 diagnostics.append({'image':name,'max_prediction_difference_from_probe':float(np.max(np.abs(pred-old))),
                     'independent_lstsq_abs_rel':score,'reported_abs_rel':float(actual['abs_rel'])})
report={'status':'passed','full_manifest_and_csv_unique_images':771,
        'domain_counts':{'indoors':325,'outdoor':446},
        'all_gt_valid_pixel_counts_match_prior_dataset_audit':True,
        'csv_aggregates_match_summary':True,'source_hash_matches_cache':True,
        'probe_prediction_and_independent_lstsq_checks':diagnostics}
(out/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='probe_prediction_and_independent_lstsq_checks'},indent=2))
