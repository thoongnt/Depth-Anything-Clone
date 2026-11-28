import json,csv,hashlib
from pathlib import Path
import numpy as np
root=Path.cwd();out=root/'results/table2/diode/audit-rerun'
old=json.loads((root/'results/table2/diode/initial-full-run-summary.json').read_text())
new=json.loads((out/'fresh-inference-summary.json').read_text())
assert new['cache_hits']==0 and new['refresh_cache']
assert new['num_images']==771 and new['full_validation_split']
assert new['data_integrity_verification']['verified']
assert new['data_integrity_verification']['files_checked']==2313
original=Path(old['prediction_cache']);fresh=Path(new['prediction_cache'])
files=list(fresh.glob('*.npy'));assert len(files)==771
identical=0;maxdiff=0.
for p in files:
 q=original/p.name
 assert q.is_file()
 if hashlib.sha256(p.read_bytes()).digest()==hashlib.sha256(q.read_bytes()).digest():
  identical+=1
 else:
  a,b=np.load(p),np.load(q);delta=float(np.max(np.abs(a-b)))
  maxdiff=max(maxdiff,delta)
  assert np.allclose(a,b,rtol=1e-5,atol=1e-5),p.name
for domain in ['all','indoors','outdoor']:
 a,b=(old,new) if domain=='all' else (old['by_domain'][domain],new['by_domain'][domain])
 assert a['num_images']==b['num_images']
 assert a['valid_pixels']==b['valid_pixels']
 for metric in ['abs_rel','delta1']:assert abs(a['metrics'][metric]-b['metrics'][metric])<1e-7
with (out/'vitl-native-all-per-image.csv').open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
assert len(rows)==len({r['image'] for r in rows})==771
for metric in ['abs_rel','delta1']:assert abs(np.mean([float(r[metric]) for r in rows])-new['metrics'][metric])<1e-12
report={'status':'passed','fresh_images':771,'cache_hits':new['cache_hits'],
        'data_files_authenticated':2313,'bitwise_identical_predictions':identical,
        'maximum_prediction_difference_when_not_bitwise_identical':maxdiff,
        'old_metrics':old['metrics'],'fresh_metrics':new['metrics'],
        'metrics_equal':old['metrics']==new['metrics'],
        'source_sha256':hashlib.sha256((root/'evaluate_diode_relative.py').read_bytes()).hexdigest()}
(out/'fresh-vs-initial-verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
