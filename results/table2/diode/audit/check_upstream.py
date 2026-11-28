import urllib.request, ast, hashlib, json
from pathlib import Path
out=Path('results/table2/diode/audit/upstream');out.mkdir(exist_ok=True)
urls={name:'https://raw.githubusercontent.com/LiheYoung/Depth-Anything/main/'+path
      for name,path in [('dpt.py','depth_anything/dpt.py'),('transform.py','depth_anything/util/transform.py'),('run.py','run.py')]}
for name,url in urls.items():
 with urllib.request.urlopen(url,timeout=60) as response: data=response.read()
 (out/name).write_bytes(data)
report={'sources':urls,'class_comparisons':{}}
for name,path in [('dpt.py','depth_anything/dpt.py'),('transform.py','depth_anything/util/transform.py')]:
 upstream=ast.parse((out/name).read_text(encoding='utf-8'))
 local=ast.parse(Path(path).read_text(encoding='utf-8'))
 a={n.name:ast.dump(n,include_attributes=False) for n in upstream.body if isinstance(n,ast.ClassDef)}
 b={n.name:ast.dump(n,include_attributes=False) for n in local.body if isinstance(n,ast.ClassDef)}
 report['class_comparisons'][path]={key:a[key]==b.get(key) for key in a}
report['source_hashes']={name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in urls}
(out/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
