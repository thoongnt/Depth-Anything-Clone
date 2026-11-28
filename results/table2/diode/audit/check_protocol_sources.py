import urllib.request,json
from pathlib import Path
out=Path('results/table2/diode/audit/upstream')
urls={'v1-tree':'https://api.github.com/repos/LiheYoung/Depth-Anything/git/trees/main?recursive=1',
      'v1-174-comments':'https://api.github.com/repos/LiheYoung/Depth-Anything/issues/174/comments',
      'v2-280-comments':'https://api.github.com/repos/DepthAnything/Depth-Anything-V2/issues/280/comments'}
for name,url in urls.items():
 req=urllib.request.Request(url,headers={'User-Agent':'Depth-Anything-reproduction-audit'})
 with urllib.request.urlopen(req,timeout=60) as response:data=json.load(response)
 (out/(name+'.json')).write_text(json.dumps(data,indent=2),encoding='utf-8')
 if name=='v1-tree':
  print('V1 commit',data['sha'])
  print('Evaluation-related paths',[x['path'] for x in data['tree'] if 'eval' in x['path'].lower() or 'diode' in x['path'].lower()])
 else:
  print(name,'comments',len(data))
  for comment in data[-2:]:print(comment['user']['login'],comment['updated_at'],comment['body'][:1100])
