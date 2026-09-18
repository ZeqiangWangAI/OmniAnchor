"""Independently audit bridge input membership against prepared official train data."""
import argparse
import json, hashlib
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
base=root/'data/prepared/valueeval-20260910-01'
sources={}
def read(p):
    sources[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_text())
train={s['id']:s for s in read(base/'official/train/samples.json')}
roles={r:read(base/r/'samples.json') for r in ['reference','search']}
for role,samples in roles.items():
    assert len({s['id'] for s in samples})==len(samples)
    assert all(s==train[s['id']] and s['metadata']['split']=='train' for s in samples)
assert not ({s['group_id'] for s in roles['reference']}&{s['group_id'] for s in roles['search']})
checks=[]
for path in sorted(root.glob('runs/bridge-search-*/search/manifest.json')):
    manifest=read(path)
    for role in roles:
        assert manifest[role+'_ids']==[s['id'] for s in roles[role]]
        assert manifest['data_hashes'][role+'/samples.json']==sources[str((base/role/'samples.json').relative_to(root))]
    count=0
    for p in sorted(path.parent.glob('candidates/*/*.parquet.manifest.json')):
        role=p.name.split('.')[0]
        if role not in roles: continue
        d=read(p)
        samples=[{k:v for k,v in s.items() if k!='content_hash'} for s in d['samples']]
        assert samples==roles[role],str(p)
        assert d['execution']['failed_items']==0,str(p)
        count+=1
    checks.append({'run':path.parts[-3],'verified_raw_score_manifests':count})
out=args.output
with out.open('x') as f:
    json.dump({'status':'passed','reference_n':len(roles['reference']),'search_n':len(roles['search']),'reference_search_group_overlap':0,'checks':checks,'source_sha256':sources,'scope':'Exact reference/search inputs match official prepared train records and frozen search input hashes; all available raw-score manifests match those exact inputs with zero failed items. Does not independently reproduce optimizer objectives or prove original dataset contamination absent.'},f,indent=2)
print([(x['run'],x['verified_raw_score_manifests']) for x in checks])
