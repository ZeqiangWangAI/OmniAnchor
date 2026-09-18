from pathlib import Path
import subprocess,json,tarfile,hashlib
root=Path('/mnt/fast/nobackup/scratch4weeks/zw00924/VLanchor-20260910')
requests={'viva':['46833','47086','45850','47259','47275','47122'],'dwug':['46170','46171'],'bridges':['46394'],'e3-emobank':['47418']}
archive=root/'final-raw-evidence-20260911-01.tgz'
assert not archive.exists()
record={'requests':requests,'runs':{},'files':{}}
for study,arrays in requests.items():
 out=subprocess.check_output(['sacct','-j',','.join(arrays),'-n','-P','--format=JobIDRaw,State,ExitCode'],text=True)
 jobs=[]
 for line in out.splitlines():
  fields=line.split('|')
  if len(fields)>=3 and fields[0].isdigit():
   assert fields[1:3]==['COMPLETED','0:0'],fields
   jobs.append(fields[0])
 assert jobs,study
 record['runs'][study]=[]
 for job in jobs:
  folders=[p for p in (root/'runs').glob('*-'+job) if p.is_dir()]
  assert len(folders)==1,(job,folders)
  folder=folders[0];assert (folder/'exit_code.txt').read_text().strip()=='0'
  record['runs'][study].append(str(folder.relative_to(root)))
  for p in folder.rglob('*'):
   relative=p.relative_to(folder)
   if any(v in {'source','venv','cache','media'} for v in relative.parts):continue
   if p.is_file(): record['files'][str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
with tarfile.open(archive,'w:gz') as t:
 for name in sorted(record['files']):t.add(root/name,arcname=name)
record['archive_sha256']=hashlib.sha256(archive.read_bytes()).hexdigest()
record['archive_bytes']=archive.stat().st_size
(root/'final-raw-evidence-20260911-01.json').write_text(json.dumps(record,indent=2))
print(len(record['files']),record['archive_bytes'],record['archive_sha256'],flush=True)
