from pathlib import Path
import json,os,subprocess,shutil,tarfile
root=Path('/mnt/fast/nobackup/scratch4weeks/zw00924/VLanchor-20260910')
release=root/'releases/e3-final-cpu-d708ada'
assert not release.exists()
shutil.copytree(root/'releases/final-drivers-fc2e398',release)
with tarfile.open(root/'vlanchor-d708ada.tar') as t: t.extractall(release,filter='data')
env={k:v for k,v in os.environ.items() if not k.startswith('VL_')}
env.update(VL_SOURCE_DIR=str(release),VL_ENV_PYTHON=str(root/'runs/smoke-44672/venv/bin/python'))
record={}
for study,job in [('emobank','47418'),('dwug','47420')]:
 request=release/f'research/e3-{study}-final-cpu-request.json'
 request.write_text(json.dumps({'script':'run_sensitivity_analysis.py','arguments':['--bank',str(root/'releases/final-drivers-fc2e398/data/prepared/e3-sensitivity-20260910-04'),'--prepared',str(root/f'releases/final-drivers-fc2e398/data/prepared/e3-{study}-final-20260911-01'),'--runs-root',str(root/'runs'),'--job-id',job,'--study',study]},indent=2))
 env['VL_ANALYSIS_REQUEST']=str(request)
 cmd=['sbatch','--parsable','--partition=debug,2080ti','--time=04:00:00',f'--dependency=afterok:{job}',f'--output={root}/logs/e3-{study}-final-statistics-%j.out',f'--error={root}/logs/e3-{study}-final-statistics-%j.err','scripts/hpc/analysis.sbatch']
 result=subprocess.check_output(cmd,cwd=release,env=env,text=True).strip()
 record[study]={'job_id':result,'command':cmd,'request':str(request)}
 (root/'e3-final-cpu-submissions-20260911-01.json').write_text(json.dumps(record,indent=2))
 print(study,result,flush=True)
