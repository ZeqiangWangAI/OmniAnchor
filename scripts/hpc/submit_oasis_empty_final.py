from pathlib import Path
import os,json,subprocess
root=Path('/mnt/fast/nobackup/scratch4weeks/zw00924/OmniAnchor-20260910')
release=root/'releases/oasis-affect12-final-fc2e398-02'
python=root/'runs/smoke-44672/venv/bin/python'
contract=release/'research/contracts/oasis-affect12-final-20260911-01.json'
empty=root/'runs/oasis12-empty-predictions-20260911-01'
control=root/'oasis12-empty-final-control-20260911-01.json'
script=root/'prepare-oasis-empty-final-20260911-01.py'
assert not script.exists() and not empty.exists() and not control.exists()
script.write_text('''from pathlib import Path
import subprocess,json,hashlib
from datetime import datetime,timezone
root=Path("'''+str(root)+'''")
release=root/'releases/oasis-affect12-final-fc2e398-02'
contract=release/'research/contracts/oasis-affect12-final-20260911-01.json'
empty=root/'runs/oasis12-empty-predictions-20260911-01'
subprocess.run(["'''+str(python)+'''",str(release/'scripts/prepare_empty_predictions.py'),'--contract',str(contract),'--empty-native-run',str(root/'runs/development-47101'),'--empty-baseline-run',str(root/'runs/baseline-development-47102'),'--output',str(empty)],check=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
record=dict(status='frozen',created_utc=datetime.now(timezone.utc).isoformat(),scoring_contract=str(contract),scoring_contract_sha256=sha(contract),analyzer_sha256=sha(release/'scripts/evaluate_empty_controls.py'),final_evaluator_sha256=sha(release/'scripts/evaluate_text_final.py'),statistics_source_sha256={p:sha(release/p) for p in ['src/omnianchor/evaluation.py','src/omnianchor/analysis.py']},empty_files_sha256={p.name:sha(p) for p in empty.iterdir() if p.is_file()},inference='1000 source-group paired bootstrap; six-method MSE-reduction family BHq.05; constant correlations undefined; label shuffle diagnostic only',timing='Previously specified OASIS controls; no refit. Existing direct OASIS results known; affect12 final metrics not inspected when this orchestration was submitted.')
(root/'oasis12-empty-final-control-20260911-01.json').write_text(json.dumps(record,indent=2))
''')
env={k:v for k,v in os.environ.items() if not k.startswith('VL_')}
env.update(PYTHONPATH=str(release/'src'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',VL_SOURCE_DIR=str(release),VL_ENV_PYTHON=str(python))
cmd=['sbatch','--parsable','--partition=debug,2080ti','--cpus-per-task=4','--mem=16G','--time=00:30:00',f'--output={root}/logs/oasis12-empty-prepare-%j.out',f'--error={root}/logs/oasis12-empty-prepare-%j.err','--wrap',f'{python} {script}']
prepare=subprocess.check_output(cmd,env=env,text=True).strip();print('prepare',prepare,flush=True)
request=root/'oasis12-empty-final-request-20260911-01.json'
request.write_text(json.dumps(dict(script='evaluate_empty_controls.py',arguments=['--contract',str(control),'--data',str(release/'data/prepared/oasis-affect12-final-20260911-01'),'--empty',str(empty),'--final-results',str(root/'runs/analysis-47435/analysis')]),indent=2))
env['VL_ANALYSIS_REQUEST']=str(request)
analysis=subprocess.check_output(['sbatch','--parsable',f'--dependency=afterok:{prepare}:47435',f'--output={root}/logs/oasis12-empty-final-%j.out',f'--error={root}/logs/oasis12-empty-final-%j.err',str(release/'scripts/hpc/analysis.sbatch')],env=env,text=True).strip()
(root/'oasis12-empty-final-submissions-20260911-01.json').write_text(json.dumps(dict(prepare=prepare,analysis=analysis,request=str(request),script=str(script)),indent=2));print('analysis',analysis,flush=True)
