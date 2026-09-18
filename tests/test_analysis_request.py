from pathlib import Path
import json

import pytest

from scripts.run_analysis_request import completed_array_runs


def test_only_complete_scientific_array_members_are_accepted(monkeypatch):
    monkeypatch.setattr("scripts.run_analysis_request.subprocess.check_output",
        lambda *a, **k: "101|COMPLETED|0:0\n101.batch|COMPLETED|0:0\n102|COMPLETED|0:0\n")
    assert completed_array_runs(100, 2, "development", Path("/runs")) == ["/runs/development-101", "/runs/development-102"]
    with pytest.raises(ValueError, match="completely successful"):
        completed_array_runs(100, 3, "development", Path("/runs"))
    monkeypatch.setattr("scripts.run_analysis_request.subprocess.check_output",
        lambda *a, **k: "101|COMPLETED|0:0\n102|FAILED|1:0\n")
    with pytest.raises(ValueError, match="completely successful"):
        completed_array_runs(100, 2, "development", Path("/runs"))


def test_interaction_resolves_both_arrays_without_dropping_members(tmp_path, monkeypatch):
    from scripts import run_analysis_request as driver
    request = tmp_path/'request.json'
    request.write_text(json.dumps(dict(script='analyze_viva_interaction.py', arguments=[],
        run_inputs={flag: dict(arrays=[dict(job_id=job, expected_tasks=3, prefix='viva-development')])
                    for flag, job in [('--base-runs', 100), ('--control-runs', 200)]})))
    output = tmp_path/'analysis-300'
    output.mkdir()
    monkeypatch.setenv('SLURM_JOB_ID', '300')
    monkeypatch.setattr('sys.argv', ['driver', '--request', str(request), '--job-output', str(output)])
    monkeypatch.setattr(driver, 'runtime_manifest', lambda: {})
    monkeypatch.setattr(driver.subprocess, 'check_output', lambda command, **kwargs:
        '\n'.join(f'{int(command[2])+i}|COMPLETED|0:0' for i in range(3)))
    commands = []
    monkeypatch.setattr(driver.subprocess, 'run', lambda command, **kwargs: commands.append(command))
    driver.main()
    command = commands[0]
    for flag, job in [('--base-runs', 100), ('--control-runs', 200)]:
        start = command.index(flag)+1
        assert command[start:start+3] == [str(tmp_path/f'viva-development-{job+i}') for i in range(3)]
