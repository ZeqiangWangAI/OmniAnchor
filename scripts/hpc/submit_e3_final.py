from pathlib import Path
import os
import json
import hashlib
import subprocess
import tarfile
import shutil
from datetime import datetime, timezone

root = Path("/mnt/fast/nobackup/scratch4weeks/zw00924/OmniAnchor-20260910")
release = root / "releases/final-drivers-fc2e398"
archive = root / "omnianchor-final-drivers-fc2e398.tgz"
assert (
    hashlib.sha256(archive.read_bytes()).hexdigest()
    == "08965d97cfe22bd92960566ddb77bd29d069f2fce6c670ebffcdccd745ffb249"
)
assert not release.exists()
shutil.copytree(root / "releases/e3-reference-80a2824", release)
with tarfile.open(archive) as t:
    t.extractall(release, filter="data")
python = root / "runs/smoke-44672/venv/bin/python"
env = os.environ.copy()
env.update(
    PYTHONPATH=str(release / "src") + ":" + str(release / "scripts"),
    VL_SOURCE_DIR=str(release),
    VL_ENV_PYTHON=str(python),
)
record = dict(
    release=str(release),
    source_archive_sha256="08965d97cfe22bd92960566ddb77bd29d069f2fce6c670ebffcdccd745ffb249",
    created_utc=datetime.now(timezone.utc).isoformat(),
    studies={},
)
for study, array in [("emobank", "46961"), ("dwug", "47117")]:
    bank = json.loads(
        (release / "data/prepared/e3-sensitivity-20260910-04/manifest.json").read_text()
    )
    names = list(bank["studies"][study]["variants"])
    output = subprocess.check_output(
        ["sacct", "-j", array, "-n", "-P", "--format=JobID,JobIDRaw,State,ExitCode"], text=True
    )
    jobs = {}
    for line in output.splitlines():
        parts = line.split("|")
        if len(parts) >= 4 and parts[1].isdigit():
            assert parts[2:4] == ["COMPLETED", "0:0"]
            jobs[int(parts[0].split("_")[1])] = parts[1]
    assert set(jobs) == set(range(len(names)))
    refs = {name: [str(root / "runs" / ("development-" + jobs[i]))] for i, name in enumerate(names)}
    refpath = release / f"research/{study}-final-reference-runs-20260911-01.json"
    refpath.write_text(json.dumps(refs, indent=2))
    folder = release / f"data/prepared/e3-{study}-final-20260911-01"
    cmd = [
        str(python),
        "scripts/prepare_sensitivity_final.py",
        "--bank",
        "data/prepared/e3-sensitivity-20260910-04",
        "--reference-runs",
        str(refpath),
        "--study",
        study,
        "--output",
        str(folder),
    ]
    subprocess.run(cmd, cwd=release, env=env, check=True)
    rows = json.loads((folder / "shards.json").read_text())["rows"]
    hours = max(row["wall_hours"] for row in rows)
    jobenv = env.copy()
    jobenv.update(
        VL_CONFIG_DIR=str(folder), VL_SHARD_DIR=str(folder), VL_FROZEN_EVALUATION_DIR=str(folder)
    )
    launch = [
        "sbatch",
        "--parsable",
        "--constraint=gpu_5000_ada",
        f"--time={hours:02d}:00:00",
        f"--array=0-{len(names) - 1}%1",
        "--output=" + str(root / f"logs/e3-{study}-final-%A-%a.out"),
        "--error=" + str(root / f"logs/e3-{study}-final-%A-%a.err"),
        "scripts/hpc/development.sbatch",
    ]
    job = subprocess.check_output(launch, cwd=release, env=jobenv, text=True).strip()
    record["studies"][study] = dict(
        job_id=job,
        command=launch,
        prepare_command=cmd,
        references=refs,
        rows=rows,
        analysis_contract=str(folder / "analysis-contract.json"),
    )
    with (root / "e3-full-final-submissions-20260911-01.json").open("w") as f:
        json.dump(record, f, indent=2)
    print(study, job, "wallhours", hours, flush=True)
