from pathlib import Path
import os
import json
import subprocess
import tarfile
import shutil
import hashlib
from datetime import datetime, timezone

root = Path("/mnt/fast/nobackup/scratch4weeks/zw00924/OmniAnchor-20260910")
release = root / "releases/oasis-affect12-final-fc2e398"
assert not release.exists()
shutil.copytree(root / "releases/oasis-vatex-full-5b999b5", release)
with tarfile.open(root / "omnianchor-final-drivers-fc2e398.tgz") as t:
    t.extractall(release, filter="data")
python = root / "runs/smoke-44672/venv/bin/python"
env = os.environ.copy()
for key in list(env):
    if key.startswith("VL_"):
        env.pop(key)
env.update(
    PYTHONPATH=str(release / "src") + ":" + str(release / "scripts"),
    VL_SOURCE_DIR=str(release),
    VL_ENV_PYTHON=str(python),
)
folder = release / "data/prepared/oasis-affect12-final-20260911-01"
contract = release / "research/contracts/oasis-affect12-final-20260911-01.json"
prepare = [
    str(python),
    "scripts/prepare_text_final.py",
    "--test",
    "data/prepared/oasis-20260910-02/prepared/test",
    "--config",
    "configs/smoke/qwen35-en.json",
    "--verification",
    "data/prepared/oasis-affect-20260910-01/smoke32.json",
    "--features",
    str(root / "runs/analysis-46916/analysis"),
    "--probes",
    str(root / "runs/analysis-46917/analysis"),
    "--output",
    str(folder),
    "--contract",
    str(contract),
    "--oasis-affect12",
]
subprocess.run(prepare, cwd=release, env=env, check=True)
env.update(
    VL_CONFIG="configs/smoke/qwen35-en.json",
    VL_SAMPLES=str(folder / "samples.json"),
    VL_FROZEN_EVALUATION=str(contract),
    VL_VISION_REUSE_GATE=str(root / "runs/vision-reuse-gate-44786/gate/summary.json"),
)
record = dict(
    release=str(release),
    created_utc=datetime.now(timezone.utc).isoformat(),
    prepare_command=prepare,
    contract_sha256=hashlib.sha256(contract.read_bytes()).hexdigest(),
    jobs={},
)
for name, script, hours in [("native", "development", 2), ("baselines", "baseline_development", 1)]:
    localenv = env.copy()
    if name == "baselines":
        localenv["VL_METHODS"] = "qwen-embedding qwen-reranker"
    cmd = [
        "sbatch",
        "--parsable",
        "--constraint=gpu_5000_ada",
        f"--time={hours:02d}:00:00",
        "--output=" + str(root / f"logs/oasis12-final-{name}-%j.out"),
        "--error=" + str(root / f"logs/oasis12-final-{name}-%j.err"),
        f"scripts/hpc/{script}.sbatch",
    ]
    job = subprocess.check_output(cmd, cwd=release, env=localenv, text=True).strip()
    record["jobs"][name] = dict(job_id=job, command=cmd)
    with (root / "oasis12-final-submissions-20260911-01.json").open("w") as f:
        json.dump(record, f, indent=2)
    print(name, job, flush=True)
native = record["jobs"]["native"]["job_id"]
baseline = record["jobs"]["baselines"]["job_id"]
request = release / "research/oasis12-final-analysis-request-20260911-01.json"
request.write_text(
    json.dumps(
        dict(
            script="evaluate_text_final.py",
            arguments=[
                "--contract",
                str(contract),
                "--data",
                str(folder),
                "--native-run",
                str(root / f"runs/development-{native}"),
                "--baseline-run",
                str(root / f"runs/baseline-development-{baseline}"),
            ],
        ),
        indent=2,
    )
)
env["VL_ANALYSIS_REQUEST"] = str(request)
cmd = [
    "sbatch",
    "--parsable",
    "--partition=debug,2080ti",
    "--time=04:00:00",
    f"--dependency=afterok:{native}:{baseline}",
    "--output=" + str(root / "logs/oasis12-final-statistics-%j.out"),
    "--error=" + str(root / "logs/oasis12-final-statistics-%j.err"),
    "scripts/hpc/analysis.sbatch",
]
job = subprocess.check_output(cmd, cwd=release, env=env, text=True).strip()
record["jobs"]["analysis"] = dict(
    job_id=job, command=cmd, request_sha256=hashlib.sha256(request.read_bytes()).hexdigest()
)
with (root / "oasis12-final-submissions-20260911-01.json").open("w") as f:
    json.dump(record, f, indent=2)
print("analysis", job, flush=True)
