"""Read-only Slurm accounting and bounded run telemetry collection; no score/label reads."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

RUN_PREFIXES = ("development", "baseline-development", "viva-development", "analysis", "fmat-validity",
    "fla-gate", "baselines", "bridge-search", "fmat32", "empty-controls", "vision-reuse-gate",
    "smoke", "reranker-media-gate", "forward-profile", "media-smoke", "vatex-reranking",
    "clean-environment", "clean-smoke", "clean-e5")


def allocated_gpus(tres):
    values = dict(item.split("=", 1) for item in tres.split(",") if "=" in item)
    if "gres/gpu" in values:
        return int(values["gres/gpu"])
    return sum(int(v) for k, v in values.items() if k.startswith("gres/gpu:"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    runs, ignored = {}, []
    for folder in (args.root/"runs").iterdir():
        match = re.fullmatch("(?:"+"|".join(RUN_PREFIXES)+r")-(\d+)", folder.name)
        if folder.is_dir() and match:
            runs.setdefault(match[1], []).append(folder)
        elif folder.is_dir():
            ignored.append(folder.name)
    if not runs:
        raise ValueError("No numeric Slurm run directories found.")
    fields = "JobIDRaw,JobName,State,ExitCode,ElapsedRaw,AllocCPUS,AllocTRES,NodeList,Submit,Start,End,MaxRSS"
    raw = subprocess.check_output(["sacct", "-j", ",".join(sorted(runs, key=int)), "-P", "--units=K", "--format="+fields], text=True)
    (args.output/"slurm-accounting.psv").write_text(raw)
    nodes = subprocess.check_output(["sinfo", "-N", "-h", "-o", "%N|%f|%G"], text=True)
    (args.output/"current-node-inventory.psv").write_text(nodes)
    telemetry, accounting = [], []
    for row in csv.DictReader(raw.splitlines(), delimiter="|"):
        identifier = row["JobIDRaw"]
        if identifier not in runs:
            continue  # No .batch/.extern double counting; raw step records remain available.
        gpus = allocated_gpus(row["AllocTRES"])
        seconds = int(row["ElapsedRaw"] or 0)
        accounting.append({**{k: row[k] for k in fields.split(",")}, "allocated_gpus": gpus,
            "allocated_gpu_hours_observed": seconds*gpus/3600,
            "allocated_cpu_hours_observed": seconds*int(row["AllocCPUS"] or 0)/3600,
            "run_directories": [str(p.relative_to(args.root)) for p in runs[identifier]],
            "complete_cost": row["State"] in {"COMPLETED", "FAILED", "TIMEOUT", "OUT_OF_MEMORY", "CANCELLED"}})
    for identifier, folders in runs.items():
        for folder in folders:
            # Explicitly bounded depth excludes copied source, weights, caches and score parts.
            paths = set()
            for parent in [folder, *[p for p in folder.iterdir() if p.is_dir() and p.name not in {"source", "venv", "cache", "media"}]]:
                for name in ["cost.json", "hardware.json", "exit_code.txt"]:
                    if (parent/name).is_file():
                        paths.add(parent/name)
            for path in sorted(paths):
                content = path.read_bytes()
                value = json.loads(content) if path.suffix == ".json" else content.decode().strip()
                telemetry.append({"job_id": identifier, "path": str(path.relative_to(args.root)),
                    "sha256": hashlib.sha256(content).hexdigest(), "value": value})
    def save(name, value):
        (args.output/name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    save("accounting.json", accounting)
    save("telemetry.json", telemetry)
    known = {r["JobIDRaw"] for r in accounting}
    save("manifest.json", {"collected_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(args.root), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "run_job_ids": sorted(runs, key=int), "missing_accounting_job_ids": sorted(set(runs)-known, key=int),
        "recognized_slurm_run_prefixes": RUN_PREFIXES, "non_slurm_directories_not_queried": sorted(ignored),
        "cost_definition": "Allocated GPU count times Slurm elapsed time; includes loading, numerical checks, failures and idle allocation. Live runs are right-censored snapshots, not final totals.",
        "measurement_definition": "Driver cost.json times are retained separately; nested timing fields overlap and must not be summed as GPU billing.",
        "hardware_definition": "hardware.json/cost.gpu are direct run evidence. Current node features only support explicitly labeled inference; not direct historical hardware records.",
        "unknown_memory": "Missing peak GPU allocation/reservation or MaxRSS remains unknown, never zero.",
        "scope": "Only job IDs identified by numeric OmniAnchor run directories; no unrelated job costs included; no scientific scores or labels opened.",
        "root_storage": subprocess.check_output(["df", "-h", str(args.root)], text=True)})
    print(f"Collected {len(accounting)} job records and {len(telemetry)} telemetry files.", flush=True)


if __name__ == "__main__":
    main()
