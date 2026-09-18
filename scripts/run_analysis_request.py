"""Execute an explicit statistical analysis request inside a Slurm CPU allocation."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from omnianchor.io import read_json, write_json
from omnianchor.provenance import file_hash, runtime_manifest


def completed_array_runs(job_id, expected_tasks, prefix, runs_root):
    if not str(job_id).isdigit() or prefix not in {"development", "baseline-development", "viva-development"}:
        raise ValueError("Invalid explicit array request.")
    result = subprocess.check_output(["sacct", "-j", str(job_id), "-n", "-P", "--format=JobIDRaw,State,ExitCode"], text=True)
    rows = [line.split("|") for line in result.splitlines() if line.split("|", 1)[0].isdigit()]
    if len(rows) != expected_tasks or any(row[1:3] != ["COMPLETED", "0:0"] for row in rows):
        raise ValueError(f"Array {job_id} is not completely successful; no implicit filtering.")
    return [str(runs_root/f"{prefix}-{row[0]}") for row in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--job-output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Require scheduled CPU allocation for cluster statistics.")
    request = read_json(args.request)
    allowed = {"build_development_features.py", "evaluate_text_final.py", "analyze_oasis_direct.py",
        "analyze_viva.py", "analyze_selected_bridges.py", "analyze_dwug.py", "analyze_development_networks.py",
        "evaluate_empty_controls.py", "analyze_sensitivity.py", "build_vatex_retrieval.py",
        "analyze_vatex.py", "analyze_vatex_final.py", "analyze_viva_interaction.py", "fit_development_probes.py",
        "bootstrap_probe_differences.py", "run_sensitivity_analysis.py"}
    if request["script"] not in allowed:
        raise ValueError("Request is not an approved statistical driver.")
    arguments = request["arguments"].copy()
    if "--output" in arguments:
        raise ValueError("Output is owned by the unique Slurm analysis run.")
    runs_root = args.job_output.parent
    for flag, inputs in request.get("run_inputs", {}).items():
        allowed_flags = {"--runs", "--native-runs", "--baseline-runs"}
        if request["script"] == "analyze_viva_interaction.py":
            allowed_flags.update({"--base-runs", "--control-runs"})
        if flag not in allowed_flags or flag in arguments:
            raise ValueError("Ambiguous scoring run inputs.")
        runs = inputs.get("fixed", []).copy()
        for array in inputs.get("arrays", []):
            runs.extend(completed_array_runs(array["job_id"], array["expected_tasks"], array["prefix"], runs_root))
        arguments.extend([flag, *runs])
    script = Path(__file__).with_name(request["script"])
    command = [sys.executable, str(script), *arguments, "--output", str(args.job_output/"analysis")]
    write_json(args.job_output/"analysis-request.json", {"request": request,
        "request_sha256": file_hash(args.request), "resolved_command": command,
        "runtime": runtime_manifest(), "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "role": "CPU statistics/feature assembly only; no neural inference or GPU fallback"})
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
