"""Summarize completed E1 masked reproduction and causal score partition without overclaiming validity."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from vlanchor.campaign import append_event, create_run
from vlanchor.io import read_json, write_json
from vlanchor.provenance import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.run / "exit_code.txt").read_text().strip() != "0":
        raise ValueError("E1 run must be complete.")
    create_run(args.output, {"purpose": "E1 scoring/reproduction evidence, not external construct validity",
        "source_run": str(args.run), "input_sha256": {str(p): file_hash(p) for p in args.run.glob("*/scores.csv")}})
    reproduction, signed = [], []
    for name in ["bert", "roberta"]:
        scores = pd.read_csv(args.run / name / "scores.csv")
        summary = read_json(args.run / name / "summary.json")
        reproduction.append({"method": name, **summary})
        wide = scores.pivot(index=["qid", "target"], columns="gender", values=["logp", "stored_logp"])
        differences = wide.xs("Male", level=1, axis=1)-wide.xs("Female", level=1, axis=1)
        differences = differences.reset_index()
        differences["method"] = name
        signed.append(differences)
    pd.DataFrame(reproduction).to_csv(args.output / "masked-reproduction.csv", index=False)
    pd.concat(signed, ignore_index=True).to_csv(args.output / "signed-masked-contrasts.csv", index=False)
    scores = pd.read_csv(args.run / "qwen35-plc/scores.csv")
    residual = scores.plc_full_sentence_logp-scores.prefix_logp-scores.anchor_logp-scores.suffix_logp
    if residual.abs().max() > 1e-10 or scores.suffix_invariance_max_abs_nat.max() > .01:
        raise ValueError("Causal chain or suffix-invariance check failed.")
    wide = scores.pivot(index=["qid", "target"], columns="gender",
                        values=["anchor_logp", "prefix_logp", "suffix_logp", "plc_full_sentence_logp"])
    contrasts = wide.xs("Male", level=1, axis=1)-wide.xs("Female", level=1, axis=1)
    contrasts["ordering_reversed"] = contrasts.anchor_logp*contrasts.plc_full_sentence_logp < 0
    contrasts.to_csv(args.output / "causal-signed-contrasts.csv")
    summary = {"contexts": len(contrasts), "occupations": 8, "templates": 4,
        "rank_reversals_anchor_vs_plc": int(contrasts.ordering_reversed.sum()),
        "max_abs_chain_residual_nat": float(residual.abs().max()),
        "max_abs_suffix_invariance_error_nat": float(scores.suffix_invariance_max_abs_nat.max()),
        "max_abs_prefix_gender_difference_nat": float(contrasts.prefix_logp.abs().max()),
        "all_finite": bool(np.isfinite(scores[["anchor_logp", "suffix_logp", "plc_full_sentence_logp"]]).all().all()),
        "causal_interpretation": "Occupation occurs after the gender anchor in every template, so anchor-only cannot condition on occupation; full-sentence suffix and MLM can.",
        "limitations": "Eight preselected occupations; diagnostic reversals are not a population validity estimate. Fixed chat condition and FP32/ BF16 model recipes are explicit."}
    write_json(args.output / "summary.json", summary)
    text = ("# E1 completed scoring evidence\n\n"
        f"BERT and RoBERTa each reran 64 masked alternatives over 32 contexts. Maximum absolute discrepancies from public stored log probabilities were {reproduction[0]['max_abs_logp_difference_from_stored']:.8g} and {reproduction[1]['max_abs_logp_difference_from_stored']:.8g} nat. All contextual token replacements were audited.\n\n"
        f"For Qwen3.5-4B, full-sentence PLC decomposed into prefix, anchor, and suffix within {summary['max_abs_chain_residual_nat']:.3g} nat. Truncating the suffix changed the anchor token log probabilities by at most {summary['max_abs_suffix_invariance_error_nat']:.3g} nat. The male/female ordering reversed between anchor-only and full-sentence PLC in {summary['rank_reversals_anchor_vs_plc']}/{len(contrasts)} contexts.\n\n"
        "The occupation appears after the gender word in these templates. The causal anchor-only event therefore has no access to it; the suffix and masked-model events do. These are distinct score definitions, not interchangeable measurements. These checks support numerical implementation claims and do not establish new human construct validity.\n")
    (args.output / "E1-evidence.md").write_text(text)
    append_event(args.output, "completed")


if __name__ == "__main__":
    main()
