"""Generate manuscript tables directly from completed, verified FMAT/OASIS analyses."""
import argparse
import json
from pathlib import Path

import pandas as pd

from vlanchor.campaign import append_event, create_run
from vlanchor.io import write_json
from vlanchor.provenance import file_hash


def emit_table(folder, name, caption, columns, rows):
    md = caption+"\n\n| "+" | ".join(columns)+" |\n| "+" | ".join(["---"]*len(columns))+" |\n"
    md += "".join("| "+" | ".join(map(str, row))+" |\n" for row in rows)
    (folder/f"{name}.md").write_text(md)
    def escape(value):
        return str(value).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
    tex = "\\begin{table}\n\\caption{"+escape(caption)+"}\n\\begin{tabular}{"+"l"*len(columns)+"}\n\\toprule\n"
    tex += " & ".join(map(escape, columns))+r" \\"+"\n\\midrule\n"
    tex += "".join(" & ".join(map(escape, row))+r" \\"+"\n" for row in rows)
    tex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    (folder/f"{name}.tex").write_text(tex)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["fmat", "fmat-supplement", "oasis", "output"]:
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    for folder in [args.fmat, args.fmat_supplement, args.oasis]:
        events = (folder/"events.jsonl").read_text().splitlines()
        if not events or json.loads(events[-1])["status"] != "completed":
            raise ValueError("A source analysis is incomplete.")
    sources = [args.fmat/"matched-correlations.csv", args.fmat_supplement/"paired-differences.csv",
        args.oasis/"direct-correlations.csv", args.oasis/"paired-differences.csv"]
    fmat, fm_pair, oasis, oa_pair = [pd.read_csv(p) for p in sources]
    if len(fm_pair) != 36 or set(oasis.n) != {201} or len(oasis) != 12:
        raise ValueError("Unexpected verified analysis coverage.")
    create_run(args.output, {"purpose": "Partial manuscript tables; remaining E1-E6 results still outstanding",
        "source_sha256": {str(p): file_hash(p) for p in sources}, "script_sha256": file_hash(Path(__file__)),
        "rounding": "Three decimals; source CSV retains full precision; no selected best-method highlighting",
        "latex": "booktabs required; no vertical rules"})
    try:
        labels = {"bert-base-uncased": "BERT base uncased (stored FMAT)", "roberta-base": "RoBERTa base (stored FMAT)",
            "qwen35-association": "Qwen3.5-4B association", "qwen35-contextual_gap": "Qwen3.5-4B contextual gap",
            "qwen3vl-association": "Qwen3-VL-4B association", "qwen3vl-contextual_gap": "Qwen3-VL-4B contextual gap",
            "qwen35-full-sentence-PLC": "Qwen3.5-4B full-sentence PLC"}
        f = fmat[fmat.subset == "all50"].set_index(["method", "study"])
        def interval(row, value, lower="ci_lower", upper="ci_upper"):
            return f"{row[value]:.3f} [{row[lower]:.3f}, {row[upper]:.3f}]"
        rows = [[label, *[interval(f.loc[(name, study)], "pearson_r", "pearson_ci_lower", "pearson_ci_upper")
            for study in ["d1a", "d1b"]]] for name, label in labels.items()]
        emit_table(args.output, "fmat-main", "FMAT external criterion: signed Pearson r [95% target-bootstrap CI], 50 targets per study. Common criterion and target aggregation; model and scoring-event differences remain.",
            ["Method", "Occupations", "Names"], rows)
        fm_pair.to_csv(args.output/"fmat-all36-exploratory-paired.csv", index=False)
        fmat.to_csv(args.output/"fmat-all-methods-and-subsets.csv", index=False)
        for metric in ["Pearson", "Spearman"]:
            o = oasis[oasis.metric == metric].set_index(["method", "dimension"])
            rows = [[label, *[interval(o.loc[(name, dimension)], "correlation") for dimension in ["V", "A"]]]
                for name, label in [("VLanchor", "VLanchor / Qwen3.5-4B"), ("qwen-embedding", "Official embedding"), ("qwen-reranker", "Official reranker")]]
            emit_table(args.output, "oasis-"+metric.lower(), f"OASIS protected201 images: direct {metric} correlation [95% image-bootstrap CI]. Four fixed affect anchors; no trained predictor.",
                ["Method", "Valence", "Arousal"], rows)
        paired = oa_pair[oa_pair.metric == "Pearson"]
        rows = []
        for _, row in paired.iterrows():
            rows.append([row.comparator, row.dimension, interval(row, "difference"), f"{row.primary_family_bh_q:.3f}"])
        emit_table(args.output, "oasis-paired", "OASIS frozen four-comparison Pearson family: native minus comparator, paired image bootstrap; BH q across all four tests.",
            ["Comparator", "Dimension", "Difference [95% CI]", "BH q"], rows)
        oa_pair.to_csv(args.output/"oasis-all-paired.csv", index=False)
        write_json(args.output/"coverage.json", {"fmat_methods": fmat.method.nunique(), "fmat_paired": len(fm_pair),
            "oasis_correlations": len(oasis), "oasis_primary_paired": len(paired),
            "missing": "E2textfinal/E3/E4/E5/E6/finalcost not filled; this is not a complete manuscript result package"})
        append_event(args.output, "completed")
    except BaseException as exc:
        append_event(args.output, "failed", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
