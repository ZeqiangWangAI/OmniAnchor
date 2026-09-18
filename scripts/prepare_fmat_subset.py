"""Freeze eight hash-selected occupations crossed with all four FMAT templates."""
import argparse
import hashlib
from pathlib import Path

import pandas as pd

from omnianchor.io import write_json
from omnianchor.provenance import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.source)
    occupations = frame.T_word.str.replace(r"^(?:a|an) ", "", regex=True)
    targets = sorted(occupations.unique(), key=lambda t: hashlib.sha256(f"42\0{t}".encode()).hexdigest())[:8]
    subset = frame[occupations.isin(targets) & frame.model.isin(["bert-base-uncased", "roberta-base"])]
    if len(subset) != 128 or subset.groupby(["model", "qid", "T_word"]).size().ne(2).any():
        raise ValueError("Expected32 contexts with paired gender tokens for each of two models.")
    args.output.mkdir(parents=True, exist_ok=False)
    subset.to_csv(args.output / "stored-subset.csv", index=False)
    write_json(args.output / "manifest.json", {"source_sha256": file_hash(args.source), "seed": 42,
        "selection": "strip initial a/an for occupation grouping only; first8 occupations by SHA256(seed NUL occupation), cross all4 templates; original target strings preserved",
        "targets": targets, "contexts": 32, "stored_rows": len(subset), "labels_used_for_selection": False,
        "purpose": "E1 small masked replication and named autoregressive full-sentence diagnostic"})


if __name__ == "__main__":
    main()
