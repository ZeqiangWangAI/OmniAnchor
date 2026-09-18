"""Direct image-human VA validity: fixed bipolar contrasts, no trained prediction layer."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from vlanchor import to_matrix
from vlanchor.analysis import bh_fdr
from vlanchor.campaign import create_run, append_event, merge_score_shards
from vlanchor.io import load_samples, load_scores, read_json, write_json
from vlanchor.provenance import file_hash


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-run",type=Path,required=True)
    parser.add_argument("--baseline-run",type=Path,required=True)
    parser.add_argument("--evaluation-samples",type=Path,required=True)
    parser.add_argument("--labels",type=Path,nargs="+",required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--frozen-evaluation",type=Path)
    args=parser.parse_args()
    for root in [args.native_run,args.baseline_run]:
        if (root/"exit_code.txt").read_text().strip()!="0":
            raise ValueError("Require completed runs; no partial-case filtering.")
    samples=load_samples(args.evaluation_samples)
    final=read_json(args.frozen_evaluation) if args.frozen_evaluation else None
    if final:
        if final["status"]!="frozen" or file_hash(args.evaluation_samples)!=final["samples_sha256"]:
            raise ValueError("Final evaluation population differs from freeze.")
        if any(s.metadata["split"]!="test" for s in samples) or [s.id for s in samples]!=final["sample_ids"]:
            raise ValueError("Final analysis requires the exact test role and IDs.")
        if len(args.labels)!=1 or file_hash(args.labels[0])!=final["labels_sha256"]:
            raise ValueError("Final gold labels differ from freeze.")
        for folder in [args.native_run/"measurement",args.baseline_run/"qwen-embedding",args.baseline_run/"qwen-reranker"]:
            if read_json(folder/"manifest.json")["frozen_evaluation"]["sha256"]!=file_hash(args.frozen_evaluation):
                raise ValueError("Model run did not use this final protocol.")
    elif any(s.metadata["split"] not in {"train","dev"} for s in samples):
        raise ValueError("This development analysis rejects test.")
    ids=[s.id for s in samples]
    labels=pd.concat([pd.read_csv(p,index_col="sample_id") for p in args.labels])
    labels.index=labels.index.astype(str)
    if labels.index.duplicated().any():
        raise ValueError("Duplicate labels.")
    gold=labels.loc[ids,["V","A"]].to_numpy(float)
    parts=sorted((args.native_run/"measurement").glob("part-*.parquet"))
    tables=[load_scores(p) for p in parts]
    all_ids=[s["id"] for t in tables for s in t.manifest["samples"]]
    table=merge_score_shards(tables,all_ids)
    matrix=to_matrix(table,variant="raw_logp",missing="error")
    required=["pleasant","unpleasant","aroused","calm"]
    if set(matrix.anchor_ids)!=set(required):
        raise ValueError("Not the predeclared directVA anchor set.")
    values={"VLanchor":matrix.values[[matrix.sample_ids.index(i) for i in ids]][:,
        [matrix.anchor_ids.index(c) for c in required]]}
    sources={"native_parts":{str(p):file_hash(p) for p in parts}}
    for method in ["qwen-embedding","qwen-reranker"]:
        path=args.baseline_run/method/"matrix.npz"
        with np.load(path,allow_pickle=False) as data:
            row_ids,col_ids=data["sample_ids"].tolist(),data["anchor_ids"].tolist()
            if len(row_ids)!=len(set(row_ids)) or set(row_ids)!=set(all_ids) or set(col_ids)!=set(required):
                raise ValueError("Baseline population or anchors differ from native scoring.")
            values[method]=data["scores"][[row_ids.index(i) for i in ids]][:,[col_ids.index(c) for c in required]]
        sources[method]=file_hash(path)
    values={name:np.asarray(v,dtype=np.float64) for name,v in values.items()}
    values={name:np.column_stack([v[:,0]-v[:,1],v[:,2]-v[:,3]]) for name,v in values.items()}
    if not all(np.isfinite(v).all() for v in [gold,*values.values()]):
        raise ValueError("Nonfinite direct scores/labels.")
    groups=np.array([s.group_id or s.id for s in samples])
    blocks=[np.flatnonzero(groups==g) for g in np.unique(groups)]
    rng=np.random.default_rng(42)
    draws=[np.concatenate([blocks[j] for j in rng.integers(len(blocks),size=len(blocks))]) for _ in range(1000)]
    create_run(args.output,{"purpose":"Direct VA construct validity; no probe or learned weights; "+("frozen final evaluation" if final else "developmental evidence"),
        "evaluation_samples_sha256":file_hash(args.evaluation_samples),"sample_ids":ids,
        "n":len(ids),"source_groups":len(blocks),"scoring_source_hashes":sources,
        "labels_sha256":{str(p):file_hash(p) for p in args.labels},"script_sha256":file_hash(Path(__file__)),
        "contrasts":{"V":"pleasant-minus-unpleasant","A":"aroused-minus-calm"},
        "native_aggregation":"mean raw logprob over3fixed bridges before contrast","seed":42,
        "bootstrap":"1000pairedsourcegroupdraws,95%percentileCI","test_used":bool(final),
        "frozen_evaluation_sha256":file_hash(args.frozen_evaluation) if final else None,
        "pvalue":"two-sided centered paired bootstrap: (1+sum(abs(delta_star-delta)>=abs(delta)))/(B+1); approximate, MonteCarloresolution1/1001",
        "primary_family":"4 Pearson differences: native minus two baselines, acrossV/A; BHq.05",
        "statistical_source":"https://aclanthology.org/D12-1091/; two-sided extension of the centered bootstrap procedure"})
    summaries,differences=[],[]
    try:
        for j,dimension in enumerate(["V","A"]):
            for metric_name,metric in [("Pearson",pearsonr),("Spearman",spearmanr)]:
                results={}
                for name,x in values.items():
                    point=float(metric(x[:,j],gold[:,j]).statistic)
                    boot=np.array([metric(x[i,j],gold[i,j]).statistic for i in draws])
                    valid=np.isfinite(boot)
                    if valid.sum()<800 or not np.isfinite(point):
                        raise ValueError("Undefined direct validity; preserve diagnostic instead of altering anchors.")
                    lo,hi=np.quantile(boot[valid],[.025,.975])
                    summaries.append({"method":name,"dimension":dimension,"metric":metric_name,
                        "correlation":point,"ci_lower":float(lo),"ci_upper":float(hi),"n":len(ids)})
                    results[name]=(point,boot)
                p,b=results["VLanchor"]
                for name in ["qwen-embedding","qwen-reranker"]:
                    q,c=results[name]
                    delta=b-c
                    valid=np.isfinite(delta)
                    if valid.sum()<800:
                        raise ValueError("Insufficient paired defined draws.")
                    lo,hi=np.quantile(delta[valid],[.025,.975])
                    null=delta[valid]-(p-q)
                    pvalue=(1+int((np.abs(null)>=abs(p-q)).sum()))/(len(null)+1)
                    differences.append({"primary":"VLanchor","comparator":name,"dimension":dimension,
                        "metric":metric_name,"difference":p-q,"ci_lower":float(lo),"ci_upper":float(hi),
                        "centered_bootstrap_p":pvalue,"bootstrap_bias":float(np.mean(delta[valid])-(p-q))})
        pd.DataFrame(summaries).to_csv(args.output/"direct-correlations.csv",index=False)
        difference_table=pd.DataFrame(differences)
        primary=difference_table.metric=="Pearson"
        difference_table.loc[primary,"primary_family_bh_q"]=bh_fdr(difference_table.loc[primary,"centered_bootstrap_p"].to_numpy())
        difference_table.to_csv(args.output/"paired-differences.csv",index=False)
        np.savez_compressed(args.output/"aligned-direct-scores.npz",sample_ids=np.array(ids),human=gold,**values)
        write_json(args.output/"summary.json",summaries)
        append_event(args.output,"completed")
    except BaseException as exc:
        append_event(args.output,"failed",error=repr(exc))
        raise


if __name__=="__main__":
    main()
