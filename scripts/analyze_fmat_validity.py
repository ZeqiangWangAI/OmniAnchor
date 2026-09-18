"""Compare new native scores with FMAT using identical targets, directions and external criteria."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from omnianchor.campaign import create_run, append_event
from omnianchor.io import read_json, write_json
from omnianchor.provenance import file_hash


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--stored",type=Path,required=True)
    parser.add_argument("--pilot-manifest",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if (args.run/"exit_code.txt").read_text().strip()!="0":
        raise ValueError("Require completed native run.")
    records=[]
    for model in ["qwen35","qwen3vl"]:
        frame=pd.read_csv(args.run/model/"scores.csv")
        for (study,method),part in frame.groupby(["study","method"]):
            wide=part.pivot(index=["template","target"],columns="gender",values="logp")
            if wide[["Male","Female"]].isna().any().any():
                raise ValueError("Missing paired gender scores.")
            d=(wide.Male-wide.Female).rename("lpr").reset_index()
            d["method"],d["study"]=model+"-"+method,study
            records.append(d)
    for study in ["d1a","d1b"]:
        frame=pd.read_csv(args.run/f"qwen35-plc-{study}/scores.csv")
        wide=frame.pivot(index=["qid","target"],columns="gender",values="plc_full_sentence_logp")
        d=(wide.Male-wide.Female).rename("lpr").reset_index().rename(columns={"qid":"template"})
        d["method"],d["study"]="qwen35-full-sentence-PLC",study
        records.append(d)
    stored=pd.read_csv(args.stored/"E1-stored-scores.csv")
    stored=stored.rename(columns={"model":"method","query":"template","T_word":"original_target","LPR_male_minus_female":"lpr"})
    stored["target"]=stored.original_target
    records.append(stored[["study","method","template","target","lpr"]])
    scores=pd.concat(records,ignore_index=True)
    scores.loc[scores.study=="d1a","target"]=scores.loc[scores.study=="d1a","target"].str.replace(r"^(?:a|an) ","",regex=True)
    if not np.isfinite(scores.lpr).all():
        raise ValueError("Nonfinite native or stored score.")
    group=scores.groupby(["study","method","template"]).lpr
    if (group.std(ddof=1)<=0).any():
        raise ValueError("Undefined template standardization; preserve rather than silently omit.")
    scores["z"]=group.transform(lambda x:(x-x.mean())/x.std(ddof=1))
    aggregate=scores.groupby(["study","method","target"],as_index=False).z.mean()
    pilot=set(read_json(args.pilot_manifest)["targets"])
    create_run(args.output,{"purpose":"FMAT matched criterion, no probe; all fixed native variants reported",
        "primary":"signedPearson withpublishedP_male", "secondary":"Spearman", "bootstrap":1000,"seed":42,
        "bootstrap_unit":"target; paired across all methods", "confidence":.95,
        "normalization":"withinmethod/template sample-SD across50targets thenaverage; bootstrap conditional on fixed normalization",
        "source_hashes":{str(p):file_hash(p) for p in args.run.rglob("scores.csv")},
        "gold_hashes":{n:file_hash(args.data/n) for n in ["stats.occupation.csv","stats.name.csv"]},
        "script_sha256":file_hash(Path(__file__)),
        "limitations":"model and conditionalevent differences are explicit; original8occupation numericalpilot known, withremaining42reportedseparately"})
    summaries,differences=[],[]
    try:
        for study,goldfile,key in [("d1a","stats.occupation.csv","job"),("d1b","stats.name.csv","name")]:
            gold=pd.read_csv(args.data/goldfile).rename(columns={key:"target"}).set_index("target").P_male
            matrix=aggregate[aggregate.study==study].pivot(index="target",columns="method",values="z")
            if len(matrix)!=50 or matrix.isna().any().any() or set(matrix.index)!=set(gold.index):
                raise ValueError("Not every method covers exactlythe same50targets.")
            matrix["P_male"]=gold.loc[matrix.index]
            matrix.to_csv(args.output/f"{study}-aligned-scores.csv")
            for subset,view in [("all50",matrix)]+([("exclude8pilot42",matrix.loc[~matrix.index.isin(pilot)])] if study=="d1a" else []):
                y=view.P_male.to_numpy()
                rng=np.random.default_rng(42)
                draws=rng.integers(len(view),size=(1000,len(view)))
                estimates={}
                for method in matrix.columns.drop("P_male"):
                    x=view[method].to_numpy()
                    point=float(pearsonr(x,y).statistic)
                    boots=np.array([pearsonr(x[i],y[i]).statistic for i in draws])
                    estimates[method]=(point,boots)
                    low,high=np.quantile(boots,[.025,.975])
                    summaries.append({"study":study,"subset":subset,"method":method,"n":len(view),
                        "pearson_r":point,"spearman_rho":float(spearmanr(x,y).statistic),
                        "pearson_ci_lower":float(low),"pearson_ci_upper":float(high)})
                for primary in ["qwen35-association","qwen35-contextual_gap","qwen3vl-association","qwen3vl-contextual_gap"]:
                    for comparator in ["bert-base-uncased","roberta-base","qwen35-full-sentence-PLC"]:
                        p,b=estimates[primary]
                        q,c=estimates[comparator]
                        lo,hi=np.quantile(b-c,[.025,.975])
                        differences.append({"study":study,"subset":subset,"primary":primary,"comparator":comparator,
                            "pearson_difference":p-q,"ci_lower":float(lo),"ci_upper":float(hi)})
        scores.to_csv(args.output/"template-scores.csv",index=False)
        pd.DataFrame(summaries).to_csv(args.output/"matched-correlations.csv",index=False)
        pd.DataFrame(differences).to_csv(args.output/"paired-differences.csv",index=False)
        write_json(args.output/"summary.json",summaries)
        append_event(args.output,"completed")
    except BaseException as exc:
        append_event(args.output,"failed",error=repr(exc))
        raise


if __name__=="__main__":
    main()
