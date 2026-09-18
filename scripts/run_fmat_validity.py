"""FMAT matched external criterion: fixed native associations and contextual gap completions."""
import argparse
import os
from pathlib import Path
from time import perf_counter

import pandas as pd

from omnianchor import measure
from omnianchor.backends.hf import HFBackend
from omnianchor.campaign import create_run, append_event
from omnianchor.io import load_spec, save_scores, write_json
from omnianchor.provenance import file_hash
from omnianchor.types import Anchor, Bridge, Part, Sample
from verify_native import verify_native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Require Slurm CUDA allocation.")
    spec = load_spec(args.config)
    bridges = tuple(Bridge(id=f"association_{i+1}", prefix=p, relation="associated_with") for i,p in enumerate([
        "A gender associated with this material is:\n",
        "The gender commonly associated with this material is:\n",
        "This material is associated with the gender:\n"]))
    pair = (Anchor(id="Male", surface="male"), Anchor(id="Female", surface="female"))
    gap = Bridge(id="gap_completion", prefix="The word completing [MASK] is:\n", relation="fills_gap")
    create_run(args.output, {"purpose":"FMAT same50targets and published male-proportion criterion; no learned probe",
        "study_spec":spec.model_dump(), "association_bridges":[b.model_dump() for b in bridges],
        "association_anchors":[a.model_dump() for a in pair], "gap_bridge":gap.model_dump(),
        "data_sha256":{n:file_hash(args.data/n) for n in ["d1a.csv","d1b.csv"]},
        "source_sha256":file_hash(Path(__file__)),"slurm_job_id":os.environ["SLURM_JOB_ID"],
        "score_definition":"Male minus Female log probabilities; within-template sample-SD standardization over50targets, then mean across templates, exactly as frozen FMAT aggregation",
        "comparison_limits":"Same external criterion; different models and conditional events. Gap completion provides all original context and exact gender alternatives; association uses male/female. No claim of identical masked probability.",
        "development":"No search or selection among templates, no ground-truth proportions read by scorer; original8occupation diagnostic already seen and disclosed."})
    import torch
    torch.manual_seed(42)
    backend = HFBackend(spec.model, spec.resources, shared_prefill=False)
    backend._ensure_loaded()
    verification=verify_native(backend, output=args.output/"native-verification.json",
                  anchors=[Anchor(id="m1",surface="a strong emotion"),Anchor(id="m2",surface="an emotional response")])
    if verification["status"]!="passed":
        raise RuntimeError("Native verification failed.")
    started=perf_counter()
    rows=[]
    try:
        for study,kind in [("d1a","Occupation"),("d1b","Name")]:
            frame=pd.read_csv(args.data/f"{study}.csv")
            frame=frame[frame.model=="bert-base-uncased"]
            frame=frame.copy()
            frame["target_key"]=(frame.T_word.str.replace(r"^(?:a|an) ","",regex=True)
                                 if study=="d1a" else frame.T_word)
            targets=sorted(frame.target_key.unique())
            if len(targets)!=50:
                raise ValueError("Require complete original50target inventory.")
            for i,target in enumerate(targets):
                sample=Sample(id=f"{study}:{i}",parts=(Part(type="text",text=f"{kind}: {target}"),),source="fmat")
                current=spec.model_copy(update={"anchors":pair,"bridges":bridges})
                table=measure([sample],current,backend=backend)
                save_scores(table,args.output/study/"association"/f"part-{i:03d}.parquet")
                if table.manifest["execution"]["failed_items"]:
                    raise RuntimeError("Association scoring failed.")
                for row in table.frame.to_dict("records"):
                    rows.append({"study":study,"method":"association","target":target,"template":row["bridge_id"],
                        "gender":row["anchor_id"],"logp":row["raw_logp"]})
                for qid,sub in frame[frame.target_key==target].groupby("qid"):
                    query=sub.iloc[0]["query"].replace("{TARGET}",sub.iloc[0].T_word)
                    anchors=tuple(Anchor(id=r.MASK,surface=r.M_word) for _,r in sub.iterrows())
                    if len(anchors)!=2 or {a.id for a in anchors}!={"Male","Female"}:
                        raise ValueError("Original gender alternatives are not paired.")
                    item=Sample(id=f"{study}:{i}:{qid}",parts=(Part(type="text",text=query),),source="fmat")
                    current=spec.model_copy(update={"anchors":anchors,"bridges":(gap,)})
                    table=measure([item],current,backend=backend)
                    save_scores(table,args.output/study/"gap"/f"part-{i:03d}-{qid}.parquet")
                    if table.manifest["execution"]["failed_items"]:
                        raise RuntimeError("Contextual gap scoring failed.")
                    for row in table.frame.to_dict("records"):
                        rows.append({"study":study,"method":"contextual_gap","target":target,"template":str(qid),
                            "gender":row["anchor_id"],"logp":row["raw_logp"]})
                append_event(args.output,"target_completed",study=study,target=target)
                print(study,i+1,target,flush=True)
        pd.DataFrame(rows).to_csv(args.output/"scores.csv",index=False)
        write_json(args.output/"cost.json",{"measurement_seconds":perf_counter()-started,"score_items":len(rows),
            "gpu":torch.cuda.get_device_name(),"peak_allocated_bytes":torch.cuda.max_memory_allocated()})
        append_event(args.output,"completed")
    except BaseException as exc:
        append_event(args.output,"failed",error=repr(exc))
        raise


if __name__=="__main__":
    main()
