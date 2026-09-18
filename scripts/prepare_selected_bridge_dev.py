"""Freeze the union of final train-selected bridges and reuse exact reference coordinates."""
import argparse
import hashlib
from pathlib import Path

from omnianchor.campaign import select_bridge_coordinates
from omnianchor.io import load_scores,load_spec,read_json,save_scores,write_json
from omnianchor.provenance import file_hash
from omnianchor.types import Bridge


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    spec=load_spec(root/"configs/studies/valueeval_fixed3.json")
    seed_ids={r["prefix"]:r["id"] for r in read_json(root/"configs/bridges/expresses_value_seed8.json")}
    runs={"reference_guided":44688,"validity":44788,"reliability":44796,"alpha":44806,"random":44787}
    args.output.mkdir(parents=True,exist_ok=False)
    sets={"fixed3":[b.model_dump() for b in spec.bridges]}
    sources,unique={},{}
    for method,job in runs.items():
        run=root/f"runs/bridge-search-{job}"
        if (run/"exit_code.txt").read_text().strip()!="0":
            raise ValueError("A selected search run is incomplete.")
        artifact=read_json(run/"search/selected.json")
        if artifact["status"]!="ok":
            raise ValueError("Search failed; do not substitute unselected templates.")
        selected=[]
        for row in artifact["bridges"]:
            canonical=seed_ids.get(row["prefix"],"rewrite_"+hashlib.sha256(row["prefix"].encode()).hexdigest()[:12])
            bridge=Bridge.model_validate({**row,"id":canonical})
            selected.append(bridge.model_dump())
            if canonical not in unique:
                unique[canonical]=bridge
                source=run/"search/candidates"/row["id"]/"reference.parquet"
                reference=select_bridge_coordinates(load_scores(source),[bridge])
                save_scores(reference,args.output/"reference"/f"{canonical}.parquet")
                sources[canonical]={"run":str(run),"original_bridge_id":row["id"],"reference_sha256":file_hash(source)}
        sets[method]=selected
    for bridge in spec.bridges:
        if bridge.id not in unique:
            source=root/"runs/bridge-search-44688/search/candidates"/bridge.id/"reference.parquet"
            save_scores(select_bridge_coordinates(load_scores(source),[bridge]),args.output/"reference"/f"{bridge.id}.parquet")
            unique[bridge.id]=bridge
            sources[bridge.id]={"reference_sha256":file_hash(source)}
    existing={b.id for b in spec.bridges}
    new=sorted([b for b in unique.values() if b.id not in existing],key=lambda b:b.id)
    for i,bridge in enumerate(new):
        write_json(args.output/f"spec-{i:05d}.json",spec.model_copy(update={"bridges":(bridge,)}))
    write_json(args.output/"manifest.json",{"sets":sets,"new_bridges":[b.model_dump() for b in new],
        "reference_sources":sources,"fixed3_existing_runs":[44695,44696,44767,44768,44811,44812,44694],
        "selection":"all final sets frozen fromtrain-onlysearch; no dev scores informed selection",
        "semantic_review":"Researcher acceptedB01-B09. Three additional generated phrasings received code/rule checks only; no fabricated human agreement. 'stated' may narrow 'expressed' to explicit statements.",
        "human_review_response_sha256":file_hash(root/"research/human-review/expresses-value-20260910-01/response-01.json"),
        "dev_samples":1896,"new_model_items":1896*20*len(new),"test_used":False})
    print(len(unique),"unique final/fixed bridges;",len(new),"new bridge scoring tasks")


if __name__=="__main__":
    main()
