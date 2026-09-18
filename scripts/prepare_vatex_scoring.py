"""Freeze native retrieval input shards without inserting paired captions into videos."""
import argparse
import hashlib
from collections import Counter
from pathlib import Path

from vlanchor.io import load_samples,read_json,write_json
from vlanchor.provenance import file_hash
from vlanchor.types import TargetSpan


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--reference",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    inventory={}
    for language in ["en","zh"]:
        for split in ["dev","test"]:
            rows=read_json(args.data/language/split/"samples.json")
            for row in rows:
                for part in row["parts"]:
                    if part["path"]:
                        part["path"]="../vatex500-20260910-01/media/"+Path(part["path"]).name
                if row["id"] in inventory and inventory[row["id"]]!=row:
                    raise ValueError("Language inventories disagree on shared video inputs.")
                inventory[row["id"]]=row
    by_role={}
    for split in ["dev","test"]:
        videos=[r for r in inventory.values() if r["metadata"]["split"]==split and r["parts"][0]["type"]=="video"]
        captions={language:[r for r in inventory.values() if r["metadata"]["split"]==split and r["parts"][0]["type"]=="text" and r["language"]==language] for language in ["en","zh"]}
        videos.sort(key=lambda r:r["id"])
        for language,rows in captions.items():
            rows.sort(key=lambda r:r["id"])
            if len(rows)!=10*len(videos):
                raise ValueError("Require all10captions perselectedvideo perlanguage.")
            write_json(args.output/f"{split}-{language}.json",rows)
        write_json(args.output/f"{split}-videos.json",videos)
        by_role[split]={"videos":videos,**captions}
    videos=sorted(by_role["dev"]["videos"],key=lambda r:hashlib.sha256(f"42\0{r['id']}".encode()).hexdigest())[:16]
    smoke=list(videos)
    for language,subset in [("en",videos[:8]),("zh",videos[8:])]:
        for video in subset:
            candidates=[r for r in by_role["dev"][language] if r["metadata"]["videoID"]==video["metadata"]["videoID"]]
            smoke.append(min(candidates,key=lambda r:hashlib.sha256(f"42\0{r['id']}".encode()).hexdigest()))
    write_json(args.output/"smoke32.json",smoke)
    reference=[]
    for sample in load_samples(args.reference):
        target=TargetSpan.model_validate(sample.metadata["original_target"])
        parts=list(sample.parts)
        parts[target.part_index]=parts[target.part_index].model_copy(update={"text":sample.metadata["original_target_text_part"]})
        metadata={k:v for k,v in sample.metadata.items() if k not in {"target_rendering","original_target","original_target_text_part"}}
        reference.append(sample.model_copy(update={"parts":tuple(parts),"target":target,"metadata":metadata}))
    if len(reference)!=64 or any(s.metadata["split"]!="train" for s in reference):
        raise ValueError("Require64WiCtrain reference usages.")
    write_json(args.output/"reference64.json",reference)
    duplicates={}
    for language in ["en","zh"]:
        counts=Counter(r["parts"][0]["text"] for split in ["dev","test"] for r in by_role[split][language])
        duplicates[language]={"repeated_text_types":sum(n>1 for n in counts.values()),
            "rows_with_repeated_text":sum(n for n in counts.values() if n>1)}
    write_json(args.output/"manifest.json",{"source_manifest_sha256":file_hash(args.data/"manifest.json"),
        "role_counts":{s:{k:len(v) for k,v in rows.items()} for s,rows in by_role.items()},
        "smoke_selection":"seed42hash16devvideos+8EN+8ZHcaptions, label-blind technicalpilot",
        "reference":"same64WiCtrainusagegroups asDWUG, restoredunmarkedtext; associated_with calibrationseparate frommeans_in_context",
        "reference_source_sha256":file_hash(args.reference),"duplicate_caption_audit":duplicates,
        "primary_relevance":"original pairedvideo ID; identicalgenericcaptions mayhaveunannotatedalternativepositives; report limitation",
        "test_model_evaluations":0})


if __name__=="__main__":
    main()
