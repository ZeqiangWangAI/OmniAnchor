"""Freeze label-blind VIVA image swaps, portable media and variable candidate inventories."""
import argparse
import hashlib
import shutil
from pathlib import Path

from vlanchor.campaign import select_smoke
from vlanchor.io import load_samples, read_json, write_json
from vlanchor.provenance import file_hash


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    media=args.output/"media"
    media.mkdir()
    origins,donors,all_samples={}, {}, {}
    for split in ["train","dev","test"]:
        samples=load_samples(args.data/split/"samples.json")
        candidates=read_json(args.data/split/"candidates.json")
        ordered=sorted(samples,key=lambda s:hashlib.sha256(f"42\0{s.id}".encode()).hexdigest())
        for i,sample in enumerate(ordered):
            donor=next(s for s in ordered[i+1:]+ordered[:i] if s.group_id!=sample.group_id)
            donors[sample.id]=donor.id
        staged=[]
        for sample in samples:
            parts=[]
            for part in sample.parts:
                if part.type=="image":
                    source=Path(part.path)
                    digest=file_hash(source)
                    name=digest+source.suffix
                    if not (media/name).exists():
                        shutil.copyfile(source,media/name)
                    origins[sample.id]={"source":str(source),"sha256":digest,"file":"media/"+name}
                    part=part.model_copy(update={"path":"media/"+name})
                parts.append(part)
            staged.append(sample.model_copy(update={"parts":tuple(parts)}))
        all_samples[split]=staged
        write_json(args.output/f"{split}.json",staged)
        write_json(args.output/f"{split}-candidates.json",candidates)
    development=all_samples["train"]+all_samples["dev"]
    write_json(args.output/"development.json",development)
    write_json(args.output/"smoke32.json",select_smoke(development,16))
    write_json(args.output/"all-samples.json",sum(all_samples.values(),[]))
    write_json(args.output/"candidates.json",{k:v for split in all_samples for k,v in read_json(args.data/split/"candidates.json").items()})
    write_json(args.output/"donors.json",donors)
    write_json(args.output/"manifest.json",{"source_manifest_sha256":file_hash(args.data/"manifest.json"),
        "media":origins,"split_counts":{s:len(v) for s,v in all_samples.items()},
        "swap_rule":"within same frozen split: hash42 order, next sample with different exact-image group; retain recipient action/candidates/labels",
        "conditions":["image_action","action_only","mismatched_image_action"],"selection_uses_labels":False,
        "candidate_semantics":"public canonical value names only; no reasons or explanations","test_model_evaluations":0})


if __name__=="__main__":
    main()
