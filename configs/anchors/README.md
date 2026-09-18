# Anchor packs

These YAML files are inputs to a StudySpec's `anchors` field, not standalone studies.
Exact surfaces and stable IDs are preserved; no automatic translation or normalization.
`values20_en` uses the 20 official ValueEval label names. `affect12_en/zh` are independent
language packs requiring their own validation; shared IDs do not establish aligned scales.

Build general64/128/256 from the same frozen WiC training bundle with
`vlanchor.datasets.build_general_anchors(bundle, n=128)`. The function counts each
pair once, ranks exact lemmas by descending frequency then Unicode string order,
and returns a tuple of `Anchor` objects. It rejects non-WiC data, undeclared/non-train
splits, and insufficient unique lemmas. Pack sizes share identical prefix IDs.
Save the source bundle manifest and serialized pack before DWUG evaluation:

```python
from vlanchor.datasets import build_general_anchors, load_wic
from vlanchor.io import write_json

bundle = load_wic("data/wic/train/train.data.txt", split="train")
anchors = build_general_anchors(bundle, n=256)
write_json("runs/anchors/general256_en.json", {
    "anchors": [anchor.model_dump(mode="json") for anchor in anchors],
    "source_manifest": bundle.manifest,
})
```

The existing `general_anchors(language="en")` function is a separate 24-concept
legacy/default fixture bank. It does not produce the train-derived general128 pack.

The twelve affect anchors do not by themselves define three VAD scores. Predeclare a
directional projection or report a training-only linear probe; do not choose the most
correlated coordinate after observing held-out annotations.
