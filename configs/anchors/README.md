# Anchor packs

Each YAML file holds a top-level `anchors:` list and is referenced from a study as
`anchors: ../anchors/<file>.yaml`. Surfaces are exact strings: no translation, trimming or
normalisation. `values20_en` uses the 20 official ValueEval label names. `affect12_en` and
`affect12_zh` are independent language packs; shared IDs do not align their scales, and
twelve affect anchors do not by themselves define valence, arousal or dominance scores.

General anchor banks for semantic-shift studies are built from a frozen WiC training
bundle with `omnianchor.datasets.build_general_anchors(bundle, n=128)`, which ranks exact
lemmas by frequency and Unicode order; save the bundle manifest with the pack.
