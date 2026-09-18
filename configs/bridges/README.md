# Bridge packs

A bridge is the relation sentence placed between the material and the anchor. Each YAML
file here holds one relation in one language under a top-level `bridges:` list, and a study
references it with `bridges: ../bridges/<file>.yaml` (path relative to the study file).
One study must use a single relation and language; different relations are different
studies. `expresses_value_seed8.json` is the frozen candidate pool of the paper's bridge
search and is not a pack.
