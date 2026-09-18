"""Local-file adapters. Fetching/redistributing upstream data is never implicit."""

from .common import DatasetBundle, grouped_hash_splits, sha256_file
from .media import load_oasis, load_vatex, load_viva
from .semantic import build_general_anchors, general_anchors, load_dwug, load_wic
from .text import VALUEEVAL_LABELS, load_chinese_emobank, load_emobank, load_fmat, load_valueeval

DATASET_REGISTRY = {
    "valueeval": load_valueeval, "emobank": load_emobank,
    "chinese_emobank": load_chinese_emobank, "oasis": load_oasis,
    "viva": load_viva, "wic": load_wic, "dwug": load_dwug,
    "vatex": load_vatex, "fmat": load_fmat,
}


def load_dataset(name: str, **kwargs) -> DatasetBundle:
    try:
        adapter = DATASET_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset {name!r}; choose {sorted(DATASET_REGISTRY)}") from exc
    return adapter(**kwargs)


__all__ = ["DatasetBundle", "DATASET_REGISTRY", "load_dataset", "grouped_hash_splits",
           "sha256_file", "general_anchors", "build_general_anchors", "VALUEEVAL_LABELS",
           *[f"load_{k}" for k in DATASET_REGISTRY]]
