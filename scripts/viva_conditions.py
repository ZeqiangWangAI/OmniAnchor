"""Explicit input removals for VIVA; recipient labels never enter model parts."""
from vlanchor.types import Part, Sample


def condition_sample(sample: Sample, condition: str, donor: Sample | None = None) -> Sample:
    if condition == "image_action":
        parts = sample.parts
    elif condition == "action_only":
        parts = tuple(p for p in sample.parts if p.type == "text")
    elif condition == "image_only":
        parts = tuple(p for p in sample.parts if p.type == "image")
    elif condition == "empty":
        parts = (Part(type="text", text=""),)
    elif condition == "mismatched_image_action":
        if donor is None or donor.metadata["split"] != sample.metadata["split"] or donor.group_id == sample.group_id:
            raise ValueError("Swap must change the image source within the same split.")
        image = next(p for p in donor.parts if p.type == "image")
        parts = tuple(image if p.type == "image" else p for p in sample.parts)
    else:
        raise ValueError("Unknown VIVA condition.")
    if not parts:
        raise ValueError("Requested modality is absent from this recipient.")
    return sample.model_copy(update={"parts": parts, "metadata": {**sample.metadata,
        "condition": condition, "image_donor": donor.id if condition == "mismatched_image_action" else None}})
