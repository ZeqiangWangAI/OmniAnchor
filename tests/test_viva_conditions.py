import pytest

from scripts.viva_conditions import condition_sample
from vlanchor.types import Part, Sample


def test_viva_controls_preserve_recipient_action_and_remove_named_modalities():
    a = Sample(id="a", group_id="image-a", parts=(Part(type="image", path="a.jpg"),
        Part(type="text", text="Helping a person")), metadata={"split": "dev"})
    b = Sample(id="b", group_id="image-b", parts=(Part(type="image", path="b.jpg"),
        Part(type="text", text="Different donor action")), metadata={"split": "dev"})
    swapped = condition_sample(a, "mismatched_image_action", b)
    assert swapped.id == "a" and swapped.parts[0].path == "b.jpg"
    assert swapped.parts[1].text == "Helping a person"
    assert condition_sample(a, "image_only").parts == (a.parts[0],)
    assert condition_sample(a, "action_only").parts == (a.parts[1],)
    assert condition_sample(a, "empty").parts == (Part(type="text", text=""),)
    assert condition_sample(a, "image_action").parts == a.parts
    with pytest.raises(ValueError, match="same split"):
        condition_sample(a, "mismatched_image_action", b.model_copy(update={"metadata": {"split": "test"}}))
