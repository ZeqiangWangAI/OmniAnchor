"""The small contract shared by scientific and explicitly synthetic backends."""

from __future__ import annotations

from typing import Any, Protocol, Sequence

from ..types import Anchor, Bridge, Event, Sample


class Backend(Protocol):
    @property
    def identity(self) -> dict[str, Any]: ...

    def score_candidates(
        self, sample: Sample, bridge: Bridge, anchors: Sequence[Anchor],
        *, event: Event = "token_prefix",
    ) -> list[dict[str, Any]]: ...
