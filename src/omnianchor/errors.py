"""Explicit failures; failed observations are never represented by zero scores."""


class OmniAnchorError(ValueError):
    pass


class BoundaryError(OmniAnchorError):
    pass


class BudgetExceeded(OmniAnchorError):
    pass


class MissingMedia(OmniAnchorError):
    pass


class IncompatibleMeasurement(OmniAnchorError):
    pass


class MissingScores(OmniAnchorError):
    pass


class ResourceUnavailable(OmniAnchorError):
    pass
