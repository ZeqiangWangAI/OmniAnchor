"""Explicit failures; failed observations are never represented by zero scores."""


class VLanchorError(ValueError):
    pass


class BoundaryError(VLanchorError):
    pass


class BudgetExceeded(VLanchorError):
    pass


class MissingMedia(VLanchorError):
    pass


class IncompatibleMeasurement(VLanchorError):
    pass


class MissingScores(VLanchorError):
    pass


class ResourceUnavailable(VLanchorError):
    pass
