"""Compatibility shim: the acceptance check now lives in ``omnianchor.verification``."""

from omnianchor.verification import verify_native  # noqa: F401

if __name__ == "__main__":
    raise SystemExit("Use `omnianchor verify-model --config <study.yaml>` or import verify_native.")
