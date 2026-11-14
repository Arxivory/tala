"""Confidence filtering and short-gap interpolation."""

from __future__ import annotations

import numpy as np
import pandas as pd


MAX_INTERPOLATION_GAP = 10


def prepare_landmark_samples(
    values,
    name: str,
    allow_all_missing: bool = False,
) -> np.ndarray:
    """Interpolate short missing intervals in landmark coordinates."""
    values = np.asarray(values, dtype=float)

    if not np.isfinite(values).any():
        if allow_all_missing:
            return values.copy()

        raise ValueError(f"No valid samples for {name}.")

    interpolated = pd.DataFrame(values).interpolate(
        method="linear",
        limit=MAX_INTERPOLATION_GAP,
        limit_area="inside",
    )

    interpolated = interpolated.ffill(
        limit=MAX_INTERPOLATION_GAP
    )

    interpolated = interpolated.bfill(
        limit=MAX_INTERPOLATION_GAP
    )

    return interpolated.to_numpy(dtype=float)