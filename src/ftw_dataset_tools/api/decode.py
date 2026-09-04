"""Derived DECODE label layers (field boundary and distance to boundary).

DECODE (Waldner et al. 2021, "Detect, Consolidate, Delineate: Scalable Mapping
of Field Boundaries Using Satellite Images", Remote Sensing 13(11):2197) trains a
multi-task network with three targets: field extent, field boundary, and
distance to the field boundary.

The field extent target is the existing ``semantic_2_class`` mask. The other two
are *derived* from it rather than rasterized from vectors, so this module holds
the array-level derivations. Both are ports of the reference implementation used
for the FTW bakeoff, kept deliberately close to it so pre-generated rasters match
what that code computes on the fly:
https://github.com/fieldsoftheworld/ftw-prue/blob/main/decode/data_module.py

The derivation relies on adjacent fields being separated in the source mask.
``create-masks`` burns the boundary lines into the non-3-class masks as
background (see ``_rasterize_mask``), so two touching fields are already split by
a one-pixel background line and each ends up with its own closed boundary ring.
"""

from __future__ import annotations

import numpy as np

# scipy is imported inside the functions that need it: this module is reachable
# from every `ftwd` invocation via masks.py -> commands, and an eager import
# adds ~0.5s to CLI startup for commands that never touch masks.

# Presence-only labels use 3 for "background, but unlabelled" rather than 0.
# Both derivations treat it as plain background.
PRESENCE_ONLY_CLASS = 3


def _without_presence_only(mask: np.ndarray) -> np.ndarray:
    """Return a copy of ``mask`` with the presence-only class folded into background."""
    result = mask.astype(np.int32, copy=True)
    result[result == PRESENCE_ONLY_CLASS] = 0
    return result


def boundary_from_mask(mask: np.ndarray) -> np.ndarray:
    """
    Derive the DECODE field-boundary layer from a semantic mask.

    A pixel is a boundary when it is a field pixel whose 3x3 neighbourhood is not
    uniform, which marks the inner one-pixel ring of every field.

    ``maximum_filter``/``minimum_filter`` use scipy's default ``reflect`` mode, so
    a field running off the edge of the chip gets no boundary along that edge -
    the chip border is not a field border.

    Args:
        mask: 2D label array (0 background, >0 field, 3 presence-only background)

    Returns:
        2D uint8 array, 1 on field boundaries and 0 elsewhere
    """
    from scipy import ndimage

    values = _without_presence_only(mask)
    local_max = ndimage.maximum_filter(values, size=3)
    local_min = ndimage.minimum_filter(values, size=3)
    return ((local_max != local_min) & (values > 0)).astype(np.uint8)


def distance_from_mask(mask: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Derive the DECODE distance-to-boundary layer from a semantic mask.

    Each field pixel gets its Euclidean distance to the nearest background pixel,
    normalized by the largest such distance in the chip so values land in [0, 1].

    That normalization is per chip, which is what DECODE trains against - the loss
    and the downstream watershed only compare values within a single chip. It does
    mean the stored values are not comparable *between* chips, so the unnormalized
    maximum is returned as well and gets written into the raster tags, making the
    scaling invertible.

    Args:
        mask: 2D label array (0 background, >0 field, 3 presence-only background)

    Returns:
        Tuple of (2D float32 array in [0, 1], maximum distance in pixels before
        normalization; 0.0 when the chip contains no field pixels)
    """
    from scipy import ndimage

    values = _without_presence_only(mask)
    binary = (values > 0).astype(np.uint8)
    distance = ndimage.distance_transform_edt(binary)

    max_distance = float(distance.max())
    if max_distance > 0:
        distance = distance / max_distance

    return distance.astype(np.float32), max_distance
