from __future__ import annotations

"""Classic, CPU-friendly segmentation methods for ZO-1 images.

This module provides a lightweight Otsu/adaptive thresholding routine that
approximates the behaviour of the deep learning models used elsewhere in the
app but runs entirely on the CPU.

Each function expects an 8-bit single channel image (``img_u8``).  Colour
images should be converted to grayscale before calling these functions.
The function returns a label image where each integer corresponds to a cell,
the binary membrane mask used for seeding, and a dictionary of basic summary
statistics (cell count and mean/median area/diameter).

Example
-------
>>> labels, membrane, stats = segment_zo1_otsu(img)
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import cv2
from skimage import filters, morphology, segmentation, measure, feature, util


@dataclass
class SegmentationResult:
    """Container for segmentation output."""

    labels: np.ndarray  # int32 labelled image
    membrane: np.ndarray  # bool membrane mask
    stats: Dict[str, float]
    binary_mask: Optional[np.ndarray] = None  # raw threshold mask


def _postprocess_and_watershed(
    mem_mask: np.ndarray,
    img_u8: np.ndarray,
    min_obj: int = 200,
    min_peak_dist: int = 8,
    skeleton_thickness: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Refine membrane mask and apply watershed to obtain cell labels and a skeleton.

    Parameters
    ----------
    mem_mask : np.ndarray
        Initial membrane mask.
    img_u8 : np.ndarray
        Original grayscale image.
    min_obj : int, optional
        Minimum object size for cleaning, by default 200.
    min_peak_dist : int, optional
        Minimum distance between watershed seeds, by default 8.
    skeleton_thickness : int, optional
        Pixel thickness of the returned skeleton, by default 1 (single-pixel).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Labelled image and processed skeleton mask.
    """

    mem = morphology.remove_small_objects(mem_mask.astype(bool), min_size=5)
    mem = morphology.binary_closing(mem, morphology.disk(1))
    mem = morphology.binary_dilation(mem, morphology.disk(1))
    mem = mem.astype(np.uint8)

    inside = 1 - mem
    dist = cv2.distanceTransform(inside, cv2.DIST_L2, 3)
    dist_supp = cv2.GaussianBlur(dist, (0, 0), 1.0)
    peaks = feature.peak_local_max(
        dist_supp,
        labels=inside.astype(bool),
        min_distance=min_peak_dist,
        exclude_border=False,
    )
    markers = np.zeros_like(img_u8, dtype=np.int32)
    for i, (r, c) in enumerate(peaks, start=1):
        markers[r, c] = i
    markers = morphology.dilation(markers, morphology.disk(1))

    grad = filters.sobel(img_u8)
    lab = segmentation.watershed(grad, markers=markers, mask=inside.astype(bool))
    lab = morphology.remove_small_objects(lab, min_size=min_obj)

    # Extract boundaries from the labelled regions to avoid spurious lines
    skel = segmentation.find_boundaries(lab, mode="outer")
    if skeleton_thickness > 1:
        skel = morphology.dilation(skel, morphology.disk(max(1, skeleton_thickness // 2)))

    return lab.astype(np.int32), skel.astype(bool)


def compute_cell_metrics(
    labels: np.ndarray, pixel_size: Optional[float] = None
) -> Tuple['pd.DataFrame', Dict[str, float]]:
    """Compute per-cell metrics and summary statistics."""

    import pandas as pd

    props = measure.regionprops_table(
        labels,
        properties=(
            'label',
            'area',
            'perimeter',
            'equivalent_diameter',
            'centroid',
            'eccentricity',
        ),
    )
    df = pd.DataFrame(props)

    stats = {
        'n_cells': int(len(df)),
        'mean_area': float(df['area'].mean()) if len(df) else 0.0,
        'median_area': float(df['area'].median()) if len(df) else 0.0,
        'mean_equiv_diam': float(df['equivalent_diameter'].mean()) if len(df) else 0.0,
        'median_equiv_diam': float(df['equivalent_diameter'].median()) if len(df) else 0.0,
    }

    if pixel_size and pixel_size > 0:
        df['area_um2'] = df['area'] * (pixel_size ** 2)
        df['equivalent_diameter_um'] = df['equivalent_diameter'] * pixel_size
        stats.update(
            {
                'mean_area_um2': float(df['area_um2'].mean()) if len(df) else 0.0,
                'median_area_um2': float(df['area_um2'].median()) if len(df) else 0.0,
                'mean_equiv_diam_um': float(df['equivalent_diameter_um'].mean()) if len(df) else 0.0,
                'median_equiv_diam_um': float(df['equivalent_diameter_um'].median()) if len(df) else 0.0,
            }
        )

    return df, stats


def _apply_ridge(img: np.ndarray) -> np.ndarray:
    """Enhance thin membranes using a ridge filter."""

    ridge = filters.sato(img.astype(float), sigmas=(1, 2, 3))
    ridge = ridge > np.percentile(ridge, 70)
    return ridge.astype(np.uint8)


def segment_zo1_otsu(
    img_u8: np.ndarray,
    adaptive: bool = False,
    block: int = 51,
    offset: float = 0.0,
    min_obj: int = 200,
    smooth_sigma: float = 1.0,
    use_ridge: bool = True,
    min_peak_dist: int = 8,
    thresh_multiplier: float = 1.0,
    skeleton_thickness: int = 1,
) -> SegmentationResult:
    """Segment membranes using Otsu or adaptive thresholding.

    Parameters
    ----------
    img_u8 : np.ndarray
        Input 8-bit image.
    adaptive : bool, optional
        Use adaptive thresholding instead of Otsu, by default False.
    block : int, optional
        Block size for adaptive thresholding, by default 51.
    offset : float, optional
        Offset for adaptive thresholding, by default 0.0.
    min_obj : int, optional
        Minimum object size for watershed, by default 200.
    smooth_sigma : float, optional
        Gaussian blur sigma, by default 1.0.
    use_ridge : bool, optional
        Apply ridge filtering before thresholding, by default True.
    min_peak_dist : int, optional
        Minimum distance between watershed peaks, by default 8.
    thresh_multiplier : float, optional
        Multiplier applied to Otsu threshold, by default 1.0.
    skeleton_thickness : int, optional
        Pixel thickness of the returned skeleton, by default 1.
    """

    blur = cv2.GaussianBlur(img_u8, (0, 0), smooth_sigma)
    if adaptive:
        if block % 2 == 0:
            block += 1  # ensure odd
        thresh = filters.threshold_local(blur, block_size=block, offset=offset)
    else:
        thresh = filters.threshold_otsu(blur) * thresh_multiplier
        thresh = np.clip(thresh, 0, 255)
    labels = (blur > thresh).astype(np.uint8)

    mem = cv2.bitwise_and(labels, _apply_ridge(blur)) if use_ridge else labels
    lab, mem_proc = _postprocess_and_watershed(
        mem, blur, min_obj, min_peak_dist, skeleton_thickness
    )
    _, stats = compute_cell_metrics(lab)
    return SegmentationResult(lab, mem_proc, stats, labels)

