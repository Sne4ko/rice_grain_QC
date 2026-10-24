from pathlib import Path
import cv2
import numpy as np

from config import (
    BINARY_MAX_VALUE,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID_SIZE,
    MEDIAN_BLUR_KERNEL,
    THRESHOLD_VALUE,
    WATERSHED_CUT_THICKNESS,
    WATERSHED_DEFECT_DEPTH_RATIO,
    WATERSHED_DILATION_ITERATIONS,
    WATERSHED_DISTANCE_MASK_SIZE,
    WATERSHED_DISTANCE_SMOOTH_SIGMA,
    WATERSHED_FOREGROUND_RATIO,
    WATERSHED_KERNEL_SIZE,
    WATERSHED_MARKER_OFFSET,
    WATERSHED_MIN_DEFECT_DEPTH,
    WATERSHED_MIN_SEED_AREA,
    WATERSHED_MIN_SPLIT_AREA,
    WATERSHED_SPLIT_SOLIDITY,
    WATERSHED_UNKNOWN_MARKER,
)
from src.visualization import plot_preprocessing_steps


def normalize_lighting(image: np.ndarray) -> np.ndarray:
    """Convert an image to grayscale and normalize its local contrast with CLAHE."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID_SIZE,
    )
    return clahe.apply(gray)


def background_subtraction(
    image: np.ndarray,
    threshold_val: int = THRESHOLD_VALUE,
) -> np.ndarray:
    """Threshold a grayscale image into a binary mask of foreground grains."""
    _, thresholded = cv2.threshold(
        image,
        threshold_val,
        BINARY_MAX_VALUE,
        cv2.THRESH_BINARY,
    )
    return thresholded


def remove_noise(
    binary_image: np.ndarray,
    kernel_size: int = MEDIAN_BLUR_KERNEL,
) -> np.ndarray:
    """Remove small specks from a binary mask using a median filter."""
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    return cv2.medianBlur(binary_image, kernel_size)


def _split_concave_blobs(binary: np.ndarray) -> np.ndarray:
    """Cut touching grains apart at their concave notches (e.g. V-shaped pairs)."""
    count, blobs, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    result = binary.copy()
    pad = 2
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if area < WATERSHED_MIN_SPLIT_AREA:
            continue
        blob = (blobs[y:y + h, x:x + w] == i).astype(np.uint8)
        padded = np.pad(blob, pad)
        contours, _ = cv2.findContours(padded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        if hull_area <= 0 or area / hull_area >= WATERSHED_SPLIT_SOLIDITY:
            continue  # already (nearly) convex: a single grain
        try:
            defects = cv2.convexityDefects(cnt, cv2.convexHull(cnt, returnPoints=False))
        except cv2.error:
            continue
        if defects is None:
            continue
        defects = defects.reshape(-1, 4)  # shape differs between OpenCV versions

        min_depth = max(WATERSHED_MIN_DEFECT_DEPTH, WATERSHED_DEFECT_DEPTH_RATIO * np.sqrt(area))
        reach = max(h, w) + 2 * pad
        cut = padded.copy()
        for start, end, far, depth in defects:
            if depth / 256.0 < min_depth:
                continue
            p = cnt[far, 0].astype(np.float32)
            chord_mid = (cnt[start, 0].astype(np.float32) + cnt[end, 0].astype(np.float32)) / 2
            direction = p - chord_mid
            norm = float(np.linalg.norm(direction))
            if norm < 1e-6:
                continue
            direction /= norm

            q = p + direction * (reach - 1)
            for step in range(1, reach):
                nxt = p + direction * step
                xi, yi = int(round(nxt[0])), int(round(nxt[1]))
                if not (0 <= xi < padded.shape[1] and 0 <= yi < padded.shape[0]) or padded[yi, xi] == 0:
                    q = nxt
                    break
            cv2.line(
                cut,
                (int(round(p[0])), int(round(p[1]))),
                (int(round(q[0])), int(round(q[1]))),
                0,
                WATERSHED_CUT_THICKNESS,
            )

        inner = cut[pad:-pad, pad:-pad]
        region = result[y:y + h, x:x + w]
        region[(blob == 1) & (inner == 0)] = 0
    return result


def _find_sure_foreground(binary_image: np.ndarray) -> np.ndarray:
    """Find one seed per grain using a per-blob (not global) distance threshold."""
    binary = (binary_image > 0).astype(np.uint8) * BINARY_MAX_VALUE
    binary = _split_concave_blobs(binary)

    distance = cv2.distanceTransform(binary, cv2.DIST_L2, WATERSHED_DISTANCE_MASK_SIZE)
    if WATERSHED_DISTANCE_SMOOTH_SIGMA > 0:
        # smooths jagged-edge bumps that would otherwise create fake local peaks
        distance = cv2.GaussianBlur(distance, (0, 0), WATERSHED_DISTANCE_SMOOTH_SIGMA)

    count, blobs, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    seeds = np.zeros_like(binary)
    for i in range(1, count):
        x, y, w, h, _ = stats[i]
        in_blob = blobs[y:y + h, x:x + w] == i
        local_distance = distance[y:y + h, x:x + w]
        peak = local_distance[in_blob].max()
        core = in_blob & (local_distance >= WATERSHED_FOREGROUND_RATIO * peak)
        seeds[y:y + h, x:x + w][core] = BINARY_MAX_VALUE

    # drop tiny seed fragments
    count, seed_labels, stats, _ = cv2.connectedComponentsWithStats(seeds)
    small = np.where(stats[:, cv2.CC_STAT_AREA] < WATERSHED_MIN_SEED_AREA)[0]
    small = small[small != 0]
    seeds[np.isin(seed_labels, small)] = 0
    return seeds


def apply_watershed(
    binary_image: np.ndarray,
    original_image: np.ndarray,
) -> np.ndarray:
    """Separate touching grains and return the watershed marker labels."""
    sure_foreground = _find_sure_foreground(binary_image)

    kernel = np.ones(
        (WATERSHED_KERNEL_SIZE, WATERSHED_KERNEL_SIZE),
        np.uint8,
    )
    sure_background = cv2.dilate(
        binary_image,
        kernel,
        iterations=WATERSHED_DILATION_ITERATIONS,
    )
    unknown = cv2.subtract(sure_background, sure_foreground)

    _, markers = cv2.connectedComponents(sure_foreground)
    markers = markers + WATERSHED_MARKER_OFFSET
    markers[unknown == BINARY_MAX_VALUE] = WATERSHED_UNKNOWN_MARKER

    if original_image.ndim == 2:
        watershed_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    else:
        watershed_image = original_image.copy()
    return cv2.watershed(watershed_image, markers)


def run_preprocessing_pipeline(image_path: str | Path) -> np.ndarray:
    """Run all preprocessing stages, plot their results, and return watershed labels."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Image could not be loaded: {image_path}")

    normalized = normalize_lighting(image)
    thresholded = background_subtraction(normalized)
    denoised = remove_noise(thresholded)
    segmented_markers = apply_watershed(denoised, image)

    plot_preprocessing_steps(
        image,
        normalized,
        thresholded,
        denoised,
        segmented_markers,
    )
    return segmented_markers
