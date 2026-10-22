from pathlib import Path
import cv2
import numpy as np

from config import (
    BINARY_MAX_VALUE,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID_SIZE,
    MEDIAN_BLUR_KERNEL,
    THRESHOLD_VALUE,
    WATERSHED_DILATION_ITERATIONS,
    WATERSHED_MARKER_OFFSET,
    WATERSHED_DISTANCE_MASK_SIZE,
    WATERSHED_FOREGROUND_RATIO,
    WATERSHED_KERNEL_SIZE,
    WATERSHED_UNKNOWN_MARKER,
)
from .visualization import plot_preprocessing_steps


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


def apply_watershed(
    binary_image: np.ndarray,
    original_image: np.ndarray,
) -> np.ndarray:
    """Separate touching grains and return the watershed marker labels."""
    distance = cv2.distanceTransform(
        binary_image,
        cv2.DIST_L2,
        WATERSHED_DISTANCE_MASK_SIZE,
    )
    _, sure_foreground = cv2.threshold(
        distance,
        WATERSHED_FOREGROUND_RATIO * distance.max(),
        BINARY_MAX_VALUE,
        cv2.THRESH_BINARY,
    )
    sure_foreground = np.uint8(sure_foreground)

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
