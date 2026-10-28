import cv2
import matplotlib.pyplot as plt
import numpy as np

from config import (
    PLOT_FIGSIZE,
    WATERSHED_BOUNDARY_COLOR,
    WATERSHED_BOUNDARY_MARKER,
)


def plot_preprocessing_steps(
    original_image: np.ndarray,
    normalized: np.ndarray,
    thresholded: np.ndarray,
    denoised: np.ndarray,
    watershed_markers: np.ndarray,
) -> None:
    """Display the input and each preprocessing result in a single figure."""
    if original_image.ndim == 2:
        original_display = original_image
        segmented_display = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    else:
        original_display = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        segmented_display = original_display.copy()

    segmented_display[
        watershed_markers == WATERSHED_BOUNDARY_MARKER
    ] = WATERSHED_BOUNDARY_COLOR
    images = (
        (original_display, "Original"),
        (normalized, "Lighting normalized"),
        (thresholded, "Background removed"),
        (denoised, "Noise removed"),
        (segmented_display, "Watershed boundaries"),
    )


    figure = plt.figure(figsize=PLOT_FIGSIZE)
    grid = figure.add_gridspec(2, 6)
    slots = (
        grid[0, 0:3],
        grid[0, 3:6],
        grid[1, 0:2],
        grid[1, 2:4],
        grid[1, 4:6],
    )

    for slot, (image, title) in zip(slots, images):
        axis = figure.add_subplot(slot)
        axis.imshow(image, cmap="gray" if image.ndim == 2 else None)
        axis.set_title(title)
        axis.axis("off")

    figure.tight_layout()
    plt.show()
    plt.close(figure)
