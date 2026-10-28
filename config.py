"""Configuration values for rice-grain preprocessing and visualization."""

MEDIAN_BLUR_KERNEL = 5
THRESHOLD_VALUE = 120
BINARY_MAX_VALUE = 255
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

WATERSHED_DISTANCE_MASK_SIZE = 5
WATERSHED_FOREGROUND_RATIO = 0.6          # now relative to each blob's OWN max distance
WATERSHED_DISTANCE_SMOOTH_SIGMA = 1.0     # blur on the distance map (0 = off)
WATERSHED_MIN_SEED_AREA = 5               # ignore seed fragments smaller than this (px)
WATERSHED_SPLIT_SOLIDITY = 0.96           # only blobs less convex than this are checked for notches (speed gate)
WATERSHED_MIN_DEFECT_DEPTH = 3            # notch must be at least this deep (px) ...
WATERSHED_DEFECT_DEPTH_RATIO = 0.15       # ... and this fraction of sqrt(blob area)
WATERSHED_MIN_SPLIT_AREA = 100            # ignore blobs smaller than this (px)
WATERSHED_CUT_THICKNESS = 2               # width of the cut line (px)
WATERSHED_KERNEL_SIZE = 3
WATERSHED_DILATION_ITERATIONS = 10
WATERSHED_MARKER_OFFSET = 1
WATERSHED_UNKNOWN_MARKER = 0
WATERSHED_BOUNDARY_MARKER = -1

PLOT_FIGSIZE = (16, 4)
WATERSHED_BOUNDARY_COLOR = (255, 0, 0)


