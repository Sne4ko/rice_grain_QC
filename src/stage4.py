import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Impossible de charger {path}")
    return img

def show_pipeline(img, dist, markers, seeds, binary, contour_img):
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes = axes.ravel()  

    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original")

    axes[1].imshow(dist, cmap="jet")
    axes[1].set_title("Distance Transform")

    axes[2].imshow(seeds * 255, cmap="gray")
    axes[2].set_title("Seeds")

    axes[3].imshow(binary, cmap="gray")
    axes[3].set_title("After Watershed")

    axes[4].imshow(markers, cmap="nipy_spectral")
    axes[4].set_title("Watershed Labels")

    axes[5].imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
    axes[5].set_title("Contours")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

def segment_rice(img,seed_ratio=0.7):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (binary > 0).astype(np.uint8)  
    dist = cv2.distanceTransform(mask,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    seed_cutoff = seed_ratio * dist.max()#à ajuster
    seeds = (dist > seed_cutoff).astype(np.uint8)
    _, markers = cv2.connectedComponents(seeds)
    markers = markers + 1
    markers[(mask == 1) & (seeds == 0)] = 0  #zone inconnue
    img_ws = img.copy()
    markers = cv2.watershed(img_ws,markers.astype(np.int32))
    #résultat 
    separated = np.zeros_like(mask)
    separated[markers > 1] = 255
    return separated, dist, seeds, markers

def detect_contours(markers):
    contours = []
    for label in np.unique(markers):
        if label <= 1:
            continue
        grain = np.uint8(markers == label)
        cnts, _ = cv2.findContours(grain,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        contours.extend(cnts)
    return contours

def draw_contours(img, contours):
    output = img.copy()
    cv2.drawContours(output,contours,-1,(0, 255, 0),2)
    return output

def extract_features(contours, min_area=50):
    features = []
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        rect = cv2.minAreaRect(cnt)
        width_rect, height_rect = rect[1]
        length = max(width_rect, height_rect)
        width = min(width_rect, height_rect)
        aspect_ratio = (length / width if width > 0 else 0)
        perimeter = cv2.arcLength(cnt, True)
        features.append({"ID": i,"Length": round(length, 2),"Width": round(width, 2),"Area": round(area, 2),"Perimeter": round(perimeter, 2),"Aspect_Ratio": round(aspect_ratio, 2)})
    return pd.DataFrame(features)

def compute_statistics(df):
    if len(df) == 0:
        print("No grains detected.")
        return
    print("\n===== RESULTS =====")
    print(f"Detected objects : {len(df)}")





FULL_COLOR = (0, 200, 0)  
BROKEN_COLOR = (0, 0, 255)  


def classify_grains(df, broken_ratio=0.75, ref_percentile=90):
    """
    Label each grain as Full or Broken.
    Reference full-grain length = a high percentile of all lengths
    (robust to a few broken pieces and to merged grains).
    A grain is Broken if Length < broken_ratio * reference length.
    """
    if len(df) == 0:
        return df, 0.0

    df = df.copy()
    ref_length = np.percentile(df["Length"], ref_percentile)
    df["Length_Ratio"] = (df["Length"] / ref_length).round(2)
    df["Class"] = np.where(df["Length_Ratio"] >= broken_ratio, "Full", "Broken")
    return df, ref_length


def draw_classified(img, contours, df, alpha=0.35):
    """
    Overlay classification on the ORIGINAL raw image:
    green = full, red = broken (semi-transparent fill + solid outline).
    """
    overlay = img.copy()
    for _, row in df.iterrows():
        cnt = contours[int(row["ID"])]
        color = FULL_COLOR if row["Class"] == "Full" else BROKEN_COLOR
        cv2.drawContours(overlay, [cnt], -1, color, thickness=-1)

    output = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    for _, row in df.iterrows():
        cnt = contours[int(row["ID"])]
        color = FULL_COLOR if row["Class"] == "Full" else BROKEN_COLOR
        cv2.drawContours(output, [cnt], -1, color, 2)

    # Legend
    n_full = int((df["Class"] == "Full").sum())
    n_broken = int((df["Class"] == "Broken").sum())
    cv2.putText(output, f"Full: {n_full}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, FULL_COLOR, 2)
    cv2.putText(output, f"Broken: {n_broken}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, BROKEN_COLOR, 2)
    return output

def to_bgr(image):
    """Make a single-channel mask drawable with colored overlays."""
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image


def show_classification(img, classified_img, processed_img, classified_processed_img):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    panels = [
        (img, "Original raw image"),
        (classified_img, "Classified on raw image\nFull (green) vs Broken (red)"),
        (processed_img, "Processed image (separated mask)"),
        (classified_processed_img, "Classified on processed image\nFull (green) vs Broken (red)"),
    ]
    for ax, (im, title) in zip(axes.ravel(), panels):
        ax.imshow(cv2.cvtColor(to_bgr(im), cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.show()
    
    

def compute_statistics(df, ref_length=None):
    if len(df) == 0:
        print("No grains detected.")
        return
    print("\n===== RESULTS =====")
    print(f"Detected objects : {len(df)}")
    if "Class" in df.columns:
        n_full = int((df["Class"] == "Full").sum())
        n_broken = len(df) - n_full
        print(f"Full grains      : {n_full} ({100 * n_full / len(df):.1f}%)")
        print(f"Broken grains    : {n_broken} ({100 * n_broken / len(df):.1f}%)")
        if ref_length:
            print(f"Reference length : {ref_length:.1f} px")
