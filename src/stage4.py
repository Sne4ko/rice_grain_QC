import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Processig function
def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Impossible de charger {path}")
    return img

def show_pipeline(img,dist,markers,seeds,binary,contour_img):
    fig, axes = plt.subplots(1,6,figsize=(20, 6))
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(dist,cmap="jet")
    axes[1].set_title("Distance Transform")
    axes[1].axis("off")
    axes[2].imshow(seeds * 255,cmap="gray")
    axes[2].set_title("Seeds")
    axes[2].axis("off")
    axes[3].imshow(binary,cmap="gray")
    axes[3].set_title("After Watershed")
    axes[3].axis("off")
    axes[4].imshow(markers,cmap="nipy_spectral")
    axes[4].set_title("Watershed Labels")
    axes[4].axis("off")
    axes[5].imshow(cv2.cvtColor(contour_img,cv2.COLOR_BGR2RGB))
    axes[5].set_title("Contours")
    axes[5].axis("off")
    plt.tight_layout()
    plt.show()

def segment_rice(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (binary > 0).astype(np.uint8)  #binary for distance transform
    dist = cv2.distanceTransform(mask,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    seed_cutoff = 0.7* dist.max()#à ajuster
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