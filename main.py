from pathlib import Path
from src.stage4 import *
from src.preprocessing import *
from src.postprocess import *

IMAGE_NAME = "blueBG_rice\\image621.jpg"
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data"
IMAGE_PATH = DATA_PATH / IMAGE_NAME

def main() -> None:
    run_preprocessing_pipeline(IMAGE_PATH)
    img = load_image(str(IMAGE_PATH))

    binary, dist, seeds, markers = segment_rice(img, seed_ratio=0.2)
    contours = detect_contours(markers)
    contours_img = draw_contours(img, contours)

    df = extract_features(contours)
    df, ref_length = classify_grains(df, broken_ratio=0.75, ref_percentile=90)
    compute_statistics(df, ref_length)

    show_pipeline(img, dist, markers, seeds, binary, contours_img)
    
    """processed_img = to_bgr(binary)
    classified_img = draw_classified(img, contours, df)
    classified_processed_img = draw_classified(processed_img, contours, df, alpha=0.6)
    show_classification(img, classified_img, processed_img, classified_processed_img)"""
    
    print("\nFeature Table")
    print(df)

    summary_table = show_postprocess(
        img, contours, df,
        ref_length=ref_length,
        broken_ratio=0.75,          # keep identical to classify_grains
        save_path="postprocess.png" # optional
    )

if __name__ == "__main__":
    main()
