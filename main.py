from pathlib import Path
from src.stage4 import *
from src.preprocessing import run_preprocessing_pipeline

IMAGE_NAME = "test2.jpg"
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data"
IMAGE_PATH = DATA_PATH / IMAGE_NAME


def main() -> None:
    run_preprocessing_pipeline(IMAGE_PATH)
    img = load_image(str(IMAGE_PATH))
    binary, dist, seeds, markers = segment_rice(img)
    contours = detect_contours(markers)
    contours_img=draw_contours(img, contours)
    df = extract_features(contours)
    compute_statistics(df)
    show_pipeline(img,dist,markers,seeds,binary,contours_img)
    print("\nFeature Table")
    print(df)

if __name__ == "__main__":
    main()
