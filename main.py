from pathlib import Path

from src.preprocessing import run_preprocessing_pipeline

IMAGE_NAME = "test2.jpg"
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data"
IMAGE_PATH = DATA_PATH / IMAGE_NAME


def main() -> None:
    run_preprocessing_pipeline(IMAGE_PATH)


if __name__ == "__main__":
    main()
