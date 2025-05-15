from AxonDeepSeg.download_model import download_model
from pathlib import Path

from get_data import DATASETS


def main():
    # Create a directory to store the downloaded models
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    # Download models
    for dataset in DATASETS:
        print(f"Downloading {dataset.name} model...")
        model_path = download_model(dataset, model_dir)


if __name__ == "__main__":
    main()