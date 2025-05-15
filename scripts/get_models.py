from pathlib import Path

from get_data import DATASETS, download_data


def main():
    # Create a directory to store the downloaded models
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    # Download models
    for dataset in DATASETS:
        print('-------------------------')
        print(f"Downloading {dataset.name} model...")
        model_path = download_data(dataset.model_release, model_dir)
        print(f"Model downloaded to {model_path}")


if __name__ == "__main__":
    main()