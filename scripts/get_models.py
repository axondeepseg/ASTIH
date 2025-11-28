from pathlib import Path

from get_data import load_datasets, download_data


def main():
    # Create a directory to store the downloaded models
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    astih_datasets = load_datasets()

    # Download models
    for dataset in astih_datasets:
        print('-------------------------')
        print(f"Downloading {dataset.name} model...")
        model_path = download_data(dataset.model_release, model_dir)
        print(f"Model downloaded to {model_path}")


if __name__ == "__main__":
    main()
