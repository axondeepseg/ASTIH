from pathlib import Path
from dandi.download import download

DANDISET_URLS = [
    "https://dandiarchive.org/dandiset/001436/draft",
    "https://dandiarchive.org/dandiset/001350/draft",
    "https://dandiarchive.org/dandiset/001440/draft",
]

def main():
    # Create a directory to store the downloaded data
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    download(DANDISET_URLS, data_dir)

if __name__ == "__main__":
    main()