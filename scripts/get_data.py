from pathlib import Path
from dandi.download import download, DownloadExisting
import argparse
import shutil
import requests, zipfile, io
import json

from preprocess_for_cellpose import preprocess_dataset


ASTIH_ASCII = '''
                      █████     ███  █████     
                     ░░███     ░░░  ░░███      
    ██████    █████  ███████   ████  ░███████  
   ░░░░░███  ███░░  ░░░███░   ░░███  ░███░░███ 
    ███████ ░░█████   ░███     ░███  ░███ ░███ 
   ███░░███  ░░░░███  ░███ ███ ░███  ░███ ░███ 
  ░░████████ ██████   ░░█████  █████ ████ █████
   ░░░░░░░░ ░░░░░░     ░░░░░  ░░░░░ ░░░░ ░░░░░ 
axon segmentation training initiative for histology
                                             
'''

class ASTIHDataset:
    def __init__(self, name, id, url, desc, test_set, model_url):

        self.name = name
        self.dandi_id = id
        self.url = url
        self.desc = desc
        if isinstance(test_set, list):
            self.test_set_type = 'internal'
            self.test_subjects = test_set
        elif type(test_set) is str:
            self.test_set_type = 'external'
            self.test_set_url = test_set
        else:
            raise ValueError("test_set must be a list of subjects or a URL")
        self.model_release = model_url

def load_datasets():
    datalist_path = Path(__file__).parent / 'dataset_list.json'
    with open(datalist_path, 'r') as f:
        datalist = json.load(f)

    astih_dsets = []
    for dset_metadata in datalist:
        astih_dsets.append(ASTIHDataset(**dset_metadata))

    return astih_dsets

def download_data(url: str, dst_dir: str):
    """Download data and return path to unzipped data."""
    r = requests.get(url)
    assert r.ok, f"Failed to download {url}"
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(dst_dir)
    return Path(dst_dir) / z.namelist()[0]

def index_bids_dataset(data_dir: Path, file_ext: str):
    """
    Index the BIDS dataset and return a list of image for which a GT exists.
    """
    filenames = []

    # Look at derivative files
    for subject_dir in (data_dir / "derivatives" / "labels").glob("sub-*"):
        # every annotated image will have an axonmyelin mask
        for mask in (subject_dir / "micr").glob(f"*_seg-axonmyelin-manual.png"):
            # Get the corresponding image
            subject = subject_dir.name
            img_fname = Path(data_dir) / subject / "micr" / mask.name.replace(f"_seg-axonmyelin-manual.png", f".{file_ext}")
            assert img_fname.exists(), f"Image {img_fname} does not exist"
            filenames.append(img_fname)
    return filenames

def split_dataset(dset: ASTIHDataset, dset_path: Path, output_dir: Path):
    """
    Splits the dataset into training and testing sets.
    """
    train_dir = output_dir / "train"
    test_dir = output_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # quick and dirty check for file extension
    nb_tiff_files = len(list(dset_path.rglob('*.tif')))
    ext = 'png' if nb_tiff_files == 0 else 'tif'
    index = index_bids_dataset(dset_path, ext)

    # utility function to find corresponding GTs
    def find_gts(dset_path, img_path):
        subject = img_path.name.split("_")[0]
        gt_location = dset_path / "derivatives" / "labels" / subject / "micr"
        pattern = f"{img_path.stem}_seg-*-manual.png"
        gts = list(gt_location.glob(pattern))
        return gts
    
    # utility to conveniently copy files
    def copy_files_associated(img_path, gt_paths, dest_dir):
        shutil.copy(img_path, dest_dir)
        for gt_path in gt_paths:
            shutil.copy(gt_path, dest_dir)

    for indexed_img in index:
        subject = indexed_img.name.split("_")[0]
        gts = find_gts(dset_path, indexed_img)
        assert len(gts) > 0, f"No GT found for {indexed_img}"

        if dset.test_set_type == 'internal' and subject in dset.test_subjects:
            copy_files_associated(indexed_img, gts, test_dir)
        else:
            copy_files_associated(indexed_img, gts, train_dir)

    # If the test set is external, download it
    if dset.test_set_type == 'external':
        testset_path = download_data(dset.test_set_url, dset_path.parent)
        testset_index = index_bids_dataset(testset_path, ext)
        for indexed_img in testset_index:
            gts = find_gts(testset_path, indexed_img)
            assert len(gts) > 0, f"No GT found for {indexed_img}"
            copy_files_associated(indexed_img, gts, test_dir)



def main(make_splits: bool, preprocess_cellpose: bool = False):
    print(ASTIH_ASCII)

    # Create a directory to store the downloaded data
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    astih_dsets = load_datasets()

    # download dandisets
    urls = [dataset.url for dataset in astih_dsets]
    download(urls, data_dir, existing=DownloadExisting.OVERWRITE_DIFFERENT)

    if make_splits:
        # Create a directory to store the splits
        splits_dir = data_dir / "splits"
        splits_dir.mkdir(exist_ok=True)

        for dataset in astih_dsets:
            dataset_path = data_dir / dataset.dandi_id
            # Create a directory for each dataset
            dataset_split_dir = splits_dir / dataset.name
            dataset_split_dir.mkdir(exist_ok=True)

            # Split the dataset
            print(f"Splitting {dataset.name} dataset...")
            split_dataset(dataset, dataset_path, dataset_split_dir)

            if preprocess_cellpose:
                print(f"Preprocessing {dataset.name} dataset for Cellpose...")
                output_cellpose_dir = data_dir / "cellpose_pipeline" / f'cellpose_preprocessed_{dataset.name}'
                output_cellpose_dir.mkdir(parents=True, exist_ok=True)
                preprocess_dataset(dataset_split_dir, output_cellpose_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and split datasets.")
    parser.add_argument(
        "--make-splits",
        action="store_true",
        help="Make splits for the datasets.",
    )
    parser.add_argument(
        "--preprocess-cellpose",
        action="store_true",
        default=False,
        help="Preprocess the datasets for Cellpose training.",
    )
    args = parser.parse_args()

    if args.preprocess_cellpose and not args.make_splits:
        parser.error("--preprocess-cellpose requires --make-splits to be set.")

    main(args.make_splits, args.preprocess_cellpose)
