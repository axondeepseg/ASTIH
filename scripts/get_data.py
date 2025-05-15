from pathlib import Path
from dandi.download import download
import argparse
import shutil
import requests, zipfile, io


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

DATASETS = [
    ASTIHDataset(
        name="TEM1",
        id="001436",
        url="https://dandiarchive.org/dandiset/001436/0.250512.1625",
        desc="TEM Images of Corpus Callosum in Control and Cuprizone-Intoxicated Mice with Axon and Myelin Segmentations",
        test_set=[
            "sub-nyuMouse26"
        ],
        model_url="https://github.com/axondeepseg/default-TEM-model/releases/download/r20240403/model_seg_mouse_axon-myelin_tem_light.zip"
    ),
    ASTIHDataset(
        name="TEM2",
        id="001350",
        url="https://dandiarchive.org/dandiset/001350/0.250511.1527",
        desc="TEM Images of Corpus Callosum in Flox/SRF-cKO Mice",
        test_set="https://github.com/axondeepseg/data_axondeepseg_srf_testing/archive/refs/tags/r20250513-neurips2025.zip",
        model_url="https://github.com/axondeepseg/model_seg_unmyelinated_tem/releases/download/r20240708-stanford/model_seg_unmyelinated_stanford_tem_best.zip"
    ),
    ASTIHDataset(
        name="SEM1",
        id="001442",
        url="https://dandiarchive.org/dandiset/001442/0.250512.1626",
        desc="SEM Images of Rat Spinal Cord with Axon and Myelin Segmentations with Myelinated and Unmyelinated Axon Segmentations",
        test_set=[
            "sub-rat6"
        ],
        model_url="https://github.com/axondeepseg/default-SEM-model/releases/download/r20240403/model_seg_rat_axon-myelin_sem_light.zip"
    ),
    ASTIHDataset(
        name="BF1",
        id="001440",
        url="https://dandiarchive.org/dandiset/001440/0.250509.1913",
        desc="BF Images of Rat Nerves at Different Regeneration Stages with Axon and Myelin Segmentations",
        test_set=[
            "sub-uoftRat02",
            "sub-uoftRat07"
        ],
        model_url="https://github.com/axondeepseg/default-BF-model/releases/download/r20240405/model_seg_rat_axon-myelin_bf_light.zip"
    )
]

def index_bids_dataset(data_dir: Path):
    """
    Index the BIDS dataset and return a list of image for which a GT exists.
    """
    filenames = []

    # Look at derivative files
    for subject_dir in (data_dir / "derivatives" / "labels").glob("sub-*"):
        # every annotated image will have an axonmyelin mask
        for mask in (subject_dir / "micr").glob("*_seg-axonmyelin-manual.png"):
            # Get the corresponding image
            subject = subject_dir.name
            img_fname = Path(data_dir) / subject / "micr" / mask.name.replace("_seg-axonmyelin-manual.png", ".png")
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
    index = index_bids_dataset(dset_path)

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
        r = requests.get(dset.test_set_url)
        assert r.ok, f"Failed to download {dset.test_set_url}"
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(dset_path.parent)
        testset_path = Path(dset_path.parent) / z.namelist()[0]
        testset_index = index_bids_dataset(testset_path)
        for indexed_img in testset_index:
            gts = find_gts(testset_path, indexed_img)
            assert len(gts) > 0, f"No GT found for {indexed_img}"
            copy_files_associated(indexed_img, gts, test_dir)



def main(make_splits: bool):
    # Create a directory to store the downloaded data
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # download dandisets
    urls = [dataset.url for dataset in DATASETS]
    # download(urls, data_dir)

    if make_splits:
        # Create a directory to store the splits
        splits_dir = data_dir / "splits"
        splits_dir.mkdir(exist_ok=True)

        for dataset in DATASETS:
            dataset_path = data_dir / dataset.dandi_id
            # Create a directory for each dataset
            dataset_split_dir = splits_dir / dataset.name
            dataset_split_dir.mkdir(exist_ok=True)

            # Split the dataset
            print(f"Splitting {dataset.name} dataset...")
            split_dataset(dataset, dataset_path, dataset_split_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and split datasets.")
    parser.add_argument(
        "--make-splits",
        action="store_true",
        help="Make splits for the datasets.",
    )
    args = parser.parse_args()

    main(args.make_splits)