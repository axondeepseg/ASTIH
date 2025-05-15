from AxonDeepSeg.segment import get_model_type
from AxonDeepSeg.apply_model import axon_segmentation
from AxonDeepSeg.ads_utils import get_imshape, imread, imwrite
from torch import cuda
from pathlib import Path
import argparse
import re

from get_data import DATASETS


def find_input_images(datapath: Path):
    return [p for p in datapath.glob("*.png") if '_seg-' not in p.name]

def main(dset_name: None):
    data_splits_path = Path("data/splits")
    assert data_splits_path.exists(), "Data splits directory does not exist. Please run get_data.py with --make-splits arg first."
    for dset in DATASETS:
        if dset_name is not None and dset.name != dset_name:
            continue

        testset_path = data_splits_path / dset.name / 'test'
        input_imgs = find_input_images(testset_path)

        print(f"Applying model to {dset.name} dataset ({len(input_imgs)} images).")
        pattern = r"([^/]+)(?=\.zip$)"
        model_name = re.search(pattern, dset.model_release).group(1)
        model_path = Path("models") / model_name

        # ensure input imgs have the expected nb of channels; if not, overwrite input
        fileformat = '.png'
        n_channels = 1
        for img_path in input_imgs:
            imshape = get_imshape(str(img_path))
            needs_conversion = not (imshape[-1] == n_channels)
            if needs_conversion:
                print(f"Converting {img_path} to proper format.")
                img = imread(str(img_path))
                imwrite(str(img_path), img, fileformat)

        axon_segmentation(
            path_inputs=input_imgs,
            path_model=model_path,
            model_type=get_model_type(model_path),
            gpu_id=0 if cuda.is_available() else -1,
        )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply models to datasets.")
    parser.add_argument(
        "-d", "--dset-name",
        type=str,
        default=None,
        help="Name of the dataset to apply the model to. If not provided, all datasets will be processed.",
    )
    args = parser.parse_args()

    main(args.dset_name)