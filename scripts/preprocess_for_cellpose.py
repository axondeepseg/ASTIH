'''This file provides utilities to preprocess the dataset into a format suitable 
for Cellpose training and inference.

NOTE: unfortunately, the Cellpose dependency collides with other dependencies in 
our project so we can't run the training in the same environment.
'''

from pathlib import Path
import argparse
import shutil

from AxonDeepSeg.ads_utils import imread, imwrite, get_imshape
from skimage import measure
import numpy as np

CELLPOSE_MASK_SUFFIX = '_seg-cellpose.png'


def find_all_images_and_masks(dir_path: Path, mask_suffix: str = '_seg-axonmyelin-manual.png') -> list[tuple[Path, Path]]:
    """
    Find all image and corresponding mask file paths in the given directory.
    
    Parameters:
    - dir_path: Path to the directory to search.
    - mask_suffix: Suffix used to identify mask files.
    
    Returns:
    - List of tuples containing (image_path, mask_path).
    """
    image_mask_pairs = []
    mask_files = list(dir_path.glob(f'*{mask_suffix}'))
    for mask_file in mask_files:
        image_file = mask_file.with_name(mask_file.name.replace(mask_suffix, '.png'))
        if image_file.exists():
            image_mask_pairs.append((image_file, mask_file))
        else: 
            # the image might actually be in TIFF format
            image_file_tiff = mask_file.with_name(mask_file.name.replace(mask_suffix, '.tif'))
            if image_file_tiff.exists():
                image_mask_pairs.append((image_file_tiff, mask_file))
            else:
                print(f'Warning: No corresponding image found for mask {mask_file}')
    return image_mask_pairs

def convert_axonmyelin_mask_to_cellpose(mask_path: Path, output_path: Path):
    """
    Convert an axon-myelin segmentation mask to a Cellpose-compatible mask.
    
    In the axon-myelin mask:
    - Background: 0
    - Myelin: 127
    - Axon: 255
    
    In the Cellpose mask:
    - Instance segmentation
    - Background: 0
    - Cell 1: 1 (axon and myelin combined)
    - ... and so on for each cell instance.
    
    Parameters:
    - mask_path: Path to the input axon-myelin mask.
    - output_path: Path to save the converted Cellpose mask.
    """
    mask = imread(str(mask_path))
    cellpose_mask = (mask > 0)  # Set axon and myelin to 1, background to 0
    cellpose_mask = measure.label(cellpose_mask, connectivity=1)  # Label connected components

    # Ensure the mask is cast to a supported data type
    cellpose_mask = cellpose_mask.astype(np.uint16)

    imwrite(str(output_path), cellpose_mask, use_16bit=True)

def preprocess_dataset(data_dir: Path, output_dir: Path):
    train_dir = data_dir / 'train'
    test_dir = data_dir / 'test'
    
    if not train_dir.exists() or not test_dir.exists():
        raise ValueError("The provided data_dir must contain 'train/' and 'test/' subdirectories.")
    
    output_train_dir = output_dir / 'train'
    output_test_dir = output_dir / 'test'
    for out_dir in [output_train_dir, output_test_dir]:
        out_dir.mkdir(parents=True, exist_ok=True)
    
    train_data = find_all_images_and_masks(train_dir)
    test_data = find_all_images_and_masks(test_dir)
    
    for data, output_dir in zip([train_data, test_data], [output_train_dir, output_test_dir]):
        for image_path, mask_path in data:
            output_image_path = output_dir / image_path.name
            output_mask_path = output_dir / (mask_path.stem.replace('_seg-axonmyelin-manual', CELLPOSE_MASK_SUFFIX))

            shutil.copy(image_path, output_image_path)
            convert_axonmyelin_mask_to_cellpose(mask_path, output_mask_path)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="Preprocess dataset for Cellpose")
    ap.add_argument('data_dir', type=str, help="Path to the dataset (split into train/ and test/ directories)")
    ap.add_argument('--output_dir', type=str, default=None, help="Path to save the preprocessed data")
    args = ap.parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else Path('.') / 'cellpose_preprocessed'
    
    preprocess_dataset(data_dir, output_dir)