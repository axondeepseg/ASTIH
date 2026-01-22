from pathlib import Path
from AxonDeepSeg.ads_utils import imread, get_imshape
from get_data import load_datasets


def compute_dataset_statistics(datapath: Path):
    # the total nb of images corresponds to the length of the samples.tsv file
    sample_file_path = datapath / 'samples.tsv'
    with open(sample_file_path, 'r') as f:
        lines = f.readlines()
    total_images = len(lines) - 1  # subtract header line
    print(f'\tTotal nb of images: {total_images}')

    # to find the total nb of labelled imgs, we look for files with the suffix '_seg-axon-manual'
    labelled_imgs = list((datapath / 'derivatives').rglob('*_seg-axonmyelin-manual.png'))
    total_labelled = len(labelled_imgs)
    print(f'\tTotal nb of labelled images: {total_labelled}')

    # for the avg image size, we compute the mean width and height across all images
    total_width = 0
    total_height = 0
    subject_dirs = datapath.glob('sub-*')
    for subject_dir in subject_dirs:
        img_files = list((subject_dir / 'micr').glob('*.png')) + list((subject_dir / 'micr').glob('*.tif'))
        for img_file in img_files:
            shape = get_imshape(str(img_file))
            total_height += shape[0]
            total_width += shape[1]
    avg_width = total_width / total_images
    avg_height = total_height / total_images
    print(f'\tAverage image size: {avg_width:.2f} x {avg_height:.2f} pixels')

    # now, for the avg foreground-background ratio, we need to look at the labelled images
    total_nb_pixels = 0
    total_fg_pixels = 0
    for labelled_img in labelled_imgs:
        mask = imread(str(labelled_img))
        total_nb_pixels += mask.size
        total_fg_pixels += (mask > 0).sum()

        other_suffixes = [
            '_seg-uaxon-manual',
            '_seg-process-manual',
            '_seg-nuclei-manual'
        ]
        potential_masks = [str(labelled_img).replace('_seg-axonmyelin-manual', suffix) for suffix in other_suffixes]
        for mask_path in potential_masks:
            if Path(mask_path).exists():
                other_mask = imread(mask_path)
                total_fg_pixels += (other_mask > 0).sum()

    avg_fg_bg_ratio = total_fg_pixels / (total_nb_pixels - total_fg_pixels) if (total_nb_pixels - total_fg_pixels) > 0 else float('inf')
    print(f'\tAverage foreground-background ratio: {avg_fg_bg_ratio:.4f}')


def main():
    datasets = load_datasets()
    for dset in datasets:
        id = dset.dandi_id
        print(f"Computing statistics for {dset.name} dataset (DANDI ID: {id})...")
        datapath = Path(__file__) .parent.parent / 'data' / id
        if not datapath.exists():
            print(f"Data for {dset.name} not found at {datapath}. Skipping...")
            continue
        compute_dataset_statistics(datapath)

if __name__ == "__main__":
    main()