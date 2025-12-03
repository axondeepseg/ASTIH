from pathlib import Path
from monai.metrics import DiceMetric, MeanIoU, compute_panoptic_quality
import torch
import cv2
import numpy as np
import pandas as pd
import warnings
from skimage import measure
from AxonDeepSeg.morphometrics.compute_morphometrics import get_watershed_segmentation
from stardist.matching import matching

from get_data import load_datasets


def compute_metrics(pred, gt, metric):
    """
    Compute the given metric for a single image
    Args:
        pred: the prediction image
        gt: the ground truth image
        metric: the metric to compute
    Returns:
        the computed metric
    """
    value = metric(pred, gt)
    return value.item()

def make_panoptic_input(instance_map):
    """
    Converts a (H, W) instance map into the (B, 2, H, W) format
    required by MONAI PanopticQualityMetric.
    """
    semantic_map = torch.tensor((instance_map > 0)).unsqueeze(dim=0).int()
    instance_map = torch.tensor(instance_map).unsqueeze(dim=0).int()
    panoptic_map = torch.cat([semantic_map, instance_map], dim=1)

    return panoptic_map

def compute_confusion(inst_pred, inst_gt):
    """
    Computes the confusion matrix (TP, FP, FN, sum of IoU)
    < IMPORTANT! >  Do NOT use this function! This is an extremly slow 
                    implementation, relying on MONAI's panoptic quality 
                    metric. Also, this algorithm diverges with images with 
                    1000+ axons. Instead, `stardist` has a much faster 
                    matching function based on sparse graph optimization
                    instead of dense matrix operations.
    Args:
        inst_pred:  instance segmentation prediction
        inst_gt:    instance segmentation ground-truth
    Returns:
        TP:         true positive count,
        FP:         false positive count,
        FN:         false negative count,
        sumIoU:     sum of IoU used to compute Panoptic Quality
    """
    y_pred_2ch = make_panoptic_input(inst_pred)
    y_true_2ch = make_panoptic_input(inst_gt)
    confusion = compute_panoptic_quality(
        pred=y_pred_2ch,
        gt=y_true_2ch,
        metric_name='rq',
        match_iou_threshold=0.5,
        output_confusion_matrix=True
    )

    return confusion

def apply_watershed(ax_mask, my_mask):
    axon_objects = measure.regionprops(measure.label(ax_mask))
    centroids = (
        [int(props.centroid[0]) for props in axon_objects],
        [int(props.centroid[1]) for props in axon_objects]
    )

    return get_watershed_segmentation(ax_mask, my_mask, centroids)


def extract_binary_masks(mask):
    '''
    This function will take as input an 8-bit image containing both the axon 
    class (value should be ~255) and the myelin class (value should be ~127).
    This function will also convert the numpy arrays read by opencv to Tensors.
    '''
    # axonmyelin masks should always have 3 unique values
    if len(np.unique(mask)) > 3:
        warnings.warn('WARNING: more than 3 unique values in the mask')
    myelin_mask = np.where(np.logical_and(mask > 100, mask < 200), 1, 0)
    myelin_mask = torch.from_numpy(myelin_mask).float()
    axon_mask = np.where(mask > 200, 1, 0)
    axon_mask = torch.from_numpy(axon_mask).float()
    return axon_mask, myelin_mask

def main():
    metrics = [DiceMetric(), MeanIoU()] #, PanopticQualityMetric(num_classes=1)]
    metric_names = [metric.__class__.__name__ for metric in metrics]
    columns = ['dataset', 'image', 'class'] + metric_names
    pixelwise_df = pd.DataFrame(columns=columns)
    columns_detection = ['dataset', 'image', 'TP', 'FP', 'FN', 'precision', 'recall', 'accuracy', 'f1']
    detection_df = pd.DataFrame(columns=columns_detection)
    
    data_splits_path = Path("data/splits")
    assert data_splits_path.exists(), "Data splits directory does not exist. Please run get_data.py with --make-splits arg first."
    
    datasets = load_datasets()
    for dset in datasets:
        print(f"Evaluation for {dset.name} dataset...")

        testset_path = data_splits_path / dset.name / 'test'
        gts = list(testset_path.glob("*_seg-axonmyelin-manual.png"))
        for gt in gts:
            # Get the corresponding image
            img_fname = gt.name.replace("_seg-axonmyelin-manual.png", ".png")
            potential_grayscale_img_fname = img_fname.replace('.png', '_grayscale.png')
            ax_pred_fname = gt.name.replace("_seg-axonmyelin-manual.png", "_seg-axon.png")
            my_pred_fname = gt.name.replace("_seg-axonmyelin-manual.png", "_seg-myelin.png")

            # check if image was converted to grayscale prior to inference
            if (testset_path / potential_grayscale_img_fname).exists():
                img_fname = potential_grayscale_img_fname
                ax_pred_fname = ax_pred_fname.replace('_seg', '_grayscale_seg')
                my_pred_fname = my_pred_fname.replace('_seg', '_grayscale_seg')
            assert (testset_path / ax_pred_fname).exists() and (testset_path / my_pred_fname).exists(), f"Predictions not found for {img_fname}"

            # load GT
            gt_im = cv2.imread(str(gt), cv2.IMREAD_GRAYSCALE)[None]
            gt_im = np.floor(gt_im / np.max(gt_im) * 255).astype(np.uint8)
            gt_ax, gt_my = extract_binary_masks(gt_im)

            # load predictions
            ax_pred = cv2.imread(str(testset_path / ax_pred_fname), cv2.IMREAD_GRAYSCALE)[None]
            ax_pred = np.floor(ax_pred / np.max(ax_pred) * 255).astype(np.uint8)
            ax_pred, _ = extract_binary_masks(ax_pred)
            my_pred = cv2.imread(str(testset_path / my_pred_fname), cv2.IMREAD_GRAYSCALE)[None]
            my_pred = np.floor(my_pred / np.max(my_pred) * 255).astype(np.uint8)
            my_pred, _ = extract_binary_masks(my_pred)

            classwise_pairs = [
                ('axon', ax_pred, gt_ax),
                ('myelin', my_pred, gt_my)
            ]

            # compute pixel-wise metrics
            for class_name, pred, gt in classwise_pairs:
                row = {
                    'dataset': dset.name,
                    'image': img_fname,
                    'class': class_name
                }
                for metric in metrics:
                    value = compute_metrics([pred], [gt], metric)
                    row[metric.__class__.__name__] = value
                pixelwise_df = pd.concat([pixelwise_df, pd.DataFrame([row])], ignore_index=True)

            # compute detection metrics
            inst_gt = apply_watershed(gt_ax, gt_my)
            inst_pred = apply_watershed(ax_pred, my_pred)
            stats = matching(inst_gt, inst_pred, thresh=0.5)
            detection_row = {
                'dataset':      dset.name,
                'image':        img_fname,
                'TP':           stats.tp,
                'FP':           stats.fp, 
                'FN':           stats.fn, 
                'precision':    stats.precision,
                'recall':       stats.recall,
                'accuracy':     stats.accuracy,
                'f1':           stats.f1
            }
            print(f'detection metrics: {detection_row}')
            detection_df = pd.concat([detection_df, pd.DataFrame([detection_row])], ignore_index=True)

    # Export the dataframe to a CSV file
    pixelwise_df.to_csv('metrics.csv', index=False)
    print("Metrics computed and saved to metrics.csv")
    detection_df.to_csv('det_metrics.csv', index=False)
    print("Detection metrics computed and saved to det_metrics.csv")


if __name__ == "__main__":
    main()
