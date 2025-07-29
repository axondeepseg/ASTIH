from pathlib import Path
from monai.metrics import (
    DiceMetric, MeanIoU, 
    HausdorffDistanceMetric, SurfaceDistanceMetric,
    PanopticQualityMetric, AveragePrecisionMetric,
)

import torch
import cv2
import numpy as np
import pandas as pd
import warnings
from skimage.measure import label

from get_data import DATASETS

def compute_map(pred_mask, gt_mask):
    """
    Compute mAP for binary segmentation masks containing multiple objects
    
    Args:
        pred_mask: Binary prediction mask
        gt_mask: Binary ground truth mask
    
    Returns:
        mAP score
    """
    # 1. Convert binary masks to instance segmentations
    pred_instances = label(pred_mask.squeeze().numpy())
    gt_instances = label(gt_mask.squeeze().numpy())
    
    # 2. Use MONAI's AveragePrecisionMetric
    ap_metric = AveragePrecisionMetric(
        include_background=False,
        reduction="mean",
        get_not_nans=True
    )
    
    # Convert instances to one-hot encoding
    num_pred_instances = pred_instances.max()
    num_gt_instances = gt_instances.max()
    
    # Create one-hot tensors
    pred_onehot = torch.zeros(1, num_pred_instances + 1, *pred_instances.shape)
    gt_onehot = torch.zeros(1, num_gt_instances + 1, *gt_instances.shape)
    
    # Fill one-hot tensors
    for i in range(1, num_pred_instances + 1):
        pred_onehot[0, i][pred_instances == i] = 1
    
    for i in range(1, num_gt_instances + 1):
        gt_onehot[0, i][gt_instances == i] = 1
    
    # Compute AP
    ap_metric(pred_onehot[:, 1:], gt_onehot[:, 1:])  # Exclude background
    map_score = ap_metric.aggregate().item()
    
    return map_score

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
    if isinstance(metric, PanopticQualityMetric):
        metric.reset()
        metric(pred, gt)
        value = metric.aggregate().mean()
    else:
        value = metric(pred, gt)
    return value.item()

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
    metrics = [
        DiceMetric(), 
        MeanIoU(), 
        HausdorffDistanceMetric(), 
        SurfaceDistanceMetric(),
        AveragePrecisionMetric(),
    ]
    metric_names = [metric.__class__.__name__ for metric in metrics]
    columns = ['dataset', 'image', 'class'] + metric_names
    df = pd.DataFrame(columns=columns)
    
    data_splits_path = Path("data/splits")
    assert data_splits_path.exists(), "Data splits directory does not exist. Please run get_data.py with --make-splits arg first."
    
    for dset in DATASETS:
        print(f"Evaluation for {dset.name} dataset...")

        testset_path = data_splits_path / dset.name / 'test'
        gts = list(testset_path.glob("*_seg-axonmyelin-manual.png"))
        for gt in gts:
            # Get the corresponding image
            img_fname = gt.name.replace("_seg-axonmyelin-manual.png", ".png")
            ax_pred_fname = gt.name.replace("_seg-axonmyelin-manual.png", "_seg-axon.png")
            my_pred_fname = gt.name.replace("_seg-axonmyelin-manual.png", "_seg-myelin.png")
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

            # compute metrics
            for class_name, pred, gt in classwise_pairs:
                row = {
                    'dataset': dset.name,
                    'image': img_fname,
                    'class': class_name
                }
                for metric in metrics:
                    if isinstance(metric, PanopticQualityMetric):
                        # For PanopticQualityMetric, we need to convert the masks to labels
                        pred_labels = torch.from_numpy(label(pred.numpy().astype(np.uint16))).float()
                        gt_labels = torch.from_numpy(label(gt.numpy().astype(np.uint16))).float()
                        pred_labels = torch.stack([pred_labels, pred], dim=1)
                        gt_labels = torch.stack([gt_labels, gt], dim=1)
                        value = compute_metrics(pred_labels, gt_labels, metric)
                    elif isinstance(metric, AveragePrecisionMetric):
                        # For AveragePrecisionMetric, we need to compute mAP
                        value = compute_map(pred, gt)
                    else:
                        value = compute_metrics([pred], [gt], metric)
                    row[metric.__class__.__name__] = value
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    # Export the dataframe to a CSV file
    df.to_csv('metrics.csv', index=False)
    print("Metrics computed and saved to metrics.csv")


if __name__ == "__main__":
    main()