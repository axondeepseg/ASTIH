from pathlib import Path
from monai.metrics import (
    DiceMetric, MeanIoU, 
    HausdorffDistanceMetric, SurfaceDistanceMetric,
    PanopticQualityMetric
)
from pycocotools.mask import encode as mask_encode
from pycocotools.mask import iou as mask_iou
from AxonDeepSeg.morphometrics.compute_morphometrics import get_watershed_segmentation
import torch
import cv2
import numpy as np
import pandas as pd
import warnings
from skimage.measure import label, regionprops

from get_data import DATASETS


def get_instance_segmentation(im_axon, im_myelin):    
    """
    Get instance segmentation from axon and myelin masks using a marker-controlled
    watershed algorithm to separate adjacent fibers.
    
    Args:
        im_axon: Axon mask
        im_myelin: Myelin mask
    
    Returns:
        Instance segmentation mask
    """
    axon_labels = label(im_axon.numpy())
    axon_objects = regionprops(axon_labels)
    index_centroids = (
        [int(property.centroid[0]) for property in axon_objects],
        [int(property.centroid[1]) for property in axon_objects]
    )
    instance_seg = get_watershed_segmentation(im_axon.numpy(), im_myelin.numpy(), index_centroids)
    return instance_seg


def compute_map(pred_ax, pred_my, gt_ax, gt_my):
    """
    Compute mAP for binary segmentation masks containing multiple objects
    
    Args:
        pred_mask: Binary prediction mask
        gt_mask: Binary ground truth mask
    
    Returns:
        mAP score
    """
    # Convert binary masks to instance segmentations
    pred_instances = get_instance_segmentation(pred_ax, pred_my)
    gt_instances = get_instance_segmentation(gt_ax, gt_my)
    
    # Convert to COCO format
    pred_masks = []
    for i in range(1, np.max(pred_instances) + 1):
        mask = (pred_instances == i).astype(np.uint8)
        pred_masks.append(mask_encode(np.asfortranarray(mask)))
    
    gt_masks = []
    for i in range(1, np.max(gt_instances) + 1):
        mask = (gt_instances == i).astype(np.uint8)
        gt_masks.append(mask_encode(np.asfortranarray(mask)))
    
    # Compute IoUs between all masks
    ious = np.zeros((len(pred_masks), len(gt_masks)))
    for i, p_mask in enumerate(pred_masks):
        for j, g_mask in enumerate(gt_masks):
            ious[i, j] = mask_iou([p_mask], [g_mask], [0])[0][0]
    
    # Calculate AP at IoU threshold of 0.5
    matched_gt = set()
    tp = 0
    
    for i in range(len(pred_masks)):
        best_iou = 0
        best_gt = -1
        
        for j in range(len(gt_masks)):
            if ious[i, j] > best_iou:
                best_iou = ious[i, j]
                best_gt = j
        
        if best_iou > 0.5 and best_gt not in matched_gt:
            tp += 1
            matched_gt.add(best_gt)
    
    # Calculate precision and recall
    precision = tp / len(pred_masks) if len(pred_masks) > 0 else 0
    recall = tp / len(gt_masks) if len(gt_masks) > 0 else 0
    
    # Simple AP approximation
    ap = precision * recall
    
    return ap

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
                    else:
                        value = compute_metrics([pred], [gt], metric)
                    row[metric.__class__.__name__] = value
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            # mAP is computed once per image, not per class
            map_value = compute_map(ax_pred, my_pred, gt_ax, gt_my)
            row = {
                'dataset': dset.name,
                'image': img_fname,
                'class': 'axon',
            }
            row['MeanAveragePrecision'] = map_value
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    # Export the dataframe to a CSV file
    df.to_csv('metrics.csv', index=False)
    print("Metrics computed and saved to metrics.csv")


if __name__ == "__main__":
    main()