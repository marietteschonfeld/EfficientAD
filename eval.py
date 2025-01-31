"""Utility function that computes a PRO curve, given pairs of anomaly and ground
truth maps.

The PRO curve can also be integrated up to a constant integration limit.
"""
from scipy.ndimage.measurements import label
import torch
import torchvision.transforms.v2 as transforms
import numpy as np
import os, json

from mvtec_ad_evaluation.pro_curve_util import compute_pro
from mvtec_ad_evaluation.roc_curve_util import compute_classification_roc
from mvtec_ad_evaluation.generic_util import trapezoid
from scipy.ndimage.measurements import label

# from kornia.contrib import connected_components

class eval():

    @staticmethod
    def calculate_scores(anomaly_maps, gt_maps):
        anomaly_maps, gt_maps = eval.prepare_maps(anomaly_maps, gt_maps)
        integration_limit=0.3

        # Compute the PRO curve.
        pro_curve = compute_pro(
            anomaly_maps=anomaly_maps,
            ground_truth_maps=gt_maps)

        # Compute the area under the PRO curve.
        au_pro = trapezoid(
            pro_curve[0], pro_curve[1], x_max=integration_limit)
        au_pro /= integration_limit

        # Derive binary labels for each input image:
        # (0 = anomaly free, 1 = anomalous).
        binary_labels = [int(np.any(x > 0)) for x in gt_maps]

        # score = torcheval.metrics.BinaryAUROC()
        # score.update(torch.from_numpy(np.concatenate([anomaly_map.flatten() for anomaly_map in anomaly_maps])).to(torch.device("mps")),
        #              torch.from_numpy(np.concatenate([gt_map.flatten() for gt_map in gt_maps])).to(torch.device("mps")))
        # pixel_auroc = score.compute().item()

        del gt_maps

        # Compute the classification ROC curve.
        roc_curve = compute_classification_roc(
            anomaly_maps=anomaly_maps,
            scoring_function=np.max,
            ground_truth_labels=binary_labels)

        # Compute the area under the classification ROC curve.
        au_roc = trapezoid(roc_curve[0], roc_curve[1])
        
        # Return the evaluation metrics.
        results = {
            "pixel_au_pro": au_pro,
            "image_au_roc": au_roc
        }
        return results
    

    @staticmethod
    def prepare_maps(anomaly_maps, gt_maps):
        anomaly_maps = [transforms.functional.resize(anomaly_map.cpu(), size=gt_maps[ind].shape[-2:]) for ind, anomaly_map in enumerate(anomaly_maps)]
        anomaly_maps = [anomaly_map.squeeze().numpy() for anomaly_map in anomaly_maps]
        gt_maps = [gt_map.squeeze().numpy() for gt_map in gt_maps]
        return anomaly_maps, gt_maps
    
