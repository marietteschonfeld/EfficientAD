"""Utility function that computes a PRO curve, given pairs of anomaly and ground
truth maps.

The PRO curve can also be integrated up to a constant integration limit.
"""
from scipy.ndimage.measurements import label
import torch
import torchvision.transforms.v2 as transforms
import numpy as np
import os, json
from torcheval.metrics import BinaryAUROC

from .mvtec_ad_evaluation.pro_curve_util import compute_pro
from .mvtec_ad_evaluation.roc_curve_util import compute_classification_roc
from .mvtec_ad_evaluation.generic_util import trapezoid
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
    
    def calculate_scores_torch(anomaly_maps, gt_maps, device=torch.device("cpu")):
        anomaly_maps, gt_maps = eval.prepare_maps_torch(anomaly_maps, gt_maps, device)
        integration_limit=0.3

        # Compute the PRO curve.
        pro_curve = eval.compute_pro_torch(
            anomaly_maps=anomaly_maps,
            ground_truth_maps=gt_maps,device=device)

        # Compute the area under the PRO curve.
        au_pro = torch.trapezoid(
            pro_curve[0], pro_curve[1])
        au_pro /= integration_limit
        au_pro = 1-au_pro

        # Derive binary labels for each input image:
        # (0 = anomaly free, 1 = anomalous).
        binary_labels = [int(torch.any(x > 0)) for x in gt_maps]

        # score = torcheval.metrics.BinaryAUROC()
        # score.update(torch.from_numpy(np.concatenate([anomaly_map.flatten() for anomaly_map in anomaly_maps])).to(torch.device("mps")),
        #              torch.from_numpy(np.concatenate([gt_map.flatten() for gt_map in gt_maps])).to(torch.device("mps")))
        # pixel_auroc = score.compute().item()

        del gt_maps

        map_labels = [torch.max(anomaly_map) for anomaly_map in anomaly_maps]

        metric = BinaryAUROC()
        metric.update(torch.tensor(map_labels,device=device), torch.tensor(binary_labels, device=device))

        au_roc = metric.compute().item()
        
        # Return the evaluation metrics.
        results = {
            "pixel_au_pro": au_pro.item(),
            "image_au_roc": au_roc
        }
        return results




    @staticmethod
    def prepare_maps(anomaly_maps, gt_maps):
        anomaly_maps = [transforms.functional.resize(anomaly_map.cpu(), size=gt_maps[ind].shape[-2:]) for ind, anomaly_map in enumerate(anomaly_maps)]
        anomaly_maps = [anomaly_map.squeeze().numpy() for anomaly_map in anomaly_maps]
        gt_maps = [gt_map.squeeze().numpy() for gt_map in gt_maps]
        return anomaly_maps, gt_maps
    
    @staticmethod
    def prepare_maps_torch(anomaly_maps, gt_maps, device):
        anomaly_maps = [torch.nn.functional.interpolate(anomaly_map.unsqueeze(0), size=gt_maps[ind].shape[-2:]) for ind, anomaly_map in enumerate(anomaly_maps)]
        anomaly_maps = [anomaly_map.squeeze().to(device) for anomaly_map in anomaly_maps]
        gt_maps = [gt_map.squeeze().to(device) for gt_map in gt_maps]
        return anomaly_maps, gt_maps
    
    @staticmethod
    def compute_pro_torch(anomaly_maps, ground_truth_maps,device=torch.device("cpu")):
        structure = torch.ones((3, 3), dtype=int)

        num_ok_pixels = 0
        num_gt_regions = 0


        fp_changes = []
        pro_changes = []

        for gt_map in ground_truth_maps:

            # Compute the connected components in the ground truth map.
            labeled, n_components = label(gt_map.cpu(), structure)
            labeled = torch.from_numpy(labeled).to(device)
            num_gt_regions += n_components

            # labeled = eval.connected_components_gpu(gt_map.unsqueeze(0).unsqueeze(0)).squeeze()
            # n_components = labeled.max()
            # num_gt_regions += n_components

            # Compute the mask that gives us all ok pixels.
            ok_mask = labeled == 0
            num_ok_pixels_in_map = torch.sum(ok_mask)
            num_ok_pixels += num_ok_pixels_in_map

            # Compute by how much the FPR changes when each anomaly score is
            # added to the set of positives.
            # fp_change needs to be normalized later when we know the final value
            # of num_ok_pixels -> right now it is only the change in the number of
            # false positives
            fp_change = torch.zeros_like(gt_map, dtype=torch.float,device=device)
            fp_change[ok_mask] = 1

            # Compute by how much the PRO changes when each anomaly score is
            # added to the set of positives.
            # pro_change needs to be normalized later when we know the final value
            # of num_gt_regions.
            pro_change = torch.zeros_like(gt_map, dtype=torch.float,device=device)
            for k in range(n_components):
                region_mask = labeled == (k + 1)
                region_size = torch.sum(region_mask)
                pro_change[region_mask] = 1. / region_size

            fp_changes.append(fp_change)
            pro_changes.append(pro_change)

        # Flatten the numpy arrays before sorting.
        anomaly_scores_flat = torch.concatenate([anomaly_map.ravel() for anomaly_map in anomaly_maps])
        fp_changes_flat = torch.concatenate([fp_change.ravel() for fp_change in fp_changes])
        pro_changes_flat = torch.concatenate([pro_change.ravel() for pro_change in pro_changes])

        # Sort all anomaly scores.
        # print(f"Sort {len(anomaly_scores_flat)} anomaly scores...")
        sort_idxs = torch.argsort(anomaly_scores_flat, descending=True)

        # Info: np.take(a, ind, out=a) followed by b=a instead of
        # b=a[ind] showed to be more memory efficient.
        anomaly_scores_flat = anomaly_scores_flat[sort_idxs]
        anomaly_scores_sorted = anomaly_scores_flat
        fp_changes_flat = fp_changes_flat[sort_idxs]
        fp_changes_sorted = fp_changes_flat
        pro_changes_flat = pro_changes_flat[sort_idxs]
        pro_changes_sorted = pro_changes_flat


        del sort_idxs

        # Get the (FPR, PRO) curve values.
        fp_changes_sorted = torch.cumsum(fp_changes_sorted, dim=0)
        # fp_changes_sorted = fp_changes_sorted.astype(torch.float, copy=False)
        fp_changes_sorted = torch.divide(fp_changes_sorted, num_ok_pixels)
        fprs = fp_changes_sorted

        pro_changes_sorted = torch.cumsum(pro_changes_sorted, dim=0)
        pro_changes_sorted = torch.divide(pro_changes_sorted, num_gt_regions)
        pros = pro_changes_sorted

        # Merge (FPR, PRO) points that occur together at the same threshold.
        # For those points, only the final (FPR, PRO) point should be kept.
        # That is because that point is the one that takes all changes
        # to the FPR and the PRO at the respective threshold into account.
        # -> keep_mask is True if the subsequent score is different from the
        # score at the respective position.
        # anomaly_scores_sorted = [7, 4, 4, 4, 3, 1, 1]
        # ->          keep_mask = [T, F, F, T, T, F]
        # print(anomaly_scores_sorted.shape)
        # print(torch.diff(anomaly_scores_sorted))
        # print(torch.diff(anomaly_scores_sorted)!=0)
        keep_mask = torch.cat((torch.diff(anomaly_scores_sorted) != 0, torch.tensor([True], device=device)))
        del anomaly_scores_sorted

        fprs = fprs[keep_mask]
        pros = pros[keep_mask]
        del keep_mask

        # To mitigate the adding up of numerical errors during the np.cumsum calls,
        # make sure that the curve ends at (1, 1) and does not contain values > 1.
        fprs = torch.clip(fprs, min=None, max=1.)
        pros = torch.clip(pros, min=None, max=1.)

        # Make the fprs and pros start at 0 and end at 1.
        zero = torch.tensor([0.],device=device)
        one = torch.tensor([1.],device=device)

        return torch.concatenate((zero, fprs, one)), torch.concatenate((zero, pros, one))
    
    # @staticmethod
    # def connected_components_gpu(image: torch.Tensor, num_iterations: int = 1000) -> torch.Tensor:
    #     """Perform connected component labeling on GPU and remap the labels from 0 to N.

    #     Args:
    #         image (torch.Tensor): Binary input image from which we want to extract connected components (Bx1xHxW)
    #         num_iterations (int): Number of iterations used in the connected component computation.

    #     Returns:
    #         Tensor: Components labeled from 0 to N.
    #     """
    #     components = connected_components(image, num_iterations=num_iterations)

    #     # # remap component values from 0 to N
    #     labels = components.unique()
    #     for new_label, old_label in enumerate(labels):
    #         components[components == old_label] = new_label

    #     return components.int()