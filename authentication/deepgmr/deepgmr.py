'''
Copyright (c) 2020 NVIDIA
Author: Wentao Yuan
'''

import os
import numpy as np
import torch
from django.conf import settings

from .model import DeepGMR
from .data import get_rri

def preprocess_point_cloud(pcd, use_rri, k=None):
    """Preprocess the point cloud data, optionally adding RRI features."""
    if use_rri:
        pcd = np.concatenate([pcd, get_rri(pcd - pcd.mean(axis=0), k)], axis=1)
    return pcd.astype('float32')

def evaluate(model, pts1, pts2, T_gt, rmse_thresh, save_results=False, results_dir=None, threshold=0.7):
    model.eval()
    # log_fmt = 'Inference time {:.3f}, Preprocessing time {:.3f}, Loss {:.4f}, ' + \
    #           'Rotation error {:.2f}, Translation error {:.4f}, RMSE {:.4f}, Recall {:.3f}'
    # inference_time = 0
    # preprocess_time = 0
    # losses = 0
    # r_errs = 0
    # t_errs = 0
    # rmses = 0
    # n_correct = 0

    # start = time()
    if torch.cuda.is_available():
        pts1 = torch.tensor(pts1).cuda()
        pts2 = torch.tensor(pts2).cuda()
        T_gt = torch.tensor(T_gt).cuda()
    else:
        pts1 = torch.tensor(pts1).cpu()
        pts2 = torch.tensor(pts2).cpu()
        T_gt = torch.tensor(T_gt).cpu()
    
    # Add batch dimension (since we're not using a DataLoader)
    pts1 = pts1.unsqueeze(0)
    pts2 = pts2.unsqueeze(0)
    T_gt = T_gt.unsqueeze(0)
    
    # preprocess_time += time() - start

    # start = time()
    with torch.no_grad():
        loss, r_err, t_err, rmse = model(pts1, pts2, T_gt)
        # inference_time += time() - start

    # losses += loss.item()
    # r_errs += r_err.sum().item()
    # t_errs += t_err.sum().item()
    # rmses += rmse.sum().item()
    # n_correct += (rmse < rmse_thresh).sum().item()

    # log_str = log_fmt.format(
    #     inference_time, preprocess_time, losses,
    #     r_errs, t_errs, rmses, n_correct
    # )

    if save_results:
        os.makedirs(results_dir, exist_ok=True)
        np.save(os.path.join(results_dir, 'rotations.npy'), model.T_12[:, :3, :3].cpu().numpy())
        np.save(os.path.join(results_dir, 'translations.npy'), model.T_12[:, :3, 3].cpu().numpy())
        np.save(os.path.join(results_dir, 'rotations_gt.npy'), T_gt[:, :3, :3].cpu().numpy())
        np.save(os.path.join(results_dir, 'translations_gt.npy'), T_gt[:, :3, 3].cpu().numpy())

    return model.evaluate_registration(0, threshold)

def options():
    class Args:
        data_file = os.path.join(settings.BASE_DIR, "authentication", "deepgmr", "data", "test", "modelnet_noisy.h5")  # Specify the data file path if needed
        results_dir = "results/"  # Directory to save results
        checkpoint = os.path.join(settings.BASE_DIR, "authentication", "deepgmr", "models", "modelnet_noisy.pth")  # Path to the model checkpoint
        save_results = False  # Whether to save the results or not
        rmse_thresh = 0.2  # RMSE threshold
        n_points = 1024  # Number of points in the dataset
        batch_size = 32  # Batch size for the DataLoader
        d_model = 1024  # Model dimensionality
        n_clusters = 16  # Number of clusters
        use_rri = False  # Whether to use RRI
        use_tnet = False  # Whether to use T-Net
        k = 20  # Parameter for model (e.g., k-NN)

    return Args()

def execute_deepgmr(pcd_source, pcd_target, threshold=0.7):
    args = options()

    torch.set_num_threads(1)
    model = DeepGMR(args).cpu()
    
    # Assume pcd_source and pcd_target are already open3d.geometry.PointCloud objects
    pcd_source = pcd_source.voxel_down_sample(0.1)
    source_pcd = np.asarray(pcd_source.points)

    pcd_target = pcd_target.voxel_down_sample(0.1)
    target_pcd = np.asarray(pcd_target.points)

    source_pcd = preprocess_point_cloud(source_pcd, args.use_rri, args.k)
    target_pcd = preprocess_point_cloud(target_pcd, args.use_rri, args.k)

    T_gt = np.eye(4).astype('float32')  # Assuming identity as the ground truth transformation

    # Load model weights from the checkpoint
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))

    # Evaluate the model with the specified parameters
    return evaluate(model, source_pcd, target_pcd, T_gt, args.rmse_thresh, args.save_results, args.results_dir, threshold=threshold)
