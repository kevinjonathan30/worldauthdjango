import argparse
import numpy as np
import os
import torch
from time import time
from torch.utils.data import DataLoader
# from tqdm import tqdm
from django.conf import settings

from .data import TestData, CustomData
from .model import DeepGMR

def evaluate(model, loader, rmse_thresh, save_results=False, results_dir=None):
    model.eval()
    log_fmt = 'Test: inference time {:.3f}, preprocessing time {:.3f}, loss {:.4f}, ' + \
              'rotation error {:.2f}, translation error {:.4f}, RMSE {:.4f}, Recall {:.3f}'
    inference_time = 0
    preprocess_time = 0
    losses = 0
    r_errs = 0
    t_errs = 0
    rmses = 0
    n_correct = 0
    N = 0

    # Initialize lists regardless of saving results
    rotations = []
    translations = []
    rotations_gt = []
    translations_gt = []

    start = time()
    for step, (pts1, pts2, T_gt) in enumerate(loader):
        if torch.cuda.is_available():
            pts1 = pts1.cpu()
            pts2 = pts2.cpu()
            T_gt = T_gt.cpu()
        preprocess_time += time() - start
        N += pts1.shape[0]

        start = time()
        with torch.no_grad():
            loss, r_err, t_err, rmse = model(pts1, pts2, T_gt)
            inference_time += time() - start

        losses += loss.item()
        r_errs += r_err.sum().item()
        t_errs += t_err.sum().item()
        rmses += rmse.sum().item()
        n_correct += (rmse < rmse_thresh).sum().item()

        rotations.append(model.T_12[:, :3, :3].cpu().numpy())
        translations.append(model.T_12[:, :3, 3].cpu().numpy())
        rotations_gt.append(T_gt[:, :3, :3].cpu().numpy())
        translations_gt.append(T_gt[:, :3, 3].cpu().numpy())

        start = time()

    log_str = log_fmt.format(
        inference_time / N, preprocess_time / N, losses / len(loader),
        r_errs / N, t_errs / N, rmses / N, n_correct / N
    )
    print(log_str)

    if save_results:
        os.makedirs(results_dir, exist_ok=True)
        np.save(os.path.join(results_dir, 'rotations.npy'), np.concatenate(rotations, 0))
        np.save(os.path.join(results_dir, 'translations.npy'), np.concatenate(translations, 0))
        np.save(os.path.join(results_dir, 'rotations_gt.npy'), np.concatenate(rotations_gt, 0))
        np.save(os.path.join(results_dir, 'translations_gt.npy'), np.concatenate(translations_gt, 0))

    return model.evaluate_registration(0)

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
        use_rri = True  # Whether to use RRI
        use_tnet = False  # Whether to use T-Net
        k = 20  # Parameter for model (e.g., k-NN)

    return Args()

def execute_deepgmr(source, target):
    args = options()

    torch.set_num_threads(1)
    model = DeepGMR(args).cpu()
    
    test_data = CustomData(source, target, args)
    test_loader = DataLoader(test_data, args.batch_size)

    # Load model weights from the checkpoint
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))

    # Evaluate the model with the specified parameters
    return evaluate(model, test_loader, args.rmse_thresh, args.save_results, args.results_dir)
