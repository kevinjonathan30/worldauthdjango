"""
Demo the feature-metric registration algorithm
Creator: Xiaoshui Huang
Date: 2021-04-13
"""
import os
import sys
import copy
import open3d
import torch
import torch.utils.data
import numpy as np
from .model import PointNet, Decoder, SolveRegistration
import open3d as o3d
from django.conf import settings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

# visualize the point clouds
def evaluate_registration(source, target, transformation, threshold=0.7):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.transform(transformation)
    evaluation = o3d.pipelines.registration.evaluate_registration(source_temp, target_temp, threshold)
    return evaluation.fitness


class Demo:
    def __init__(self):
        self.dim_k = 1024
        self.max_iter = 10  # max iteration time for IC algorithm
        self._loss_type = 1  # see. self.compute_loss()

    def create_model(self):
        # Encoder network: extract feature for every point. Nx1024
        ptnet = PointNet(dim_k=self.dim_k)
        # Decoder network: decode the feature into points, not used during the evaluation
        decoder = Decoder()
        # feature-metric ergistration (fmr) algorithm: estimate the transformation T
        fmr_solver = SolveRegistration(ptnet, decoder, isTest=True)
        return fmr_solver

    def evaluate(self, solver, p0, p1, device):
        solver.eval()
        with torch.no_grad():
            p0 = torch.tensor(p0,dtype=torch.float).to(device)  # target (1, N, 3)
            p1 = torch.tensor(p1,dtype=torch.float).to(device)  # source (1, M, 3)
            solver.estimate_t(p0, p1, self.max_iter)

            est_g = solver.g  # (1, 4, 4)
            g_hat = est_g.cpu().contiguous().view(4, 4)  # --> [1, 4, 4]

            return g_hat

def execute_fmr(source, target, threshold=0.7):
    p0_src = open3d.io.read_point_cloud(target)
    downpcd0 = p0_src.voxel_down_sample(voxel_size=0.1)
    p0 = np.asarray(downpcd0.points)
    p0 = np.expand_dims(p0,0)

    '''
    # generate random rotation sample
    trans = transforms.RandomTransformSE3(0.8, True)
    p0_src_tensor = torch.tensor((np.asarray(p0_src.points)),dtype=torch.float)
    p0_tensor_transformed = trans(p0_src_tensor)
    p1_src = p0_tensor_transformed.cpu().numpy()
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(p1_src)
    open3d.io.write_point_cloud(path1, pcd)
    '''

    p1 = open3d.io.read_point_cloud(source)
    downpcd1 = p1.voxel_down_sample(voxel_size=0.1)
    p1 = np.asarray(downpcd1.points)
    p1 = np.expand_dims(p1, 0)
    
    torch.set_num_threads(1)
    fmr = Demo()
    model = fmr.create_model()
    pretrained_path = os.path.join(settings.BASE_DIR, "authentication", "fmr", "result", "fmr_model_7scene.pth")
    model.load_state_dict(torch.load(pretrained_path, map_location='cpu'))

    device = "cpu"
    model.to(device)
    T_est = fmr.evaluate(model, p0, p1, device)

    return evaluate_registration(downpcd1, downpcd0, T_est, threshold=threshold)
