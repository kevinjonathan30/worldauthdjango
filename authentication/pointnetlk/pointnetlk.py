import open3d as o3d
import os
import torch
import numpy as np
from .model import PointLK, PointNet_features
from django.conf import settings

def load_ply_file(filepath):
    pcd = o3d.io.read_point_cloud(filepath)
    pcd = pcd.voxel_down_sample(0.1)
    return np.asarray(pcd.points)

def evaluate_registration(target, source, est_T, threshold=0.7):
    target, source, est_T = target[0], source[0], est_T[0]
    transformed_source = np.matmul(est_T[0:3, 0:3], source.T).T + est_T[0:3, 3]

    target_ = o3d.geometry.PointCloud()
    source_ = o3d.geometry.PointCloud()
    transformed_source_ = o3d.geometry.PointCloud()

    target_.points = o3d.utility.Vector3dVector(target)
    source_.points = o3d.utility.Vector3dVector(source)
    transformed_source_.points = o3d.utility.Vector3dVector(transformed_source)
    
    evaluation = o3d.pipelines.registration.evaluate_registration(transformed_source_, target_, threshold)
    print(evaluation)
    return evaluation.fitness

def read_data(source_ply, target_ply):
    source = load_ply_file(source_ply)
    target = load_ply_file(target_ply)
    return torch.tensor(source, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

def test(args, model, data, threshold):
    model.eval()
    source, target = data

    target = target.view(1, -1, 3).to(args.device)
    source = source.view(1, -1, 3).to(args.device)

    result = model(target, source)
    est_T = result['est_T']
    r = result['r']

    print(est_T)
    print(r)
    return evaluate_registration(target.detach().cpu().numpy(), source.detach().cpu().numpy(), est_T.detach().cpu().numpy(), threshold)

def options():
    class Args:
        emb_dims = 1024
        symfn = 'max'
        max_iter = 10
        delta = 1.0e-2
        learn_delta = False
        pretrained = os.path.join(settings.BASE_DIR, "authentication", "pointnetlk", "pretrained", "best_model.t7")
        device = 'cpu'

    return Args()

def execute_pointnetlk(source, target, threshold=0.7):
    args = options()

    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        args.device = 'cpu'
    args.device = torch.device(args.device)

    # Create PointNetLK Model.
    ptnet = PointNet_features(emb_dims=args.emb_dims, symfn=args.symfn)
    model = PointLK(ptnet=ptnet)

    if args.pretrained:
        assert os.path.isfile(args.pretrained)
        model.load_state_dict(torch.load(args.pretrained, map_location='cpu'))
    model.to(args.device)

    data = read_data(source, target)
    return test(args, model, data, threshold)
