import open3d as o3d
import os
import torch
import numpy as np
from .model import PointLK, PointNet_features
from django.conf import settings
import logging

logger = logging.getLogger('authentication')

def load_ply_file(filepath):
    pcd = o3d.io.read_point_cloud(filepath)
    return np.asarray(pcd.points)

def evaluate_registration(template, source, est_T):
    template, source, est_T = template[0], source[0], est_T[0]
    transformed_source = np.matmul(est_T[0:3, 0:3], source.T).T + est_T[0:3, 3]

    template_ = o3d.geometry.PointCloud()
    source_ = o3d.geometry.PointCloud()
    transformed_source_ = o3d.geometry.PointCloud()

    template_.points = o3d.utility.Vector3dVector(template)
    source_.points = o3d.utility.Vector3dVector(source)
    transformed_source_.points = o3d.utility.Vector3dVector(transformed_source)
    
    evaluation = o3d.pipelines.registration.evaluate_registration(template_, transformed_source_, 0.7)
    logger.info(evaluation)
    return evaluation.fitness

def read_data(source, target):
    template = load_ply_file(target)
    source = load_ply_file(source)
    return torch.tensor(template, dtype=torch.float32), torch.tensor(source, dtype=torch.float32)

def test(args, model, data):
    model.eval()
    template, source = data

    template = template.view(1, -1, 3).to(args.device)
    source = source.view(1, -1, 3).to(args.device)

    result = model(template, source)
    est_T = result['est_T']
    r = result['r']

    logger.info(est_T)
    logger.info(r)
    return evaluate_registration(template.detach().cpu().numpy(), source.detach().cpu().numpy(), est_T.detach().cpu().numpy())

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

def execute_pointnetlk(source, target):
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
    return test(args, model, data)
