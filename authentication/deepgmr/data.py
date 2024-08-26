'''
Copyright (c) 2020 NVIDIA
Author: Wentao Yuan
'''

import h5py
import numpy as np
import os
import torch
from scipy.spatial import cKDTree
from torch.utils.data import Dataset
import open3d as o3d


def knn_idx(pts, k):
    kdt = cKDTree(pts) 
    _, idx = kdt.query(pts, k=k+1)
    return idx[:, 1:]


def get_rri(pts, k):
    # pts: N x 3, original points
    # q: N x K x 3, nearest neighbors
    q = pts[knn_idx(pts, k)]
    p = np.repeat(pts[:, None], k, axis=1)
    # rp, rq: N x K x 1, norms
    rp = np.linalg.norm(p, axis=-1, keepdims=True)
    rq = np.linalg.norm(q, axis=-1, keepdims=True)
    pn = p / rp
    qn = q / rq
    dot = np.sum(pn * qn, -1, keepdims=True)
    # theta: N x K x 1, angles
    theta = np.arccos(np.clip(dot, -1, 1))
    T_q = q - dot * p
    sin_psi = np.sum(np.cross(T_q[:, None], T_q[:, :, None]) * pn[:, None], -1)
    cos_psi = np.sum(T_q[:, None] * T_q[:, :, None], -1)
    psi = np.arctan2(sin_psi, cos_psi) % (2*np.pi)
    idx = np.argpartition(psi, 1)[:, :, 1:2]
    # phi: N x K x 1, projection angles
    phi = np.take_along_axis(psi, idx, axis=-1)
    feat = np.concatenate([rp, rq, theta, phi], axis=-1)
    return feat.reshape(-1, k * 4)


def jitter_pcd(pcd, sigma=0.01, clip=0.05):
    pcd += np.clip(sigma * np.random.randn(*pcd.shape), -1 * clip, clip)
    return pcd


def random_pose(max_angle, max_trans):
    R = random_rotation(max_angle)
    t = random_translation(max_trans)
    return np.concatenate([np.concatenate([R, t], 1), [[0, 0, 0, 1]]], 0)


def random_rotation(max_angle):
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    angle = np.random.rand() * max_angle
    A = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * A + (1 - np.cos(angle)) * np.dot(A, A)
    return R


def random_translation(max_dist):
    t = np.random.randn(3)
    t /= np.linalg.norm(t)
    t *= np.random.rand() * max_dist
    return np.expand_dims(t, 1)

def generate_random_transform():
    # Generate a random rotation matrix
    R = o3d.geometry.get_rotation_matrix_from_xyz(np.random.rand(3) * 2 * np.pi)
    # Generate a random translation vector
    t = np.random.rand(3) * 0.1  # small random translation
    # Combine into a transformation matrix
    transform = np.eye(4)
    transform[:3, :3] = R
    transform[:3, 3] = t
    return transform


class TestData(Dataset):
    def __init__(self, path, args):
        super(TestData, self).__init__()
        with h5py.File(path, 'r') as f:
            self.source = f['source'][...]
            self.target = f['target'][...]
            self.transform = f['transform'][...]
        self.n_points = args.n_points
        self.use_rri = args.use_rri
        self.k = args.k if self.use_rri else None
        self.get_rri = get_rri

    def __getitem__(self, index):
        pcd1 = self.source[index][:self.n_points]
        pcd2 = self.target[index][:self.n_points]
        if self.use_rri:
            pcd1 = np.concatenate([pcd1, self.get_rri(pcd1 - pcd1.mean(axis=0), self.k)], axis=1)
            pcd2 = np.concatenate([pcd2, self.get_rri(pcd2 - pcd2.mean(axis=0), self.k)], axis=1)
        transform = self.transform[index]
        return pcd1.astype('float32'), pcd2.astype('float32'), transform.astype('float32')

    def __len__(self):
        return self.transform.shape[0]
    
class CustomData(Dataset): 
    def __init__(self, source_path, target_path, args):
        super(CustomData, self).__init__()
        self.source = [np.asarray(o3d.io.read_point_cloud(source_path).points)]
        self.target = [np.asarray(o3d.io.read_point_cloud(target_path).points)]
        self.transform = [np.eye(4)]
        self.n_points = args.n_points
        self.use_rri = args.use_rri
        self.k = args.k if self.use_rri else None
        self.get_rri = get_rri

    def __getitem__(self, index):
        pcd1 = self.source[index]
        pcd2 = self.target[index]
        if self.use_rri:
            pcd1 = np.concatenate([pcd1, self.get_rri(pcd1 - pcd1.mean(axis=0), self.k)], axis=1)
            pcd2 = np.concatenate([pcd2, self.get_rri(pcd2 - pcd2.mean(axis=0), self.k)], axis=1)
        transform = self.transform[index]
        return pcd1.astype('float32'), pcd2.astype('float32'), transform.astype('float32')

    def __len__(self):
        return min(len(self.source), len(self.target))
    

class TrainData(Dataset):
    def __init__(self, path, args):
        super(TrainData, self).__init__()
        with h5py.File(path, 'r') as f:
            self.points = f['points'][...]
        self.n_points = args.n_points
        self.max_angle = args.max_angle / 180 * np.pi
        self.max_trans = args.max_trans
        self.noisy = not args.clean
        self.use_rri = args.use_rri
        self.k = args.k if self.use_rri else None
        self.get_rri = get_rri

    def __getitem__(self, index):
        pcd1 = self.points[index][:self.n_points]
        pcd2 = self.points[index][:self.n_points]
        transform = random_pose(self.max_angle, self.max_trans / 2)
        pose1 = random_pose(np.pi, self.max_trans)
        pose2 = transform @ pose1
        pcd1 = pcd1 @ pose1[:3, :3].T + pose1[:3, 3]
        pcd2 = pcd2 @ pose2[:3, :3].T + pose2[:3, 3]
        if self.noisy:
            pcd1 = jitter_pcd(pcd1)
            pcd2 = jitter_pcd(pcd2)
        if self.use_rri:
            pcd1 = np.concatenate([pcd1, self.get_rri(pcd1 - pcd1.mean(axis=0), self.k)], axis=1)
            pcd2 = np.concatenate([pcd2, self.get_rri(pcd2 - pcd2.mean(axis=0), self.k)], axis=1)
        return pcd1.astype('float32'), pcd2.astype('float32'), transform.astype('float32')

    def __len__(self):
        return self.points.shape[0]

