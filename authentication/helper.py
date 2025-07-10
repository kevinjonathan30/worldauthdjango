import os

import open3d as o3d
import numpy as np

from django.conf import settings
import shutil

def write_point_cloud(filename, fileContent, format='ply', write_ascii=False, compressed=True):
    points_np = np.loadtxt(fileContent)
    ply = o3d.geometry.PointCloud()
    ply.points = o3d.utility.Vector3dVector(points_np)

    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)

    o3d.io.write_point_cloud(filename=filename, pointcloud=ply, format=format, write_ascii=write_ascii, compressed=compressed)

def delete_parent_directory(authorization):
    dir_path = os.path.join(settings.MEDIA_ROOT, authorization)
    remaining_files = [
        f for f in os.listdir(dir_path) if not f.startswith('.')
    ]
    if not remaining_files:
        shutil.rmtree(dir_path, ignore_errors=True)