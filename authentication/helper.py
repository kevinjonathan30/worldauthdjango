import open3d as o3d
import numpy as np
import os

def write_point_cloud(filename, fileContent, format='ply', write_ascii=False, compressed=True):
    points_np = np.loadtxt(fileContent)
    ply = o3d.geometry.PointCloud()
    ply.points = o3d.utility.Vector3dVector(points_np)

    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)

    o3d.io.write_point_cloud(filename=filename, pointcloud=ply, format=format, write_ascii=write_ascii, compressed=compressed)