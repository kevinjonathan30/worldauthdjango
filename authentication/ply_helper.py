import open3d as o3d
import numpy as np
from django.core.files.base import ContentFile
from .storage import OverwriteStorage
from django.core.files import File

def compute_curvature(pcd, k=30):
    # Compute the curvature of each point in the point cloud
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    curvatures = []

    for i in range(len(pcd.points)):
        [_, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], k)
        if len(idx) < k:
            curvatures.append(0)
            continue

        neighbors = np.asarray(pcd.points)[idx, :]
        covariance_matrix = np.cov(neighbors.T)
        eigenvalues, _ = np.linalg.eig(covariance_matrix)
        curvature = np.min(eigenvalues) / np.sum(eigenvalues)
        curvatures.append(curvature)

    return np.array(curvatures)

def convert_ply_pcd(pcd, fileName):
    # Estimate normals if not provided
    if not pcd.has_normals():
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(k=30)

    # Compute curvature
    curvatures = compute_curvature(pcd)

    # Add curvature to the point cloud
    pcd_curvature = np.zeros((len(pcd.points), 4))
    pcd_curvature[:, :3] = np.asarray(pcd.points)
    pcd_curvature[:, 3] = curvatures

    # Create new PointCloud object to include curvature
    pcd_with_curvature = o3d.geometry.PointCloud()
    pcd_with_curvature.points = o3d.utility.Vector3dVector(pcd_curvature[:, :3])
    pcd_with_curvature.normals = pcd.normals
    pcd_with_curvature.colors = pcd.colors
    
    # Combine point, normal, and curvature data for saving
    pcd_array = np.hstack((np.asarray(pcd_with_curvature.points),
                           np.asarray(pcd_with_curvature.normals),
                           curvatures[:, np.newaxis]))
    
    # Define the PCD header
    header = '# .PCD v0.7 - Point Cloud Data file format\nVERSION .7\nFIELDS x y z normal_x normal_y normal_z curvature\nSIZE 4 4 4 4 4 4 4\nTYPE F F F F F F F\nCOUNT 1 1 1 1 1 1 1\nWIDTH {}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {}\nDATA ascii'.format(pcd_array.shape[0], pcd_array.shape[0])
    
    # Save the PCD file with curvature
    storage = OverwriteStorage()
    file_content = header + '\n' + '\n'.join(' '.join(map(str, row)) for row in pcd_array)
    storage.save(fileName, ContentFile(file_content))
