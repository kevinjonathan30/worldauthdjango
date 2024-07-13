# icp-based 3D authentication based on Open3D 0.15.2
#
#このコードは、Open3D 0.15.2 を使用して Iterative Closest Point (ICP) ベースの 3D 認証を実行するものです。主な機能は以下のとおりです。
#
#1. 点群データの前処理（ダウンサンプリング、法線推定、FPFH 特徴量計算）
#2. 点群データセットの準備
#3. グローバルレジストレーションの実行（RANSAC）
#4. 点対点または点対面の ICP アルゴリズムを使用した精細な位置合わせ
#5. 最終的な変換行列を表示し、一致した点群を視覚化
#
#このコードは、2つの点群データセット（`test2.pcd` と `test3.pcd`）を入力として受け取り、それらを適切に位置合わせます。

#EN:
# icp-based 3D authentication based on Open3D 0.15.2
#
#This code uses Open3D 0.15.2 to perform Iterative Closest Point (ICP) based 3D authentication. The main features are:
#
#1. Preprocessing of point cloud data (downsampling, normal estimation, FPFH feature calculation)
#2. Preparing point cloud dataset
#3. Execute global registration (RANSAC)
#4. Fine alignment using point-to-point or point-to-point ICP algorithms
#5. Display the final transformation matrix and visualize the matched point cloud
#
#This code takes two point cloud datasets (`test2.pcd` and `test3.pcd`) as input and aligns them appropriately.

import open3d as o3d  # Open3D ライブラリ (Library)
import numpy as np  # NumPy ライブラリ (Library)

def execute_icp(source, target):
    threshold = 0.7
    voxel_size = 0.1  # Means 10cm for this dataset
    
    # Remove outliers from the source and target point clouds
    # source, _ = remove_outlier(source)
    # target, _ = remove_outlier(target)
    
    source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_size, source, target)

    # グローバルレジストレーション（RANSAC Global Registration）
    print("Initial alignment with RANSAC")
    result = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
    
    # result = execute_fast_global_registration(source_down, target_down,
                                                    # source_fpfh, target_fpfh,
                                                    # voxel_size)

    trans_init = result.transformation
    evaluation = o3d.pipelines.registration.evaluate_registration(source, target,
                                                        threshold, trans_init)
    print(evaluation)

    # Point-to-plain ICP
    print("Now, applying the point-to-plane ICP")

    try:
        source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        source.orient_normals_consistent_tangent_plane(100)

        target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        target.orient_normals_consistent_tangent_plane(100)
        p2l = o3d.pipelines.registration.TransformationEstimationPointToPlane()
        reg_p2l_result = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init, p2l)
    except Exception as e:
        print(e)

    print(reg_p2l_result)
    print("Transformation is:")
    print(reg_p2l_result.transformation)
    print("")
    return reg_p2l_result.fitness

# 点群データの前処理関数 (Preprocessing functions for point cloud data)
def preprocess_point_cloud(pcd, voxel_size):
    # ダウンサンプリング (Downsampling)
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    # 法線推定 (Normal estimation)
    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    # FPFH 特徴量計算 (Feature calculation)
    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

# データセット準備関数 (Prepare dataset function)
def prepare_dataset(voxel_size, source, target):
    # 2つの点群データの読み込みと初期姿勢の変更 trans_init で適当に回転 (Load two point cloud data and change the initial orientation and rotate appropriately using trans_init)
    print(":: Load two point clouds and disturb initial pose.")
    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    
    source.transform(trans_init)

    # 前処理の適用 (Applying preprocessing)
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

    # 初期位置合わせの描画 (Drawing initial alignment)
    return source_down, target_down, source_fpfh, target_fpfh

# グローバルレジストレーション実行 (Execute global registration)
def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    # パラメータ設定 (Parameter setting)
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)

    mutual_filter = True
    checkers = [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)]
    criteria = o3d.pipelines.registration.RANSACConvergenceCriteria(10000,500)
    ransac_n = 3
    reg = o3d.pipelines.registration.TransformationEstimationPointToPoint(False)

    # RANSAC ベースのグローバルレジストレーション (Base global registration)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        mutual_filter,
        distance_threshold,
        reg,
        ransac_n,
        checkers,
        criteria)
    return result

def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: Apply fast global registration with distance threshold %.3f" \
            % distance_threshold)
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result

def remove_outlier(cloud):
    # Using statistical outlier removal
    print("Removing outliers")
    cl, ind = cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return cl, ind
