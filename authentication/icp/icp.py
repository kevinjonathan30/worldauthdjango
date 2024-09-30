import open3d as o3d  # Open3D ライブラリ (Library)
import numpy as np  # NumPy ライブラリ (Library)
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('authentication')

# icp-based 3D authentication based on Open3D 0.15.2
#
# このコードは、Open3D 0.15.2 を使用して Iterative Closest Point (ICP) ベースの 3D 認証を実行するものです。
# 主な機能は以下のとおりです。
# 1. 点群データの前処理（ダウンサンプリング、法線推定、FPFH 特徴量計算）
# 2. 点群データセットの準備
# 3. グローバルレジストレーションの実行（RANSAC）
# 4. 点対点または点対面の ICP アルゴリズムを使用した精細な位置合わせ
# 5. 最終的な変換行列を表示し、一致した点群を視覚化
#
# このコードは、2つの点群データセット（`test2.pcd` と `test3.pcd`）を入力として受け取り、それらを適切に位置合わせます。

def execute_icp(source, target, fast_registration=False, threshold=0.7):
    source = o3d.io.read_point_cloud(source)
    target = o3d.io.read_point_cloud(target)

    voxel_size = 0.1  # このデータセットに対して10cmを意味します (Means 10cm for this dataset)
    
    # 点群の外れ値を除去 (Remove outliers from the source and target point clouds)
    source, _ = remove_outlier(source)
    target, _ = remove_outlier(target)

    # 法線推定 (Estimate normals)
    estimate_normals(source, voxel_size)
    estimate_normals(target, voxel_size)
    
    # データセットの準備 (Prepare datasets)
    source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_size, source, target)

    # 初期位置合わせを実行 (Perform initial alignment)
    if fast_registration:
        logger.info("Initial alignment with Fast Global Registration")
        result = execute_fast_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    else:
        logger.info("Initial alignment with RANSAC")
        result = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)

    trans_init = result.transformation
    evaluation = o3d.pipelines.registration.evaluate_registration(source, target, threshold, trans_init)
    logger.info(evaluation)

    # 点対面ICPを適用 (Apply point-to-plane ICP)
    logger.info("Now, applying the point-to-plane ICP")

    try:
        reg_p2l = o3d.pipelines.registration.TransformationEstimationPointToPlane()
        reg_p2l_result = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init, reg_p2l)
        logger.info(reg_p2l_result)
        logger.info("Transformation is:")
        logger.info(reg_p2l_result.transformation)
    except Exception as e:
        logger.error(f"ICP alignment failed: {e}")
        return None

    return reg_p2l_result.fitness

# 点群データの法線を推定する関数 (Function to estimate normals for a point cloud)
def estimate_normals(pcd, voxel_size):
    radius_normal = voxel_size * 2
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(100)

# 点群データの前処理関数 (Preprocessing function for point cloud data)
def preprocess_point_cloud(pcd, voxel_size):
    # ダウンサンプリング (Downsampling)
    logger.info(f":: Downsample with a voxel size {voxel_size:.3f}.")
    pcd_down = pcd.voxel_down_sample(voxel_size)

    # 法線推定 (Normal estimation)
    radius_normal = voxel_size * 2
    logger.info(f":: Estimate normal with search radius {radius_normal:.3f}.")
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    # FPFH 特徴量計算 (FPFH feature calculation)
    radius_feature = voxel_size * 5
    logger.info(f":: Compute FPFH feature with search radius {radius_feature:.3f}.")
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

# データセット準備関数 (Prepare dataset function)
def prepare_dataset(voxel_size, source, target):
    # 2つの点群データの読み込みと初期姿勢の変更 (Load two point cloud data and change the initial orientation)
    logger.info(":: Load two point clouds and disturb initial pose.")
    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)

    # 前処理の適用 (Apply preprocessing)
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

    return source_down, target_down, source_fpfh, target_fpfh

# グローバルレジストレーション実行 (Execute global registration)
def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    logger.info(":: RANSAC registration on downsampled point clouds.")
    logger.info(f"   Since the downsampling voxel size is {voxel_size:.3f},")
    logger.info(f"   we use a liberal distance threshold {distance_threshold:.3f}.")

    checkers = [
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
    ]
    criteria = o3d.pipelines.registration.RANSACConvergenceCriteria(10000, 500)
    reg = o3d.pipelines.registration.TransformationEstimationPointToPoint(False)

    # RANSACベースのグローバルレジストレーション (RANSAC-based global registration)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=reg,
        ransac_n=3,
        checkers=checkers,
        criteria=criteria)
    return result

# 高速グローバルレジストレーション実行 (Execute fast global registration)
def execute_fast_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    logger.info(f":: Apply fast global registration with distance threshold {distance_threshold:.3f}")
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result

# 点群の外れ値を除去する関数 (Function to remove outliers from point cloud)
def remove_outlier(cloud):
    logger.info("Removing outliers")
    cl, ind = cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return cl, ind
