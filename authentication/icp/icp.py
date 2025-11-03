import open3d as o3d  # Open3D ライブラリ (Library)
from profiling_decorator import profile

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

@profile
def execute_icp(source, target, fast_registration=False, threshold=0.7, fgr_config=None):
    voxel_size = 0.1  # 10cm for this dataset

    source, _ = remove_outlier(source)
    target, _ = remove_outlier(target)

    source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_size, source, target)

    if fast_registration:
        print("Initial alignment with Fast Global Registration")

        # Pass fgr_config here:
        result = execute_fast_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size, fgr_config=fgr_config)
    else:
        print("Initial alignment with RANSAC")
        result = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)

    trans_init = result.transformation
    evaluation = o3d.pipelines.registration.evaluate_registration(source, target, threshold, trans_init)
    print(evaluation)

    print("Now, applying the point-to-plane ICP")

    try:
        reg_p2l = o3d.pipelines.registration.TransformationEstimationPointToPlane()
        reg_p2l_result = o3d.pipelines.registration.registration_icp(
            source_down, target_down, threshold, trans_init, reg_p2l)
        print(reg_p2l_result)
        print("Transformation is:")
        print(reg_p2l_result.transformation)
    except Exception as e:
        print(f"ICP alignment failed: {e}")
        return 0

    return reg_p2l_result.fitness

# 点群データの前処理関数 (Preprocessing function for point cloud data)
def preprocess_point_cloud(pcd, voxel_size):
    # ダウンサンプリング (Downsampling)
    print(f":: Downsample with a voxel size {voxel_size:.3f}.")
    pcd_down = pcd.voxel_down_sample(voxel_size)

    # 法線推定 (Normal estimation)
    radius_normal = voxel_size * 2
    print(f":: Estimate normal with search radius {radius_normal:.3f}.")
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    pcd_down.orient_normals_consistent_tangent_plane(100)

    # FPFH 特徴量計算 (FPFH feature calculation)
    radius_feature = voxel_size * 5
    print(f":: Compute FPFH feature with search radius {radius_feature:.3f}.")
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

# データセット準備関数 (Prepare dataset function)
def prepare_dataset(voxel_size, source, target):
    # 2つの点群データの読み込みと初期姿勢の変更 (Load two point cloud data and change the initial orientation)
    print(":: Load two point clouds and disturb initial pose.")
    
    # 前処理の適用 (Apply preprocessing)
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

    return source_down, target_down, source_fpfh, target_fpfh

# グローバルレジストレーション実行 (Execute global registration)
def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print(f"   Since the downsampling voxel size is {voxel_size:.3f},")
    print(f"   we use a liberal distance threshold {distance_threshold:.3f}.")

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
def execute_fast_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size, fgr_config=None):
    # Set defaults
    distance_threshold = voxel_size * 1.1
    iteration_number = 512
    maximum_tuple_count = 1534
    tuple_scale = 0.98
    use_absolute_scale = False
    decrease_mu = True
    tuple_test = True
    division_factor = 1.01

    # Override defaults if fgr_config provided
    if fgr_config:
        distance_threshold = voxel_size * fgr_config.get('distance_multiplier', 1.1)
        iteration_number = fgr_config.get('iteration_number', iteration_number)
        maximum_tuple_count = fgr_config.get('maximum_tuple_count', maximum_tuple_count)
        tuple_scale = fgr_config.get('tuple_scale', tuple_scale)  # optional, you can add it to config if needed
        use_absolute_scale = fgr_config.get('use_absolute_scale', use_absolute_scale)
        decrease_mu = fgr_config.get('decrease_mu', decrease_mu)
        tuple_test = fgr_config.get('tuple_test', tuple_test)
        division_factor = fgr_config.get('division_factor', division_factor)

    print(f":: Apply fast global registration with distance threshold {distance_threshold:.3f}")
    print(f"FGR params: iteration_number={iteration_number}, maximum_tuple_count={maximum_tuple_count}, division_factor={division_factor}")

    option = o3d.pipelines.registration.FastGlobalRegistrationOption(
        maximum_correspondence_distance=distance_threshold,
        iteration_number=iteration_number,
        maximum_tuple_count=maximum_tuple_count,
        tuple_scale=tuple_scale,
        use_absolute_scale=use_absolute_scale,
        decrease_mu=decrease_mu,
        tuple_test=tuple_test,
        division_factor=division_factor
    )

    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down,
        source_fpfh, target_fpfh,
        option
    )
    return result


# 点群の外れ値を除去する関数 (Function to remove outliers from point cloud)
def remove_outlier(cloud):
    print("Removing outliers")
    cl, ind = cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return cl, ind
