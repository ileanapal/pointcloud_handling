import os
import re
import numpy as np
import open3d as o3d
import math


def parse_foldername(foldername):
    # Parse the folder name to extract position coordinates and orientation angle
    match = re.match(r"\((-?\d+\.?\d*),\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*)\),\s*(\d+)", foldername)
    if match:
        # Extract coordinates and angle
        x, y, z = float(match.group(3)), float(match.group(1)), float(match.group(2))
        angle = float(match.group(4))
        pitch = 90 - angle  # Calculate pitch
        # if pitch > 180:
        #     pitch -= 360
        #     pitch = -pitch
        if pitch <= -180:
            pitch += 360
        return (x, y, z), pitch
    else:
        return None, None


def get_transform_matrices(position, roll, yaw, pitch, scale=100):
    # Calculate rotation and translation matrices based on position and orientation
    pitch_rad = math.radians(pitch)
    roll_rad = math.radians(roll)
    yaw_rad = math.radians(yaw)

    cos_pitch, sin_pitch = math.cos(pitch_rad), math.sin(pitch_rad)
    cos_roll, sin_roll = math.cos(roll_rad), math.sin(roll_rad)
    cos_yaw, sin_yaw = math.cos(yaw_rad), math.sin(yaw_rad)


    # Rotation matrix for pitch (around Y-axis)
    pitch_matrix = np.array([[cos_pitch, 0, sin_pitch],
                             [0, 1, 0],
                             [-sin_pitch, 0, cos_pitch]])

    # Rotation matrix for roll (around X-axis)
    roll_matrix = np.array([[1, 0, 0],
                            [0, cos_roll, -sin_roll],
                            [0, sin_roll, cos_roll]])

    # Rotation matrix for yaw (around Z-axis)
    yaw_matrix = np.array([[cos_yaw, -sin_yaw, 0],
                           [sin_yaw, cos_yaw, 0],
                           [0, 0, 1]])

    rot_matrix = roll_matrix @ (pitch_matrix @ yaw_matrix)

    # Translation matrix, scaling the position coordinates
    trans_matrix = np.array([position[0] * scale, position[1] * scale, position[2] * scale])

    return rot_matrix, trans_matrix


def merge_clouds(cloud_dirs, every_k_points=10):
    merged_cloud_p1 = None  # Store the merged point cloud for z=3 and y=0
    merged_cloud_p2 = None  # Store the merged point cloud for z=1 and y=2

    for dir in cloud_dirs:
        # Parse the folder name to get position and orientation
        position, pitch = parse_foldername(os.path.basename(dir))

        if position is None:
            print(f"Skipping directory {dir} due to invalid name format")
            continue

        # Get all .ply files in the directory
        cloud_files = [f for f in os.listdir(dir) if f.endswith(".ply")]
        if not cloud_files:
            print(f"No .ply files found in directory {dir}")
            continue

        cloud_file = os.path.join(dir, cloud_files[0])
        print(f"Processing {cloud_file}")
        cloud = o3d.io.read_point_cloud(cloud_file)
        print(f"Original point cloud size: {len(cloud.points)}")

        # Downsample the point cloud
        cloud = cloud.uniform_down_sample(every_k_points)
        print(f"Downsampled point cloud size: {len(cloud.points)}")

        scale = 100
        # Calculate rotation and translation matrices
        rot_matrix, trans_matrix = get_transform_matrices(position, roll=0, yaw=0, pitch=pitch, scale=scale)

        # Apply rotation and translation transformations
        rotated_points = np.asarray(cloud.points) @ rot_matrix.T
        translated_points = rotated_points + trans_matrix
        cloud.points = o3d.utility.Vector3dVector(translated_points)

        merged_cloud_p1, merged_cloud_p2, twoD_position_1, twoD_position_2 = get_two_point_clouds(merged_cloud_p1, merged_cloud_p2, cloud_dirs, position, cloud)
    
    # Fine tuning with mean and largest
    counter = 0
    while counter < 3:
        trans_correction_vector_largest = find_largest_coordinates(merged_cloud_p2) - find_largest_coordinates(merged_cloud_p1) + np.array([0, (twoD_position_2[0]-twoD_position_1[0]) * scale, (twoD_position_2[1]-twoD_position_1[1]) * scale])
        merged_cloud_p1 = translate(merged_cloud_p1, trans_correction_vector_largest)
        
        trans_correction_vector_mean = np.mean(merged_cloud_p2.points, axis=0) - np.mean(merged_cloud_p1.points, axis=0)
        merged_cloud_p1 = translate(merged_cloud_p1, trans_correction_vector_mean)
        
        counter += 1

    if merged_cloud_p1 is None:
        print("No ply files found for z=3 and y=0.")
    if merged_cloud_p2 is None:
        print("No ply files found for z=1 and y=2.")
    
    merged_cloud_all = merged_cloud_p1 + merged_cloud_p2

    return merged_cloud_p1, merged_cloud_p2, merged_cloud_all

def get_two_point_clouds(merged_cloud_p1, merged_cloud_p2, cloud_dirs, position, cloud):
    
    # Find the two positions (y, z) on the 2D plane where the pictures were taken
    set_of_positions = store_positions(cloud_dirs)
    twoD_positions = find_2D_points(set_of_positions)
    twoD_position_1, twoD_position_2 = twoD_positions[0], twoD_positions[1]

    # Check z and y coordinates for separation
    if position[1] == twoD_position_1[0] and position[2] == twoD_position_1[1]:
        if merged_cloud_p1 is None:
            merged_cloud_p1 = cloud
        else:
            merged_cloud_p1 += cloud
    elif position[1] == twoD_position_2[0] and position[2] == twoD_position_2[1]:
        if merged_cloud_p2 is None:
            merged_cloud_p2 = cloud
        else:
            merged_cloud_p2 += cloud
    
    return merged_cloud_p1, merged_cloud_p2, twoD_position_1, twoD_position_2


def translate(merged_cloud, trans_matrix):
    # Convert the point cloud to a NumPy array
    points = np.asarray(merged_cloud.points)

    # Add the translation correction vector to each point
    translated_points = points + trans_matrix

    # Update the translated points in the point cloud
    merged_cloud.points = o3d.utility.Vector3dVector(translated_points)

    return merged_cloud

def find_largest_coordinates(merged_cloud):
    max_x = max_y = max_z = -float('inf')  # Initialize with negative infinity

    for point in merged_cloud.points:
        if point[0] > max_z:
            max_x = point[0]
        if point[1] > max_y:
            max_y = point[1]
        if point[2] > max_x:
            max_z = point[2]
    
    max_array = np.array([max_x, max_y, max_z])

    return max_array

def find_smallest_coordinates(merged_cloud):
    min_x = min_y = min_z = float('inf')  # Initialize with positive infinity

    for point in merged_cloud.points:
        if point[0] < min_x:
            min_x = point[0]
        if point[1] < min_y:
            min_y = point[1]
        if point[2] < min_z:
            min_z = point[2]
    
    min_array = np.array([min_x, min_y, min_z])

    return min_array

def find_2D_points(positions):
    combinations = set()
    
    for array in positions:
        last_two_elements = array[-2:]
        combinations.add(last_two_elements)
        
    return np.array(list(combinations))

def store_positions(cloud_dirs):
    positions = set()

    for dir in cloud_dirs:
        position, _ = parse_foldername(os.path.basename(dir))
        if position is not None:
            positions.add(position)

    return positions

def draw_registration_result(source, target, transformation):
    copy = None
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4559,
                                      front=[0.6452, -0.3036, -0.7011],
                                      lookat=[1.9892, 2.0208, 1.8945],
                                      up=[-0.2779, -0.9482, 0.1556])
    
def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def prepare_dataset(voxel_size, merged_cloud_p1, merged_cloud_p2):
    print(":: Load two point clouds and disturb initial pose.")

    source = merged_cloud_p1
    target = merged_cloud_p2
    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)
    draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh

def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

if __name__ == "__main__":
    # Root directory path
    root_dir = "C:\\Users\\ilean\\Downloads\\pointcloud_handling-main-jul5\\pointcloud_handling-main\\PQ512"
    # Get all subdirectory paths under the root directory
    cloud_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    every_k_points = 10  # Downsample factor

    merged_cloud_p1, merged_cloud_p2, merged_cloud_all = merge_clouds(cloud_dirs, every_k_points)
    
    voxel_size = 0.05  # means 5cm for this dataset
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_size, merged_cloud_p1, merged_cloud_p2)

    result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    print(result_ransac)
    draw_registration_result(source_down, target_down, result_ransac.transformation)

    # if merged_cloud_all is not None:
    #     print("Merged point cloud")
    #     print(merged_cloud_all)

    #     o3d.visualization.draw_geometries([merged_cloud_all])
