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
        x, y, z = float(match.group(1)), float(match.group(2)), float(match.group(3))
        angle = float(match.group(4))
        pitch = 90 - angle  # Calculate pitch
        if pitch > 180:
            pitch -= 360
        elif pitch <= -180:
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
    merged_cloud = None  # Store the merged point cloud
    for dir in cloud_dirs:
        # Parse the folder name to get position and orientation
        position, pitch = parse_foldername(os.path.basename(dir))
        if position is None:
            print(f"Skipping directory {dir} due to invalid name format")
            continue

        # For debugging:
        # if not (position[0] == 0 and position[1] == 3):
        #     continue

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

        # Calculate rotation and translation matrices
        rot_matrix, trans_matrix = get_transform_matrices(position, roll=0, yaw=0 ,pitch=pitch, scale=100)

        # Apply rotation and translation transformations
        rotated_points = np.asarray(cloud.points) @ rot_matrix.T
        translated_points = rotated_points + trans_matrix
        cloud.points = o3d.utility.Vector3dVector(translated_points)

        # Merge point clouds
        if merged_cloud is None:
            merged_cloud = cloud
        else:
            merged_cloud += cloud

    if merged_cloud is None:
        print("No valid point clouds found")
        return None

    print(f"Final merged point cloud size: {len(merged_cloud.points)}")
    # Write the merged point cloud to a file
    o3d.io.write_point_cloud("merged_cloud.ply", merged_cloud)

    return merged_cloud


if __name__ == "__main__":
    # Root directory path
    root_dir = "C:\\Users\\iansy\\Downloads\\PQ512(32)"
    # Get all subdirectory paths under the root directory
    cloud_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

    every_k_points = 100  # Downsampling parameter, specifying the number of points to skip
    merged_cloud = merge_clouds(cloud_dirs, every_k_points)

    if merged_cloud is not None:
        o3d.visualization.draw_geometries([merged_cloud])