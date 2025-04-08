import glob
import json
import numpy as np
import open3d as o3d
import cv2
from scipy.spatial.transform import Rotation as R
import os
import math

camera_h_fov = 86
half_fov = camera_h_fov / 2  # 43 degrees

coverage = []

def extract_point_cloud(folder_path, pitch):
    global camera_h_fov, coverage

    def normalize_angle(angle):
        while angle <= -180: angle += 360
        while angle > 180: angle -= 360
        return angle

    pc_left_angle = pitch * (180 / math.pi) - half_fov
    pc_right_angle = pitch * (180 / math.pi) + half_fov
    print([pc_left_angle, pc_right_angle])
    l = r = fov1 = fov2 = 0
    inside = False
    print(coverage)

    if not coverage:
        coverage.append([pc_left_angle, pc_right_angle])
    else:
        pc_left_angle = normalize_angle(pc_left_angle)
        pc_right_angle = normalize_angle(pc_right_angle)
        for fov in coverage:
            if pc_left_angle < pc_right_angle:
                if fov[0] < pc_left_angle and fov[1] > pc_right_angle:
                    inside = True
                    break
                if pc_left_angle > fov[0] and pc_left_angle < fov[1] and pc_right_angle > fov[1]:
                    fov1 = fov
                if pc_right_angle > fov[0] and pc_right_angle < fov[1] and pc_left_angle < fov[0]:
                    fov2 = fov
            else:
                if pc_right_angle < fov[1] and pc_left_angle > fov[0]:
                    fov1 = fov

        if fov1 == 0 and fov2 == 0 and not inside:
            coverage.append([pc_left_angle, pc_right_angle])
        if fov1 != 0:
            l = abs(pc_left_angle - fov1[1]) / camera_h_fov
            coverage.remove(fov1)
        if fov2 != 0:
            r = abs(fov2[0] - pc_right_angle) / camera_h_fov
            coverage.remove(fov2)

    if fov1 and fov2:
        coverage.append([fov1[0], fov2[1]])
    elif fov1:
        coverage.append([fov1[0], pc_right_angle])
    elif fov2:
        coverage.append([pc_left_angle, fov2[1]])
    if inside:
        l = r = 0.5

    print(coverage)
    print(l, r)

    # Load images
    depth_path = os.path.join(folder_path, "average_depth_image.png")
    color_path = os.path.join(folder_path, "color_image.png")
    depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    color_image = cv2.imread(color_path)
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    # Apply horizontal cutoff based on FOV overlap
    width = depth_image.shape[1]
    if l != 0:
        left_cutoff = int(l * width)
        depth_image[:, :left_cutoff] = 0
    if r != 0:
        right_cutoff = width - int(r * width)
        depth_image[:, right_cutoff:] = 0

    # Load intrinsics
    with open(os.path.join(folder_path, "intrinsics.json")) as f:
        intr = json.load(f)
    intr_o3d = o3d.camera.PinholeCameraIntrinsic(
        intr["width"], intr["height"], intr["fx"], intr["fy"], intr["ppx"], intr["ppy"]
    )

    # Convert to Open3D images
    depth_o3d = o3d.geometry.Image(depth_image)
    color_o3d = o3d.geometry.Image(color_image)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d, depth_scale=1000.0, depth_trunc=16.0, convert_rgb_to_intensity=False
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr_o3d)

    # Add pink cube at origin to represent camera position
    cube = o3d.geometry.TriangleMesh.create_box(0.1, 0.1, 0.1)
    cube.paint_uniform_color([1.0, 0.0, 1.0])
    return np.asarray(pcd.points), np.asarray(pcd.colors), cube


def load_imu_data(json_file):
    with open(json_file, 'r') as f:
        imu_data = json.load(f)
    angles = imu_data["angle"]
    return angles["roll"], angles["pitch"], angles["yaw"]


def rotate_point_cloud(points, roll, pitch, yaw):
    rotation_matrix = R.from_euler('xyz', [roll, pitch, yaw], degrees=False).as_matrix()
    return np.dot(points, rotation_matrix.T)


def main():
    folders = sorted(glob.glob("recordings/*"))
    
    merged_points = []
    merged_colors = []
    cubes = []  # List to store cubes for visualization
    for folder_path in folders:
        imu_file = os.path.join(folder_path, "imu_data.json")
        if not os.path.exists(imu_file):
            continue
        roll, pitch, yaw = load_imu_data(imu_file)
        
        result = extract_point_cloud(folder_path, pitch)
        if result is None:
            continue
        points, colors, cube = result
        
        # Apply rotation using IMU data
        rotated_points = rotate_point_cloud(points, roll, pitch, yaw)
        merged_points.append(rotated_points)
        merged_colors.append(colors)
        cubes.append(cube)  # Add the cube to the list
    
    if merged_points:
        final_cloud = np.vstack(merged_points)
        final_colors = np.vstack(merged_colors)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(final_cloud)
        pcd.colors = o3d.utility.Vector3dVector(final_colors)
        
        # Save and visualize the merged point cloud with the pink cube
        o3d.io.write_point_cloud("merged.ply", pcd)
        print("Merged point cloud saved as merged.ply")
        
        # Add the cubes to the visualization
        o3d.visualization.draw_geometries([pcd] + cubes)  # Visualize the point cloud with cubes


if __name__ == "__main__":
    main()
