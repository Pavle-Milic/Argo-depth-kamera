import glob
import json
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
import cv2
from scipy.spatial.transform import Rotation as R
import math

camera_h_fov = 86  # degrees (D455)
half_fov = camera_h_fov / 2  # 43 degrees

coverage=[]

def extract_point_cloud(bag_file, pitch):
    global camera_h_fov, coverage

    def normalize_angle(angle):
        """Normalize angle to be within the range [-180, 180] degrees."""
        while angle <= -180:
            angle += 360
        while angle > 180:
            angle -= 360
        return angle
    
    pc_left_angle=pitch* (180 / math.pi)-half_fov
    pc_right_angle=pitch* (180 / math.pi)+half_fov
    print([pc_left_angle,pc_right_angle])
    l=0
    r=0
    fov1=0
    fov2=0
    inside=False
    print(coverage)

    if not coverage :
        coverage.append([pc_left_angle,pc_right_angle])
    else :
        pc_left_angle=normalize_angle(pc_left_angle)
        pc_right_angle=normalize_angle(pc_right_angle)

        for fov in coverage:

            if pc_left_angle<pc_right_angle:
                
                if fov[0]<pc_left_angle and fov[1]>pc_right_angle:
                    inside=True
                    break

                if pc_left_angle>fov[0] and pc_left_angle<fov[1] and fov[0]<fov[1] and pc_right_angle>fov[1]:
                    fov1=fov
                    continue
                if pc_right_angle>fov[0] and pc_right_angle<fov[1] and fov[0]<fov[1] and pc_left_angle<fov[0]:
                    fov2=fov
                    continue
                if pc_left_angle<fov[1] and fov[1]<fov[0] and pc_right_angle>fov[1] and pc_right_angle<fov[0]:
                    fov1=fov
                    continue
                if pc_right_angle>fov[0] and fov[0]>fov[1] and pc_left_angle<fov[0] and pc_left_angle>fov[1]:
                    fov2=fov
                    continue
            else:

                if pc_left_angle>fov[0] and pc_left_angle>fov[0] and pc_right_angle>fov[0] and pc_right_angle<fov[1]:
                    fov2=fov
                    continue
                if pc_right_angle<fov[0] and pc_right_angle<fov[1] and pc_left_angle>fov[0] and pc_left_angle<fov[1]:
                    fov1=fov
                    continue
                if pc_left_angle>fov[0] and pc_right_angle>fov[1]:
                    fov1=fov
                    continue
                if pc_left_angle<fov[0] and pc_right_angle<fov[1]:
                    fov2=fov
                    continue
        
        if fov1==0 and fov2==0 and not inside:
            coverage.append([pc_left_angle,pc_right_angle])

        if fov1 !=0:
            l=pc_left_angle-fov1[1]
            l=abs(l)/camera_h_fov
            coverage.remove(fov1)
        if fov2 !=0:
            r=fov2[0]-pc_right_angle
            r=abs(r)/camera_h_fov
            coverage.remove(fov2)
    
    if fov1!=0 and fov2!=0:
        coverage.append([fov1[0],fov2[1]])
    elif fov1!=0 and fov2==0:
        coverage.append([fov1[0],pc_right_angle])
    elif fov2!=0 and fov1==0:
        coverage.append([pc_left_angle,fov2[1]])

    if inside:
        l=0.5
        r=0.5

    print(coverage)
    print(l)
    print(r)

    # Initialize RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(bag_file, repeat_playback=False)
    profile = pipeline.start(config)

    # Create an align object; here we choose to align depth to color.
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Extract depth intrinsics for later use in Open3D
    depth_stream = profile.get_stream(rs.stream.depth)
    depth_profile = depth_stream.as_video_stream_profile()
    intrinsics = depth_profile.get_intrinsics()
    intr_o3d = o3d.camera.PinholeCameraIntrinsic(
        intrinsics.width, intrinsics.height,
        intrinsics.fx, intrinsics.fy,
        intrinsics.ppx, intrinsics.ppy)

    # Variables for depth accumulation
    depth_sum = None
    valid_count = None
    frame_count = 0
    last_color_frame = None

    try:
        while True:
            frames = pipeline.wait_for_frames()
            # Align the frames so that depth and color are in the same coordinate space
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data()) 

            width = depth_image.shape[1]
            if l!=0:
                left_cutoff = int(l * width)
                depth_image[:, :left_cutoff] = 0
            if r!=0:
                right_cutoff = width-int(r * width)
                depth_image[:, right_cutoff:] = 0

            # Initialize accumulators on first frame
            if depth_sum is None:
                depth_sum = np.zeros_like(depth_image, dtype=np.float32)
                valid_count = np.zeros_like(depth_image, dtype=np.float32)

            # Create mask for valid depth (0.5 m to 16 m; sensor gives depth in mm)
            valid_mask = (depth_image > 500) & (depth_image < 16000)
            depth_sum[valid_mask] += depth_image[valid_mask]
            valid_count[valid_mask] += 1

            frame_count += 1
            last_color_frame = color_image  # Keep the last captured color frame

    except RuntimeError:
        print("✅ End of .bag file reached!")
    finally:
        pipeline.stop()
        print(f"✅ Processed {frame_count} frames.")

    if frame_count == 0:
        return None

    # Compute average depth image per pixel where valid measurements exist
    avg_depth_image = np.zeros_like(depth_sum, dtype=np.uint16)
    valid_pixels = valid_count > 0
    avg_depth_image[valid_pixels] = (depth_sum[valid_pixels] / valid_count[valid_pixels]).astype(np.uint16)

    # Ensure we have a valid color frame and convert it to RGB
    if last_color_frame is None:
        return None
    last_color_frame = cv2.cvtColor(last_color_frame, cv2.COLOR_BGR2RGB)

    # Convert the NumPy images into Open3D images
    depth_o3d = o3d.geometry.Image(avg_depth_image)
    color_o3d = o3d.geometry.Image(last_color_frame)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d,
        depth_scale=1000.0,  # because depth is in mm
        depth_trunc=16.0,    # truncate at 16 meters
        convert_rgb_to_intensity=False)

    # Create point cloud from the RGBD image using the extracted intrinsics.
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intr_o3d)

    # Create a pink cube to represent the camera position
    camera_position = [0, 0, 0]  # Camera's position in the point cloud (origin)
    cube_size = 0.1  # Cube size in meters (adjust as needed)
    cube = o3d.geometry.TriangleMesh.create_box(width=cube_size, height=cube_size, depth=cube_size)
    cube.translate(camera_position)  # Position the cube at the camera location
    cube.paint_uniform_color([1.0, 0.0, 1.0])  # Set the color to pink (RGB)

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
    bag_files = sorted(glob.glob("recordings/*/output.bag"))
    imu_files = sorted(glob.glob("recordings/*/imu_data.json"))
    
    merged_points = []
    merged_colors = []
    cubes = []  # List to store cubes for visualization
    for bag_file, imu_file in zip(bag_files, imu_files):
        roll, pitch, yaw = load_imu_data(imu_file)
        result = extract_point_cloud(bag_file,pitch)
        if result is None:
            continue
        points, colors, cube = result
        
        # Apply rotation using IMU data (this worked well in your first iteration)
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
