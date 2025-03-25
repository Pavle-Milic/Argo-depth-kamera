import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import cv2
import os

# Path to .bag file
bag_file = "output.bag"  # Change this if needed

# Configure the pipeline to read from the .bag file
pipeline = rs.pipeline()
config = rs.config()
config.enable_device_from_file(bag_file, repeat_playback=False)

# Start the pipeline
profile = pipeline.start(config)

# Create a point cloud object
pc = rs.pointcloud()

# Initialize variables for depth accumulation
depth_sum = None
frame_count = 0
last_color_frame = None

try:
    while True:
        frames = pipeline.wait_for_frames()

        # Extract depth and color frames
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Convert depth frame to numpy array
        depth_image = np.asanyarray(depth_frame.get_data())

        # Accumulate depth values for averaging
        if depth_sum is None:
            depth_sum = np.zeros_like(depth_image, dtype=np.float32)
        
        # Create a mask for valid depth values (0.5m < depth < 16m)
        valid_mask = (depth_image > 500) & (depth_image < 16000)  # Convert meters to mm

        # Accumulate only valid depth values
        depth_sum[valid_mask] += depth_image[valid_mask]
        frame_count += 1

        # Store the last color frame
        last_color_frame = np.asanyarray(color_frame.get_data())

except RuntimeError:
    print("✅ End of .bag file reached!")

finally:
    pipeline.stop()
    print(f"✅ Processed {frame_count} depth frames.")

    if frame_count == 0:
        print("⚠️ No valid frames found in the .bag file!")
        exit()

    # Compute the average depth image
    avg_depth_image = (depth_sum / frame_count).astype(np.uint16)

    # Ensure last_color_frame is valid
    if last_color_frame is None:
        print("⚠️ No valid color frames found in the .bag file!")
        exit()

    # Convert last color frame to RGB
    last_color_frame = cv2.cvtColor(last_color_frame, cv2.COLOR_BGR2RGB)

    color_frame = frames.get_color_frame()
    if not color_frame:
        print("⚠️ No valid color frames available for mapping!")
        exit()

    pc.map_to(color_frame)

    # PROBLEM: rs.depth_frame(avg_depth_image) is incorrect.
    # FIX: Use RealSense processing blocks to process the depth frame properly.
    depth_frame = frames.get_depth_frame()
    if not depth_frame:
        print("⚠️ No valid depth frames available for point cloud generation!")
        exit()

    points = pc.calculate(depth_frame)

    # Convert RealSense point cloud to NumPy array
    vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
    tex_coords = np.asanyarray(points.get_texture_coordinates()).view(np.float32).reshape(-1, 2)

    # Create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vtx)

    # Get RGB colors from the last color frame
    h, w, _ = last_color_frame.shape
    colors = []
    for u, v in tex_coords:
        x = min(max(int(u * w), 0), w - 1)
        y = min(max(int(v * h), 0), h - 1)
        colors.append(last_color_frame[y, x] / 255.0)  # Normalize colors to [0,1]

    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

    # Visualize the colored point cloud
    o3d.visualization.draw_geometries([pcd])