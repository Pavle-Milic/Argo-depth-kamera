import pyrealsense2 as rs
import numpy as np
import cv2
import os
import json
from datetime import datetime
import time

# Configure the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 5)  
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 5)   
config.enable_stream(rs.stream.gyro)   
config.enable_stream(rs.stream.accel)  

# Start the pipeline normally
profile = pipeline.start(config)

# Get intrinsics
depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

# Angle tracking variables
angle = np.array([0.0, 0.0, 0.0])  # Initial roll, pitch, yaw
prev_time = time.time()
gyro_threshold = 0.01  # Ignore noise below this threshold

def get_gyro_data():
    """Retrieve gyroscope data and update angle using integration, ignoring small changes."""
    global angle, prev_time

    frames = pipeline.wait_for_frames()  # Ensure we get fresh frames
    gyro_frame = frames.first_or_default(rs.stream.gyro)

    if gyro_frame:
        gyro_data = gyro_frame.as_motion_frame().get_motion_data()
        curr_time = time.time()
        dt = curr_time - prev_time  

        if dt > 0.001:  # Prevent division by zero
            gyro_rates = np.array([gyro_data.x, gyro_data.y, gyro_data.z])  
            gyro_rates[np.abs(gyro_rates) < gyro_threshold] = 0  # Ignore small movements
            angle += gyro_rates * dt  
            prev_time = curr_time  

def capture():
    """Capture a single frame into a .bag file and save RGB, Depth, and IMU data."""
    global pipeline

    # Generate a unique folder for this capture
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"recordings/{timestamp}"
    os.makedirs(folder_name, exist_ok=True)

    with open(os.path.join(folder_name, "intrinsics.json"), "w") as f:
        json.dump({
            "width": depth_intrinsics.width,
            "height": depth_intrinsics.height,
            "ppx": depth_intrinsics.ppx,
            "ppy": depth_intrinsics.ppy,
            "fx": depth_intrinsics.fx,
            "fy": depth_intrinsics.fy,
            "model": str(depth_intrinsics.model),
            "coeffs": list(depth_intrinsics.coeffs)
        }, f, indent=4)

    try:
        align = rs.align(rs.stream.color)  # Create the align object (only once, not per frame)

        # In your main loop or in capture():
        frames = pipeline.wait_for_frames()       # Get the raw frames
        aligned_frames = align.process(frames)    # Align depth to color
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # Convert images
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Save images
        cv2.imwrite(os.path.join(folder_name, "color_image.png"), color_image)
        cv2.imwrite(os.path.join(folder_name, "depth_image.png"), depth_image)

        depth_frames = []  

        for i in range(20): 
            frames = pipeline.wait_for_frames()       # Get the raw frames
            depth_frame = aligned_frames.get_depth_frame()
            
            if not depth_frame:
                continue

            # Convert depth frame to numpy array and store
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_frames.append(depth_image)

            print(f"Captured depth frame {i + 1}")

        if len(depth_frames) == 20:
            # Calculate the average depth frame
            depth_stack = np.stack(depth_frames, axis=0)
            average_depth_image = np.mean(depth_stack, axis=0).astype(np.uint16)

            # Save the averaged depth image
            cv2.imwrite(os.path.join(folder_name, "average_depth_image.png"), average_depth_image)
            print(f"Averaged depth frame saved in: {folder_name}")

        # Save IMU data separately as a JSON file
        imu_data = {
            "angle": {"roll": angle[0], "pitch": angle[1], "yaw": angle[2]}
            }
    
        with open(os.path.join(folder_name, "imu_data.json"), "w") as f:
            json.dump(imu_data, f, indent=4)

        print(f"Frame saved in: {folder_name}")

    finally:
        print("Sacuvano")

# OpenCV Window Loop
cv2.namedWindow("RealSense Stream", cv2.WINDOW_AUTOSIZE)

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        get_gyro_data()  # Update angles

        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())

        angle_text = f"Roll: {angle[0]:.2f}, Pitch: {angle[1]:.2f}, Yaw: {angle[2]:.2f}"
        cv2.putText(color_image, angle_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("RealSense Stream", color_image)

        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # Spacebar
            capture()
        elif key == 27:  # ESC
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
