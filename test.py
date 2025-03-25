import pyrealsense2 as rs
import numpy as np
import cv2
import os

# Path to the .bag file
bag_file = "output.bag"  # Change this to your actual .bag file path

# Check if the file exists
if not os.path.exists(bag_file):
    print(f"❌ Error: .bag file NOT found at {bag_file}")
    exit()

# Configure pipeline
pipeline = rs.pipeline()
config = rs.config()

# Load from .bag file (do NOT manually enable streams)
config.enable_device_from_file(bag_file, repeat_playback=False)

# Initialize variables
first_color_frame = None
last_color_frame = None
depth_frame_count = 0

try:
    pipeline.start(config)  # Start pipeline
    print("✅ Pipeline started!")

    while True:
        frames = pipeline.wait_for_frames()

        # Get the color frame
        color_frame = frames.get_color_frame()
        if color_frame:
            color_image = np.asanyarray(color_frame.get_data())

            # Save first and last color frame
            if first_color_frame is None:
                first_color_frame = color_image
            last_color_frame = color_image  # Keep updating to get the last frame

        # Get the depth frame
        depth_frame = frames.get_depth_frame()
        if depth_frame:
            depth_image = np.asanyarray(depth_frame.get_data())

            # Normalize depth image for better visualization
            depth_image_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # Show depth frame
            cv2.imshow("Depth Frame", depth_image_normalized)
            depth_frame_count += 1

            if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit early
                break

except RuntimeError:
    print("✅ End of .bag file reached!")

finally:
    pipeline.stop()
    print(f"✅ Pipeline stopped. Total depth frames displayed: {depth_frame_count}")

    # Display first and last RGB frames
    if first_color_frame is not None and last_color_frame is not None:
        cv2.imshow("First RGB Frame", first_color_frame)
        cv2.imshow("Last RGB Frame", last_color_frame)
        cv2.waitKey(0)  # Wait for user input to close
        cv2.destroyAllWindows()
    else:
        print("⚠️ No RGB frames found in the .bag file!")

    cv2.destroyAllWindows()  # Close all OpenCV windows
