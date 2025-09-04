# ==============================================================================
# SCRIPT 1: extract_poses.py
#
# Purpose: Processes a video file to extract and save human pose keypoints.
# Usage: python extract_poses.py your_video.mp4 --output poses.parquet
# ==============================================================================

import time
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import cv2
from extract_human_pose import HumanPoseExtractor # Assuming this file is in the same directory

def extract_and_save_poses(video_path, output_path, left_handed=False, start_frame=None):
    """
    Processes a video to extract human pose data and saves it to a file.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames to process: {total_frames}")

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        cap.release()
        return

    human_pose_extractor = HumanPoseExtractor(frame.shape)
    
    all_features = []
    frame_id = 0
    start_time = time.time()

    while cap.isOpened():
        if frame_id > 0: # First frame is already read
            ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_id += 1

        # --- Progress Tracking ---
        if frame_id % 100 == 0:
            elapsed_time = time.time() - start_time
            progress_percent = (frame_id / total_frames) * 100
            fps = frame_id / elapsed_time if elapsed_time > 0 else 0
            eta = (total_frames - frame_id) / fps if fps > 0 else 0
            print(f"Progress: {frame_id}/{total_frames} ({progress_percent:.1f}%) | "
                  f"Speed: {fps:.1f} fps | ETA: {eta/60:.1f} min")

        if start_frame is not None and frame_id < start_frame:
            continue

        # --- Pose Extraction ---
        human_pose_extractor.extract(frame)
        
        # We only care about the keypoints, not the ROI validity for saving
        human_pose_extractor.discard(["left_eye", "right_eye", "left_ear", "right_ear"])
        
        features = human_pose_extractor.keypoints_with_scores.reshape(17, 3)
        
        if left_handed:
            features[:, 1] = 1 - features[:, 1] # Flip x-coordinates

        # Filter for confident keypoints and flatten
        confident_features = features[features[:, 2] > 0][:, 0:2].flatten()
        
        # Create a record for this frame
        # We pad with NaNs if not all 13 keypoints are visible
        # The model expects a 26-feature vector (13 keypoints * 2 coords)
        feature_vector = np.full(26, np.nan)
        feature_vector[:len(confident_features)] = confident_features
        
        all_features.append({
            'frame_id': frame_id,
            'features': feature_vector
        })

    cap.release()
    
    # --- Save to a structured format ---
    print("\nProcessing complete. Converting to DataFrame and saving...")
    df = pd.DataFrame(all_features)
    
    # Explode the features array into separate columns for easier processing later
    feature_cols = [f'feature_{i}' for i in range(26)]
    df[feature_cols] = pd.DataFrame(df['features'].tolist(), index=df.index)
    df = df.drop(columns=['features'])
    
    # Using Parquet for efficiency with numerical data
    df.to_parquet(output_path, index=False)
    print(f"✅ Successfully saved pose data to {output_path}")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Extract human pose keypoints from a video and save them."
    )
    parser.add_argument("video", help="Path to the input video file.")
    parser.add_argument("--output", default="poses.parquet", help="Path to save the output Parquet file.")
    parser.add_argument("-f", type=int, help="Frame ID to start processing from.")
    parser.add_argument(
        "--left-handed",
        action="store_true", # Changed to action="store_true" for a simple flag
        help="Set this flag if the player is left-handed.",
    )
    args = parser.parse_args()
    
    extract_and_save_poses(args.video, args.output, args.left_handed, args.f)
