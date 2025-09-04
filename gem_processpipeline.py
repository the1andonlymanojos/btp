import os
# This line MUST be before the tensorflow import
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import time
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import cv2
import imageio
from collections import deque
from tqdm import tqdm

# --- Constants ---
# Biomechanics
RACQUET_LENGTH_M = 0.4 
TORSO_TO_HEIGHT_RATIO = 0.3
# Processing
SLIDING_WINDOW_SIZE = 30
# Keypoint Definitions
KEYPOINTS = {
    "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6, "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10, "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16,
}
KEYPOINTS_INV = {v: k for k, v in KEYPOINTS.items()}
EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7), (7, 9), (6, 8),
    (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
]

class RoI:
    """Manages the Region of Interest for player tracking."""
    def __init__(self, frame_shape):
        self.frame_height, self.frame_width = frame_shape[:2]
        self.reset()

    def reset(self):
        self.width = self.frame_width
        self.height = self.frame_height
        self.center_x = self.frame_width // 2
        self.center_y = self.frame_height // 2
        self.valid = False

    def update(self, keypoints):
        visible_kps = keypoints[keypoints[:, 2] > 0.1]
        if len(visible_kps) < 5:
            self.reset()
            return

        min_x = int(np.min(visible_kps[:, 1]))
        max_x = int(np.max(visible_kps[:, 1]))
        min_y = int(np.min(visible_kps[:, 0]))
        max_y = int(np.max(visible_kps[:, 0]))

        self.center_x = (min_x + max_x) // 2
        self.center_y = (min_y + max_y) // 2
        
        roi_w = int((max_x - min_x) * 1.5)
        roi_h = int((max_y - min_y) * 1.5)
        self.width = self.height = max(roi_w, roi_h, 192) # Ensure a minimum size

        # Clamp to frame boundaries
        self.center_x = np.clip(self.center_x, self.width // 2, self.frame_width - self.width // 2)
        self.center_y = np.clip(self.center_y, self.height // 2, self.frame_height - self.height // 2)
        self.valid = True

    def extract_subframe(self, frame):
        y1 = self.center_y - self.height // 2
        y2 = self.center_y + self.height // 2
        x1 = self.center_x - self.width // 2
        x2 = self.center_x + self.width // 2
        return frame[y1:y2, x1:x2]

class HumanPoseExtractor:
    """Handles Movenet inference."""
    def __init__(self, model_path="4.tflite"):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_size = self.input_details[0]['shape'][1]

    def run_inference(self, image):
        img_resized = tf.image.resize_with_pad(image, self.input_size, self.input_size)
        input_image = tf.cast(img_resized, dtype=tf.uint8)
        
        self.interpreter.set_tensor(self.input_details[0]['index'], np.expand_dims(input_image, axis=0))
        self.interpreter.invoke()
        
        keypoints = self.interpreter.get_tensor(self.output_details[0]['index'])
        return np.squeeze(keypoints)

class ShotDetector:
    """Detects shots from a sequence of keypoints and triggers analysis."""
    def __init__(self, model, class_names, confidence_threshold=0.9, cooldown=90):
        self.model = model
        self.class_names = class_names
        self.threshold = confidence_threshold
        self.cooldown = cooldown
        self.frames_since_last_shot = cooldown
        self.keypoint_history = deque(maxlen=SLIDING_WINDOW_SIZE)
        self.shot_count = 0
        self.last_probs = np.zeros(len(class_names))

    def process_keypoints(self, keypoints, frame_buffer):
        # We only care about the 13 keypoints used for training
        pose_kps = keypoints[:13, :2].flatten()
        self.keypoint_history.append(pose_kps)
        self.frames_since_last_shot += 1

        if len(self.keypoint_history) == SLIDING_WINDOW_SIZE and self.frames_since_last_shot >= self.cooldown:
            feature_vector = np.array(self.keypoint_history)
            feature_vector = np.expand_dims(feature_vector, axis=0)
            
            probs = self.model.predict(feature_vector, verbose=0)[0]
            self.last_probs = probs
            
            prediction_idx = np.argmax(probs)
            confidence = probs[prediction_idx]
            
            if confidence > self.threshold and self.class_names[prediction_idx] != 'neutral':
                self.frames_since_last_shot = 0
                self.shot_count += 1
                shot_name = self.class_names[prediction_idx]
                
                # Return all necessary info for analysis and saving
                return {
                    "shot_name": shot_name,
                    "shot_sequence_data": np.array(self.keypoint_history),
                    "shot_sequence_frames": list(frame_buffer),
                    "shot_number": self.shot_count
                }
        return None

class VideoAnnotator:
    """Handles all drawing operations on the video frames."""
    def __init__(self, class_names):
        self.class_names = class_names
        self.shot_counters = {name: 0 for name in class_names}
        self.last_detected_shot = "None"
        self.last_metrics = {}

    def update_shot_data(self, shot_info, metrics):
        self.last_detected_shot = shot_info['shot_name']
        self.shot_counters[self.last_detected_shot] += 1
        self.last_metrics = metrics

    def draw_frame(self, frame, roi, keypoints, probs):
        # Draw skeleton
        for i, j in EDGES:
            if keypoints[i, 2] > 0.1 and keypoints[j, 2] > 0.1:
                p1 = (int(keypoints[i, 1]), int(keypoints[i, 0]))
                p2 = (int(keypoints[j, 1]), int(keypoints[j, 0]))
                cv2.line(frame, p1, p2, (0, 255, 255), 2)
        # Draw RoI
        if roi.valid:
            y1, y2 = roi.center_y - roi.height // 2, roi.center_y + roi.height // 2
            x1, x2 = roi.center_x - roi.width // 2, roi.center_x + roi.width // 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Draw probability bars
        self._draw_probs(frame, probs)
        # Draw counters and metrics
        self._draw_counters(frame)
        self._draw_metrics(frame)
        return frame

    def _draw_probs(self, frame, probs):
        h, w = frame.shape[:2]
        for i, prob in enumerate(probs):
            label = self.class_names[i][:4]
            bar_height = int(prob * 100)
            cv2.rectangle(frame, (w - 40 - i*30, h - 20), (w - 20 - i*30, h - 20 - bar_height), (255,100,0), -1)
            cv2.putText(frame, label, (w - 40 - i*30, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

    def _draw_counters(self, frame):
        y_offset = 30
        for name, count in self.shot_counters.items():
            if name == 'neutral': continue
            text = f"{name}: {count}"
            cv2.putText(frame, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30

    def _draw_metrics(self, frame):
        h, w = frame.shape[:2]
        cv2.putText(frame, f"Last Shot: {self.last_detected_shot}", (w//2 - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        if self.last_metrics:
            speed = self.last_metrics.get('est_racquet_head_speed_kmh', 0)
            arc = self.last_metrics.get('swing_arc_degrees', 0)
            text = f"Speed: {speed:.1f} km/h | Arc: {arc:.1f} deg"
            cv2.putText(frame, text, (w//2 - 150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)


def analyze_biomechanics(shot_data, label, fps=30, user_height_m=None):
    """Calculates biomechanical metrics from a single shot's keypoint data."""
    if 'forehand' in label:
        wrist_kp_idx, shoulder_kp_idx = 10, 6 # right_wrist, right_shoulder
    else: # backhand, serve, etc. (assuming right-handed)
        wrist_kp_idx, shoulder_kp_idx = 9, 5 # left_wrist, left_shoulder
    
    wrist_traj = shot_data[:, [2*wrist_kp_idx, 2*wrist_kp_idx+1]]
    velocities = np.linalg.norm(np.diff(wrist_traj, axis=0), axis=1) * fps
    if len(velocities) == 0: return {}
    
    max_wrist_vel_relative = np.max(velocities)
    contact_frame_idx = np.argmax(velocities) + 1

    scale_m_per_unit = None
    if user_height_m:
        shoulder_pos = shot_data[0, [2*shoulder_kp_idx, 2*shoulder_kp_idx+1]]
        hip_kp_idx = 12 if 'right' in KEYPOINTS_INV[shoulder_kp_idx*17//13] else 11
        hip_pos = shot_data[0, [2*hip_kp_idx, 2*hip_kp_idx+1]]
        torso_dist_norm = np.linalg.norm(shoulder_pos - hip_pos)
        if torso_dist_norm > 0:
            scale_m_per_unit = (user_height_m * TORSO_TO_HEIGHT_RATIO) / torso_dist_norm

    metrics = {"contact_frame": contact_frame_idx}
    if scale_m_per_unit:
        max_wrist_speed_mps = max_wrist_vel_relative * scale_m_per_unit
        metrics["max_wrist_speed_kmh"] = max_wrist_speed_mps * 3.6
        metrics["est_racquet_head_speed_kmh"] = (max_wrist_speed_mps + (max_wrist_speed_mps / 0.7) * RACQUET_LENGTH_M) * 3.6

    shoulder_traj = shot_data[:, [2*shoulder_kp_idx, 2*shoulder_kp_idx+1]]
    vec_start = wrist_traj[0] - shoulder_traj[0]
    vec_end = wrist_traj[-1] - shoulder_traj[-1]
    angle_start = np.arctan2(vec_start[1], vec_start[0])
    angle_end = np.arctan2(vec_end[1], vec_end[0])
    sweep_angle = abs(np.rad2deg(angle_end - angle_start))
    metrics["swing_arc_degrees"] = sweep_angle if sweep_angle < 270 else 360 - sweep_angle

    return metrics

def main():
    parser = argparse.ArgumentParser(description="Process a tennis video for shot classification and analysis.")
    parser.add_argument("video_path", type=str, help="Path to the input video file.")
    parser.add_argument("model_path", type=str, help="Path to the trained Keras model file.")
    parser.add_argument("--height", type=float, default=None, help="Player height in meters for realistic metrics.")
    parser.add_argument("--threshold", type=float, default=0.98, help="Confidence threshold for shot detection.")
    args = parser.parse_args()

    # --- Setup ---
    # Create output directory
    video_name = os.path.splitext(os.path.basename(args.video_path))[0]
    output_dir = f"{video_name}_analysis"
    gif_dir = os.path.join(output_dir, "shots_gifs")
    metrics_dir = os.path.join(output_dir, "shots_metrics")
    os.makedirs(gif_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Load model and class names (assuming they are saved with the model or known)
    model = keras.models.load_model(args.model_path, compile=False)
    # This needs to be adjusted based on how you save your model/classes
    class_names = ['backhand', 'backhand_slice', 'backhand_volley', 'forehand', 'forehand_volley', 'neutral', 'serve']

    # Initialize components
    pose_extractor = HumanPoseExtractor()
    shot_detector = ShotDetector(model, class_names, args.threshold)
    annotator = VideoAnnotator(class_names)
    
    cap = cv2.VideoCapture(args.video_path)
    fps = 60
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    output_video_path = os.path.join(output_dir, f"{video_name}_annotated.mp4")
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    roi = RoI((height, width))
    frame_buffer = deque(maxlen=SLIDING_WINDOW_SIZE)
    
    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    # --- Main Loop ---
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        pbar.update(1)
        frame_buffer.append(frame.copy())
        
        # Pose Estimation
        roi_frame = roi.extract_subframe(frame)
        keypoints_norm = pose_extractor.run_inference(roi_frame)
        
        # Convert keypoints to original frame coordinates
        keypoints_frame = keypoints_norm.copy()
        keypoints_frame[:, 0] = roi.center_y - roi.height // 2 + keypoints_frame[:, 0] * roi.height
        keypoints_frame[:, 1] = roi.center_x - roi.width // 2 + keypoints_frame[:, 1] * roi.width
        
        roi.update(keypoints_frame)
        
        # Shot Detection
        detected_shot_info = shot_detector.process_keypoints(keypoints_norm, frame_buffer)
        
        if detected_shot_info:
            print("Shot detected")
            # Analyze, save, and update annotations
            metrics = analyze_biomechanics(detected_shot_info["shot_sequence_data"], detected_shot_info["shot_name"], fps, args.height)
            annotator.update_shot_data(detected_shot_info, metrics)
            
            # Save GIF
            gif_path = os.path.join(gif_dir, f'shot_{detected_shot_info["shot_number"]:03d}_{detected_shot_info["shot_name"]}.gif')
            imageio.mimsave(gif_path, detected_shot_info["shot_sequence_frames"], fps=15)
            
            # Save Metrics
            metrics_path = os.path.join(metrics_dir, f'shot_{detected_shot_info["shot_number"]:03d}_{detected_shot_info["shot_name"]}_metrics.txt')
            with open(metrics_path, 'w') as f:
                f.write(f"Shot Type: {detected_shot_info['shot_name']}\n")
                for key, val in metrics.items():
                    f.write(f"{key.replace('_', ' ').title()}: {val:.2f}\n")

        # Draw final frame
        annotated_frame = annotator.draw_frame(frame, roi, keypoints_frame, shot_detector.last_probs)
        out.write(annotated_frame)

    # --- Cleanup ---
    pbar.close()
    cap.release()
    out.release()
    print(f"\nProcessing complete. Annotated video and analysis saved to '{output_dir}'")

if __name__ == "__main__":
    main()
