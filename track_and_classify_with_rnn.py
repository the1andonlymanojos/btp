"""
With this script, you can provide a video and your RNN model (e.g tennis_rnn.h5)
and see a shot classification/detection.For this, we feed our neural network with
a sliding window of 30 frame (1 second) and classify the shot.
Same kind of shot counter is used then.
"""

import time
from argparse import ArgumentParser
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
import cv2
from tcn import TCN
from extract_human_pose import HumanPoseExtractor


class ShotCounter:
    """
    Pretty much the same principle than in track_and_classify_frame_by_frame
    except that we dont have any history here, and confidence threshold can be much higher.
    """

    MIN_FRAMES_BETWEEN_SHOTS = 60

    def __init__(self, threshold=None):
        self.nb_history = 30
        self.probs = np.zeros(7)  # Updated to 7 classes

        self.nb_forehands = 0
        self.nb_backhands = 0
        self.nb_serves = 0
        self.nb_backhand_slices = 0
        self.nb_backhand_volleys = 0
        self.nb_forehand_volleys = 0

        self.last_shot = "neutral"
        self.frames_since_last_shot = self.MIN_FRAMES_BETWEEN_SHOTS

        self.results = []
        
        # Set threshold - use provided value or default thresholds
        if threshold is not None:
            # Use the same threshold for all shot types
            self.thresholds = {
                0: threshold,  # backhand
                1: threshold,  # backhand_slice
                2: threshold,  # backhand_volley
                3: threshold,  # forehand
                4: threshold,  # forehand_volley
                6: threshold,  # serve
            }
        else:
            # Use default thresholds
            self.thresholds = {
                0: 0.98,  # backhand
                1: 0.97,  # backhand_slice
                2: 0.97,  # backhand_volley
                3: 0.98,  # forehand
                4: 0.97,  # forehand_volley
                6: 0.98,  # serve (model is confident here)
            }

    def update(self, probs, frame_id):
        """Update current state with shot probabilities"""
        shot_names = [
            "backhand", "backhand_slice", "backhand_volley",
            "forehand", "forehand_volley", "overhead", "serve"
        ]

        if len(probs) == 7:
            self.probs = probs
        else:
            # Handle cases where model outputs different number of classes
            self.probs[:len(probs)] = probs

        for i, prob in enumerate(probs):
            if prob > self.thresholds.get(i, 0.98) and self.frames_since_last_shot > self.MIN_FRAMES_BETWEEN_SHOTS:
                self.last_shot = shot_names[i]
                self.frames_since_last_shot = 0
                self.results.append({"FrameID": frame_id, "Shot": self.last_shot})
                print(f"Shot detected: {self.last_shot} at frame {frame_id}")

                if i == 0: self.nb_backhands += 1
                elif i == 1: self.nb_backhand_slices += 1
                elif i == 2: self.nb_backhand_volleys += 1
                elif i == 3: self.nb_forehands += 1
                elif i == 4: self.nb_forehand_volleys += 1
                elif i == 6: self.nb_serves += 1
                break  # only fire one per frame

        self.frames_since_last_shot += 1

    def display(self, frame):
        """Display counter"""
        y_positions = [frame.shape[0] - 160, frame.shape[0] - 130, frame.shape[0] - 100, 
                      frame.shape[0] - 70, frame.shape[0] - 40, frame.shape[0] - 10]
        
        shots_info = [
            (f"Backhands = {self.nb_backhands}", "backhand"),
            (f"Backhand Slices = {self.nb_backhand_slices}", "backhand_slice"),
            (f"Backhand Volleys = {self.nb_backhand_volleys}", "backhand_volley"),
            (f"Forehands = {self.nb_forehands}", "forehand"),
            (f"Forehand Volleys = {self.nb_forehand_volleys}", "forehand_volley"),
            (f"Serves = {self.nb_serves}", "serve")
        ]
        
        for i, (text, shot_type) in enumerate(shots_info):
            cv2.putText(
                frame,
                text,
                (20, y_positions[i]),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.7,
                color=(0, 255, 0)
                if (self.last_shot == shot_type and self.frames_since_last_shot < 30)
                else (0, 0, 255),
                thickness=2,
            )


BAR_WIDTH = 25
BAR_HEIGHT = 150
MARGIN_ABOVE_BAR = 30
SPACE_BETWEEN_BARS = 35
TEXT_ORIGIN_X = 950
BAR_ORIGIN_X = 945


def draw_probs(frame, probs):
    """Draw vertical bars representing probabilities"""
    
    # Ensure we have 7 probabilities
    if len(probs) < 7:
        probs = np.pad(probs, (0, 7 - len(probs)), 'constant')
    
    # Labels for each shot type (shortened to fit)
    labels = ["BH", "BS", "BV", "FH", "FV", "N", "S"]
    
    # Colors for each bar (BGR format)
    colors = [
        (255, 0, 0),    # Blue for backhand
        (255, 100, 0),  # Light blue for backhand slice
        (255, 200, 0),  # Cyan for backhand volley
        (0, 255, 0),    # Green for forehand
        (0, 255, 100),  # Light green for forehand volley
        (128, 128, 128), # Gray for neutral
        (0, 0, 255)     # Red for serve
    ]
    
    # Mapping of probabilities to correct indices
    # probs order: [backhand, backhand_slice, backhand_volley, forehand, forehand_volley, neutral, serve]
    prob_indices = [0, 1, 2, 3, 4, 5, 6]
    
    for i, (label, color, prob_idx) in enumerate(zip(labels, colors, prob_indices)):
        x_pos = TEXT_ORIGIN_X + SPACE_BETWEEN_BARS * i
        bar_x_pos = BAR_ORIGIN_X + SPACE_BETWEEN_BARS * i
        
        # Draw label
        cv2.putText(
            frame,
            label,
            (x_pos, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.6,
            color=(0, 0, 255),
            thickness=2,
        )
        
        # Draw probability bar
        bar_height = int(BAR_HEIGHT * probs[prob_idx])
        cv2.rectangle(
            frame,
            (bar_x_pos, BAR_HEIGHT + MARGIN_ABOVE_BAR - bar_height),
            (bar_x_pos + BAR_WIDTH, BAR_HEIGHT + MARGIN_ABOVE_BAR),
            color=color,
            thickness=-1,
        )
        
        # Draw white border around bar
        cv2.rectangle(
            frame,
            (bar_x_pos, MARGIN_ABOVE_BAR),
            (bar_x_pos + BAR_WIDTH, BAR_HEIGHT + MARGIN_ABOVE_BAR),
            color=(255, 255, 255),
            thickness=1,
        )

    return frame


class GT:
    """GT to optionnally assess your results"""

    def __init__(self, path_to_annotation):
        self.shots = pd.read_csv(path_to_annotation)
        self.current_row_in_shots = 0
        self.nb_backhands = 0
        self.nb_forehands = 0
        self.nb_serves = 0
        self.nb_backhand_slices = 0
        self.nb_backhand_volleys = 0
        self.nb_forehand_volleys = 0
        self.last_shot = "neutral"
        
        # Debug: Print total GT counts for each class
        self._print_gt_totals()
    
    def _print_gt_totals(self):
        """Print total GT counts for each shot class"""
        print("\n" + "="*50)
        print("GROUND TRUTH SHOT COUNTS:")
        print("="*50)
        
        # Print all values in self.shots
        print("Complete Ground Truth Data:")
        print(self.shots.to_string())
        print("\n" + "-"*50)
        
        # Count total shots for each class
        total_backhands = len(self.shots[self.shots['Shot'] == 'backhand'])
        total_forehands = len(self.shots[self.shots['Shot'] == 'forehand'])
        total_serves = len(self.shots[self.shots['Shot'] == 'overhead'])
        total_backhand_slices = len(self.shots[self.shots['Shot'] == 'backhand-slice'])
        total_backhand_volleys = len(self.shots[self.shots['Shot'] == 'backhand-volley'])
        total_forehand_volleys = len(self.shots[self.shots['Shot'] == 'forehand-volley'])
        
        print("Summary by Shot Type:")
        print(f"Total Backhands: {total_backhands}")
        print(f"Total Forehands: {total_forehands}")
        print(f"Total Serves: {total_serves}")
        print(f"Total Backhand Slices: {total_backhand_slices}")
        print(f"Total Backhand Volleys: {total_backhand_volleys}")
        print(f"Total Forehand Volleys: {total_forehand_volleys}")
        print(f"Total Shots: {len(self.shots)}")
        print("="*50 + "\n")

    def display(self, frame, frame_id):
        """Display shot counter"""
        if self.current_row_in_shots < len(self.shots):
            if frame_id == self.shots.iloc[self.current_row_in_shots]["FrameId"]:
                shot_type = self.shots.iloc[self.current_row_in_shots]["Shot"]
                if shot_type == "backhand":
                    self.nb_backhands += 1
                elif shot_type == "forehand":
                    self.nb_forehands += 1
                elif shot_type == "serve":
                    self.nb_serves += 1
                elif shot_type == "backhand_slice":
                    self.nb_backhand_slices += 1
                elif shot_type == "backhand_volley":
                    self.nb_backhand_volleys += 1
                elif shot_type == "forehand_volley":
                    self.nb_forehand_volleys += 1
                self.last_shot = shot_type
                self.current_row_in_shots += 1

        # Display counters on the right side of the frame
        y_positions = [frame.shape[0] - 160, frame.shape[0] - 130, frame.shape[0] - 100, 
                      frame.shape[0] - 70, frame.shape[0] - 40, frame.shape[0] - 10]
        
        shots_info = [
            (f"GT BH = {self.nb_backhands}", "backhand"),
            (f"GT BS = {self.nb_backhand_slices}", "backhand_slice"),
            (f"GT BV = {self.nb_backhand_volleys}", "backhand_volley"),
            (f"GT FH = {self.nb_forehands}", "forehand"),
            (f"GT FV = {self.nb_forehand_volleys}", "forehand_volley"),
            (f"GT S = {self.nb_serves}", "serve")
        ]
        
        for i, (text, shot_type) in enumerate(shots_info):
            cv2.putText(
                frame,
                text,
                (frame.shape[1] - 200, y_positions[i]),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.6,
                color=(0, 255, 0) if self.last_shot == shot_type else (0, 0, 255),
                thickness=2,
            )


def draw_fps(frame, fps):
    """Draw fps to demonstrate performance"""
    cv2.putText(
        frame,
        f"{int(fps)} fps",
        (20, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.8,
        color=(0, 165, 255),
        thickness=2,
    )


def draw_frame_id(frame, frame_id):
    """Used for debugging purpose"""
    cv2.putText(
        frame,
        f"Frame {frame_id}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.8,
        color=(0, 165, 255),
        thickness=2,
    )


def compute_recall_precision(gt, shots):
    """
    Assess your results against a Groundtruth
    like number of misses (recall) and number of false positives (precision)
    """

    gt_numpy = gt.to_numpy()
    nb_match = 0
    nb_misses = 0
    nb_fp = 0
    fp_backhands = 0
    fp_forehands = 0
    fp_serves = 0
    fp_backhand_slices = 0
    fp_backhand_volleys = 0
    fp_forehand_volleys = 0
    
    for gt_shot in gt_numpy:
        found_match = False
        for shot in shots:
            if shot["Shot"] == gt_shot[0]:
                if abs(shot["FrameID"] - gt_shot[1]) <= 30:
                    found_match = True
                    break
        if found_match:
            nb_match += 1
        else:
            nb_misses += 1

    for shot in shots:
        found_match = False
        for gt_shot in gt_numpy:
            if shot["Shot"] == gt_shot[0]:
                if abs(shot["FrameID"] - gt_shot[1]) <= 30:
                    found_match = True
                    break
        if not found_match:
            nb_fp += 1
            if shot["Shot"] == "backhand":
                fp_backhands += 1
            elif shot["Shot"] == "forehand":
                fp_forehands += 1
            elif shot["Shot"] == "serve":
                fp_serves += 1
            elif shot["Shot"] == "backhand_slice":
                fp_backhand_slices += 1
            elif shot["Shot"] == "backhand_volley":
                fp_backhand_volleys += 1
            elif shot["Shot"] == "forehand_volley":
                fp_forehand_volleys += 1

    precision = nb_match / (nb_match + nb_fp) if (nb_match + nb_fp) > 0 else 0
    recall = nb_match / (nb_match + nb_misses) if (nb_match + nb_misses) > 0 else 0

    print(f"Recall {recall*100:.1f}%")
    print(f"Precision {precision*100:.1f}%")

    print(f"FP: BH = {fp_backhands}, FH = {fp_forehands}, S = {fp_serves}")
    print(f"FP: BS = {fp_backhand_slices}, BV = {fp_backhand_volleys}, FV = {fp_forehand_volleys}")


if __name__ == "__main__":
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # or 'XVID' for .avi
    out = cv2.VideoWriter("output_newold.mp4", fourcc, 60.0, (1280, 720))

    parser = ArgumentParser(
        description="Track tennis player and display shot probabilities"
    )
    parser.add_argument("video")
    parser.add_argument("model")
    parser.add_argument("--evaluate", help="Path to annotation file")
    parser.add_argument("-f", type=int, help="Forward to")
    parser.add_argument(
        "--left-handed",
        action="store_const",
        const=True,
        default=False,
        help="If player is left-handed",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Confidence threshold for shot detection (0.0-1.0). If not provided, uses default thresholds for each shot type"
    )
    args = parser.parse_args()

    shot_counter = ShotCounter(threshold=args.threshold)

    if args.evaluate is not None:
        gt = GT(args.evaluate)

    m1 = keras.models.load_model(args.model, custom_objects={"TCN": TCN})

    cap = cv2.VideoCapture(args.video)

    assert cap.isOpened()

    # Get total frame count for progress tracking
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames to process: {total_frames}")

    ret, frame = cap.read()

    human_pose_extractor = HumanPoseExtractor(frame.shape)

    NB_IMAGES = 30

    FRAME_ID = 0

    features_pool = []

    prev_time = time.time()
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        FRAME_ID += 1

        # Progress tracking every 100 frames
        if FRAME_ID % 100 == 0:
            elapsed_time = time.time() - start_time
            progress_percent = (FRAME_ID / total_frames) * 100
            frames_per_sec = FRAME_ID / elapsed_time
            estimated_total_time = total_frames / frames_per_sec
            estimated_remaining = estimated_total_time - elapsed_time
            
            print(f"Progress: {FRAME_ID}/{total_frames} frames ({progress_percent:.1f}%) | "
                  f"Processing speed: {frames_per_sec:.1f} fps | "
                  f"ETA: {estimated_remaining/60:.1f} minutes")

        if args.f is not None and FRAME_ID < args.f:
            continue

        assert frame is not None

        human_pose_extractor.extract(frame)

        # if not human_pose_extractor.roi.valid:
        #    features_pool = []
        #    continue

        # dont draw non-significant points/edges by setting probability to 0
        human_pose_extractor.discard(["left_eye", "right_eye", "left_ear", "right_ear"])

        features = human_pose_extractor.keypoints_with_scores.reshape(17, 3)

        if args.left_handed:
            features[:, 1] = 1 - features[:, 1]

        features = features[features[:, 2] > 0][:, 0:2].reshape(1, 13 * 2)

        features_pool.append(features)
        # print(features_pool)

        if len(features_pool) == NB_IMAGES:
            features_seq = np.array(features_pool).reshape(1, NB_IMAGES, 26)
            assert features_seq.shape == (1, 30, 26)
            probs = m1.__call__(features_seq)[0]
            shot_counter.update(probs, FRAME_ID)

            # Give space to pool
            features_pool = features_pool[1:]

        draw_probs(frame, shot_counter.probs)
        shot_counter.display(frame)

        if args.evaluate is not None:
            gt.display(frame, FRAME_ID)

        fps = 1 / (time.time() - prev_time)
        prev_time = time.time()
        draw_fps(frame, fps)
        draw_frame_id(frame, FRAME_ID)

        # Display results on original frame
        human_pose_extractor.draw_results_frame(frame)
        if (
            shot_counter.frames_since_last_shot < 30
            and shot_counter.last_shot != "neutral"
        ):
            human_pose_extractor.roi.draw_shot(frame, shot_counter.last_shot)

        # cv2.imshow("Frame", frame)
        human_pose_extractor.roi.update(human_pose_extractor.keypoints_pixels_frame)

        out.write(frame)


        # k = cv2.waitKey(0)
        # if k == 27:
        #    break

    # Final progress update
    total_time = time.time() - start_time
    print(f"\nProcessing complete! Total time: {total_time/60:.1f} minutes")
    print(f"Average processing speed: {total_frames/total_time:.1f} fps")

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(shot_counter.results)

    if args.evaluate is not None:
        compute_recall_precision(gt.shots, shot_counter.results)
