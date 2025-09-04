#!/usr/bin/env python3
"""
Batch process videos and annotations to extract shots as features.
Matches video files with annotation files based on filename and calls
extract_shots_as_features.py for each matched pair.
"""

import os
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path


def find_video_files(videos_dir):
    """Find all video files in the directory."""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
    video_files = []
    
    videos_path = Path(videos_dir)
    for file_path in videos_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in video_extensions:
            video_files.append(file_path)
    
    return sorted(video_files)


def find_annotation_files(annotations_dir):
    """Find all annotation files (CSV) in the directory."""
    annotations_path = Path(annotations_dir)
    annotation_files = {}
    
    for file_path in annotations_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() == '.csv':
            # Use stem (filename without extension) as key
            annotation_files[file_path.stem] = file_path
    
    return annotation_files


def match_files(video_files, annotation_files):
    """Match video files with annotation files based on filename."""
    matches = []
    unmatched_videos = []
    
    for video_file in video_files:
        video_stem = video_file.stem
        # Look for annotation file named "annotation_videoname"
        annotation_key = f"annotation_{video_stem}"
        if annotation_key in annotation_files:
            matches.append((video_file, annotation_files[annotation_key]))
        else:
            unmatched_videos.append(video_file)
    
    return matches, unmatched_videos


def run_extraction(video_file, annotation_file, output_dir, show=False, debug=False):
    """Run extract_shots_as_features.py for a video-annotation pair."""
    cmd = [
        sys.executable,  # Use the same Python interpreter
        'extract_shots_as_features.py',
        str(video_file),
        str(annotation_file),
        str(output_dir)
    ]
    
    if show:
        cmd.append('--show')
    if debug:
        cmd.append('--debug')
    
    print(f"Processing: {video_file.name} with {annotation_file.name}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ Successfully processed {video_file.name}")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error processing {video_file.name}:")
        print(f"Error code: {e.returncode}")
        if e.stdout:
            print(f"Stdout: {e.stdout}")
        if e.stderr:
            print(f"Stderr: {e.stderr}")
        return False


def main():
    parser = ArgumentParser(
        description="Batch extract shots from videos using corresponding annotations"
    )
    parser.add_argument("videos_dir", help="Directory containing video files")
    parser.add_argument("annotations_dir", help="Directory containing annotation CSV files")
    parser.add_argument("output_dir", help="Output directory for extracted features")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show frames during processing (passed to extract_shots_as_features.py)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode (passed to extract_shots_as_features.py)"
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing other files if one fails"
    )
    
    args = parser.parse_args()
    
    # Validate directories
    videos_dir = Path(args.videos_dir)
    annotations_dir = Path(args.annotations_dir)
    output_dir = Path(args.output_dir)
    
    if not videos_dir.exists():
        print(f"Error: Videos directory '{videos_dir}' does not exist")
        sys.exit(1)
    
    if not annotations_dir.exists():
        print(f"Error: Annotations directory '{annotations_dir}' does not exist")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find files
    print("Finding video files...")
    video_files = find_video_files(videos_dir)
    print(f"Found {len(video_files)} video files")
    
    print("Finding annotation files...")
    annotation_files = find_annotation_files(annotations_dir)
    print(f"Found {len(annotation_files)} annotation files")
    
    # Match files
    matches, unmatched_videos = match_files(video_files, annotation_files)
    
    print(f"\nMatched {len(matches)} video-annotation pairs:")
    for video_file, annotation_file in matches:
        print(f"  {video_file.name} ↔ {annotation_file.name}")
    
    if unmatched_videos:
        print(f"\nUnmatched videos ({len(unmatched_videos)}):")
        for video_file in unmatched_videos:
            print(f"  {video_file.name}")
    
    if not matches:
        print("No matching video-annotation pairs found!")
        sys.exit(1)
    
    # Process each match
    print(f"\nProcessing {len(matches)} pairs...")
    successful = 0
    failed = 0
    
    for i, (video_file, annotation_file) in enumerate(matches, 1):
        print(f"\n--- Processing {i}/{len(matches)} ---")
        
        success = run_extraction(
            video_file, 
            annotation_file, 
            output_dir, 
            show=args.show, 
            debug=args.debug
        )
        
        if success:
            successful += 1
        else:
            failed += 1
            if not args.continue_on_error:
                print("Stopping due to error. Use --continue-on-error to process remaining files.")
                break
    
    print(f"\n--- Summary ---")
    print(f"Successfully processed: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {len(matches)}")
    
    if failed > 0 and not args.continue_on_error:
        sys.exit(1)


if __name__ == "__main__":
    main() 