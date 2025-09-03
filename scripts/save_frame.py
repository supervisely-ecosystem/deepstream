import cv2
import os
import argparse

def save_frame(video_path, frame_number, output_name=None):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Failed to open the video")

    # Set the position to the desired frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # Read the frame
    ret, frame = cap.read()
    if not ret:
        raise ValueError(f"Failed to read frame {frame_number}")

    # Get the folder where the video is located
    folder = os.path.dirname(video_path)

    # Build output filename
    if output_name is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        output_name = f"{base}_frame{frame_number}.jpg"

    output_path = os.path.join(folder, output_name)

    # Save the frame as an image
    cv2.imwrite(output_path, frame)
    cap.release()

    return output_path


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Save a specific frame from a video.")
    parser.add_argument("--video", type=str, required=True, help="Path to the video file")
    parser.add_argument("--frame", type=int, required=True, help="Frame number to extract")
    parser.add_argument("--output", type=str, default=None, help="Optional output image name")
    args = parser.parse_args()

    # Call the save_frame function
    saved_path = save_frame(args.video, args.frame, args.output)
    print(f"Frame saved to: {saved_path}")


if __name__ == "__main__":
    main()
