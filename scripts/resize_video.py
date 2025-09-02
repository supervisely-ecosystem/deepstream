import cv2
import argparse

def main():
    parser = argparse.ArgumentParser(description="Resize video to given size.")
    parser.add_argument("--input", type=str, required=True, help="Path to input video (e.g., input.avi)")
    parser.add_argument("--output", type=str, required=True, help="Path to save resized video (e.g., output.avi)")
    parser.add_argument("--size", type=int, nargs=2, default=[640, 640], help="Target size as width height (default: 640 640)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.input)

    if not cap.isOpened():
        raise ValueError(f"❌ Cannot open video file: {args.input}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(args.output, fourcc, fps, (args.size[0], args.size[1]))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        resized = cv2.resize(frame, (args.size[0], args.size[1]))
        out.write(resized)

    cap.release()
    out.release()
    print(f"✅ Video saved to {args.output} with size {args.size[0]}x{args.size[1]}")

if __name__ == "__main__":
    main()
