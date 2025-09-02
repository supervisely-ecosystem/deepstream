import supervisely as sly
import argparse

def main():
    parser = argparse.ArgumentParser(description="Download video from Supervisely storage.")
    parser.add_argument("--video_id", type=int, help="Video ID in Supervisely")
    parser.add_argument("--local_path", type=str, required=True, help="Path to save file locally")
    args = parser.parse_args()

    api = sly.Api()
    api.video.download_path(args.video_id, args.local_path)
    print(f"✅ File downloaded to {args.local_path}")

if __name__ == "__main__":
    main()
