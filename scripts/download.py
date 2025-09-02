import supervisely as sly
import argparse

def main():
    parser = argparse.ArgumentParser(description="Download file from Supervisely storage.")
    parser.add_argument("--team_id", type=int, default=3, help="Team ID in Supervisely")
    parser.add_argument("--source_path", type=str, required=True, help="Path to file in Supervisely storage")
    parser.add_argument("--local_path", type=str, required=True, help="Path to save file locally")
    args = parser.parse_args()

    api = sly.Api()
    api.file.download(args.team_id, args.source_path, args.local_path)
    print(f"✅ File downloaded to {args.local_path}")

if __name__ == "__main__":
    main()
