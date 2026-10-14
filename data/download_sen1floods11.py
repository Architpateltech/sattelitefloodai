import os
import sys
from google.cloud import storage
from google.api_core import exceptions

def download_sen1floods11_handlabeled(target_dir):
    """
    Downloads the high-quality 'HandLabeled' subset of Sen1Floods11.
    This subset contains pixel-level flood masks and is less than 2 GB.
    """
    os.makedirs(target_dir, exist_ok=True)
    
    # Initialize anonymous GCS client
    client = storage.Client.create_anonymous_client()
    bucket_name = "sen1floods11"
    
    try:
        bucket = client.bucket(bucket_name)
        # We only want the v1.1 hand-labeled data
        prefix = "v1.1/data/flood_events/HandLabeled/"
        
        print(f"Listing files in gs://{bucket_name}/{prefix}...")
        blobs = list(bucket.list_blobs(prefix=prefix))
        
        if not blobs:
            print("No files found. The bucket structure might have changed.")
            return
            
        total_files = len(blobs)
        total_size_mb = sum(b.size for b in blobs if b.size) / (1024 * 1024)
        
        print(f"Found {total_files} files (Total size: {total_size_mb:.2f} MB).")
        print("Starting download...\n")
        
        downloaded = 0
        for i, blob in enumerate(blobs):
            # Skip directories
            if blob.name.endswith('/'):
                continue
                
            # Create local folder structure
            relative_path = blob.name.replace(prefix, "")
            local_path = os.path.join(target_dir, relative_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Download file if it doesn't exist
            if not os.path.exists(local_path):
                print(f"[{i+1}/{total_files}] Downloading {relative_path} ({blob.size / 1024 / 1024:.2f} MB)...")
                blob.download_to_filename(local_path)
            else:
                pass # Already downloaded
                
            downloaded += 1
            
        print("\nDownload complete!")
        print(f"Sen1Floods11 (HandLabeled subset) is ready at: {target_dir}")
        
    except exceptions.NotFound:
        print(f"Bucket {bucket_name} not found or is not public.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    target = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "Sen1Floods11_HandLabeled")
    download_sen1floods11_handlabeled(target)
