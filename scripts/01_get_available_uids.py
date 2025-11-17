import os
import sys

# --- Configuration ---
# TODO: Change this to the path where you unzipped the 42GB file
# It should be a folder containing thousands of .zip files (e.g., 'yFocwO8x2UiIQYWxt2HtH1x1k.zip')
RAW_ZIPS_DIR = 'data/raw/all_asset_zips' 

# This is where we will save our list of UIDs
OUTPUT_FILE = 'data/available_uids.txt'
# --- End Configuration ---

def get_available_uids():
    """
    Scans the RAW_ZIPS_DIR for all .zip files and saves their filenames
    (without the .zip extension) to the OUTPUT_FILE.
    """
    print(f"Scanning directory: {RAW_ZIPS_DIR}")
    
    if not os.path.exists(RAW_ZIPS_DIR) or not os.path.isdir(RAW_ZIPS_DIR):
        print(f"Error: Directory not found: {RAW_ZIPS_DIR}", file=sys.stderr)
        print("Please download and unzip the 'compressed_imgs_perobj_000.zip' file into that directory.", file=sys.stderr)
        sys.exit(1)

    available_uids = []
    
    # Iterate over all files in the directory
    for filename in os.listdir(RAW_ZIPS_DIR):
        if filename.endswith('.zip'):
            # Remove the '.zip' extension to get the UID
            uid = filename[:-4]
            available_uids.append(uid)
            
    if not available_uids:
        print(f"Warning: No .zip files found in {RAW_ZIPS_DIR}", file=sys.stderr)
        return

    print(f"Found {len(available_uids)} available asset UIDs.")

    # Ensure the output directory (data/) exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # Write the UIDs to the output file, one per line
    with open(OUTPUT_FILE, 'w') as f:
        for uid in available_uids:
            f.write(f"{uid}\n")

    print(f"Successfully saved UID list to {OUTPUT_FILE}")

if __name__ == '__main__':
    get_available_uids()