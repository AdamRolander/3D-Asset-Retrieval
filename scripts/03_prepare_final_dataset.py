import os
import pandas as pd
import sys
from pathlib import Path
import zipfile
from tqdm import tqdm # For a nice progress bar

# --- Configuration ---
# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).resolve().parent 
# Get the project root directory
PROJECT_ROOT = SCRIPT_DIR.parent 

# --- Input Paths ---
# The metadata file we created in script 02
METADATA_IN_PATH = PROJECT_ROOT / 'data' / 'processed' / 'metadata_10k.csv'
# The folder with all the individual asset zips
RAW_ZIPS_DIR = PROJECT_ROOT / 'data' / 'raw' / 'all_asset_zips'

# --- Output Paths ---
# The final, clean metadata file
METADATA_OUT_PATH = PROJECT_ROOT / 'data' / 'processed' / 'final_metadata_10k.csv'
# The new folder where we'll save the extracted .png files
IMAGES_OUT_DIR = PROJECT_ROOT / 'data' / 'processed' / 'images'
# --- End Configuration ---

def extract_renders():
    """
    1. Loads the 10k metadata CSV.
    2. Creates the 'data/processed/images/' directory.
    3. Loops through each asset, unzipping its '000.png' render.
    4. Saves the render to the new images directory.
    5. Saves a new CSV file with an 'image_path' column.
    """
    
    # --- Step 1: Load Metadata ---
    print(f"Loading metadata from {METADATA_IN_PATH}...")
    if not METADATA_IN_PATH.exists():
        print(f"Error: File not found: {METADATA_IN_PATH}", file=sys.stderr)
        print("Please run 'scripts/02_create_processing_subset.py' first.", file=sys.stderr)
        sys.exit(1)
        
    try:
        df = pd.read_csv(METADATA_IN_PATH)
    except Exception as e:
        print(f"Error reading CSV: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Step 2: Create Output Directory ---
    print(f"Creating output image directory at {IMAGES_OUT_DIR}...")
    IMAGES_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Step 3: Loop, Extract, and Save ---
    print(f"Extracting {len(df)} images. This may take a few minutes...")
    
    processed_data = [] # To store our new metadata
    
    # Use tqdm to create a progress bar
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing assets"):
        uid = row['uid']
        caption = row['caption']
        
        source_zip_path = RAW_ZIPS_DIR / f"{uid}.zip"
        # We'll save the image as a .png, which is cleaner
        target_image_path = IMAGES_OUT_DIR / f"{uid}.png"
        
        # We'll store the *relative* path in the CSV for portability
        relative_image_path = target_image_path.relative_to(PROJECT_ROOT)

        try:
            # Open the source .zip file for this asset
            with zipfile.ZipFile(source_zip_path, 'r') as asset_zip:
                # Extract the '000.png' file from the zip *in memory*
                with asset_zip.open('00000.png') as render_file:
                    # Write the in-memory file to our new target .png file
                    with open(target_image_path, 'wb') as out_f:
                        out_f.write(render_file.read())
            
            # If successful, add this to our new metadata list
            processed_data.append({
                'uid': uid,
                'caption': caption,
                'image_path': str(relative_image_path) # Store the portable path
            })

        except FileNotFoundError:
            print(f"\nWarning: Zip file not found for {uid}. Skipping.", file=sys.stderr)
        except KeyError:
            print(f"\nWarning: '000.png' not found in {uid}.zip. Skipping.", file=sys.stderr)
        except Exception as e:
            print(f"\nError processing {uid}: {e}. Skipping.", file=sys.stderr)

    # --- Step 4: Save Final Metadata CSV ---
    print("\nExtraction complete.")
    
    if not processed_data:
        print("Error: No data was processed successfully.", file=sys.stderr)
        sys.exit(1)

    # Convert our list of dicts to a new DataFrame
    df_final = pd.DataFrame(processed_data)
    
    # Save the new, final CSV
    df_final.to_csv(METADATA_OUT_PATH, index=False)
    
    print(f"Successfully processed {len(df_final)} assets.")
    print(f"Final metadata with image paths saved to: {METADATA_OUT_PATH}")
    print("\n--- Sample of the final data ---")
    print(df_final.head())
    print("----------------------------------")

if __name__ == '__main__':
    extract_renders()