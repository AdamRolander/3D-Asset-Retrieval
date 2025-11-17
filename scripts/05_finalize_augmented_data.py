import pandas as pd
import sys
from pathlib import Path
import json

# --- Configuration ---
SCRIPT_DIR = Path(__file__).resolve().parent 
PROJECT_ROOT = SCRIPT_DIR.parent 

PROGRESS_FILE = PROJECT_ROOT / 'data' / 'processed' / 'augmented_paraphrases.jsonl'
FINAL_CSV_PATH = PROJECT_ROOT / 'data' / 'processed' / 'final_metadata_augmented.csv'
# --- End Configuration ---

def finalize_dataset():
    """
    Reads the .jsonl progress file and "unpacks" it into a clean
    CSV file, ready for the PyTorch/Colab data loader.
    """
    
    if not PROGRESS_FILE.exists():
        print(f"Error: Progress file not found: {PROGRESS_FILE}", file=sys.stderr)
        print("Please run '04_augment_paraphrase.py' first.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading data from {PROGRESS_FILE}...")
    
    final_rows = []
    
    with open(PROGRESS_FILE, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                uid = data['uid']
                image_path = data['image_path']
                
                # 1. Add the original caption as a row
                final_rows.append({
                    "uid": uid,
                    "image_path": image_path,
                    "caption": data['original_caption'],
                    "is_paraphrase": False
                })
                
                # 2. Add all the paraphrases as rows
                for paraphrase in data['paraphrases']:
                    final_rows.append({
                        "uid": uid,
                        "image_path": image_path,
                        "caption": paraphrase,
                        "is_paraphrase": True
                    })
                    
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Skipping corrupt line in .jsonl: {e}", file=sys.stderr)
                continue
                
    if not final_rows:
        print("Error: No data was loaded. The .jsonl file might be empty or corrupt.", file=sys.stderr)
        sys.exit(1)

    # Convert the list of dicts into the final DataFrame
    df_final = pd.DataFrame(final_rows)
    
    # Save to CSV
    df_final.to_csv(FINAL_CSV_PATH, index=False)
    
    print(f"\nSuccessfully created final dataset with {len(df_final)} total pairs.")
    print(f"Saved to: {FINAL_CSV_PATH}")
    print("\n--- Sample of the final data ---")
    print(df_final.head(7)) # Show 7 rows to see the pattern
    print("----------------------------------")

if __name__ == "__main__":
    finalize_dataset()