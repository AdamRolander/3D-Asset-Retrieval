import os
import pandas as pd
import sys
from pathlib import Path
import time
import json
import csv  # Added for CSV writing
import requests
from tqdm import tqdm
from dotenv import load_dotenv

# --- Configuration ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Input: The full augmented dataset
AUGMENTED_METADATA_PATH = PROJECT_ROOT / 'data' / 'processed' / 'final_metadata_augmented.csv'
# Intermediate: The test split (subset of unseen assets)
TEST_SPLIT_PATH = PROJECT_ROOT / 'data' / 'processed' / 'test_split.csv'
# Output: The final adversarial test dataset
OUTPUT_ADVERSARIAL_PATH = PROJECT_ROOT / 'data' / 'processed' / 'adversarial_test_set.csv'
# Progress file for resumability (Changed to .csv)
PROGRESS_FILE = PROJECT_ROOT / 'data' / 'processed' / 'adversarial_generation_progress.csv'

# Constants
TEST_SET_SIZE = 500 # Number of assets to generate adversarial queries for
RANDOM_SEED = 42

# --- API Configuration ---
load_dotenv(PROJECT_ROOT / '.env')
API_KEY = os.environ.get('GOOGLE_API_KEY')
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={API_KEY}"
MAX_RETRIES = 5

# --- Prompt Engineering ---
PROMPT_TEMPLATE = """
You are a robust data generation assistant. Your task is to generate 3 specific adversarial perturbations for a 3D asset description.

Original Description: "{caption}"

Generate exactly one variation for each of these three categories:
1. lexical_substitution: Replace words with synonyms or near-synonyms (e.g., 'armchair' -> 'lounge chair').
2. syntactic_variation: Reorder phrases or modify sentence structure without altering semantics (e.g., 'table made of glass' -> 'glass table').
3. semantic_distraction: Add mild modifiers or irrelevant adjectives that do not change the core object (e.g., 'wooden chair' -> 'comfortable wooden chair').

Output *only* a valid JSON object with these three keys.

JSON Schema:
{{
  "lexical_substitution": "string",
  "syntactic_variation": "string",
  "semantic_distraction": "string"
}}
"""

GENERATION_CONFIG = {
    "responseMimeType": "application/json",
    "responseSchema": {
        "type": "OBJECT",
        "properties": {
            "lexical_substitution": {"type": "STRING"},
            "syntactic_variation": {"type": "STRING"},
            "semantic_distraction": {"type": "STRING"}
        },
        "required": ["lexical_substitution", "syntactic_variation", "semantic_distraction"]
    }
}

def load_test_split():
    """
    Loads the test split from the augmented metadata.
    """
    if TEST_SPLIT_PATH.exists():
        print(f"Loading existing test split from {TEST_SPLIT_PATH}...")
        df_test = pd.read_csv(TEST_SPLIT_PATH)

        # Ensure exact size
        if len(df_test) > TEST_SET_SIZE:
            df_test = df_test.head(TEST_SET_SIZE)

        return df_test
    else:
        #ERROR HANDLING
        print(f"Error: Test split file not found at {TEST_SPLIT_PATH}. Please create the test split first.", file=sys.stderr)
        sys.exit(1)


def generate_adversarial_queries(caption: str) -> dict | None:
    """
    Calls Gemini API to generate the 3 adversarial variations.
    """
    payload = {
        "contents": [{ "parts": [{ "text": PROMPT_TEMPLATE.format(caption=caption) }] }],
        "generationConfig": GENERATION_CONFIG
    }
    headers = {'Content-Type': 'application/json'}

    for i in range(MAX_RETRIES):
        try:
            response = requests.post(API_URL, headers=headers, data=json.dumps(payload), timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                text_content = result['candidates'][0]['content']['parts'][0]['text']
                return json.loads(text_content)
                
            elif response.status_code == 429:
                time.sleep((2 ** i) * 2) # Exponential backoff
            else:
                print(f"API Error {response.status_code}: {response.text}", file=sys.stderr)
                time.sleep(2)
                
        except Exception as e:
            print(f"Request failed: {e}", file=sys.stderr)
            time.sleep(2)
            
    return None

def main():
    if not API_KEY:
        print("Error: GOOGLE_API_KEY not found in .env", file=sys.stderr)
        sys.exit(1)

    # 1. Get the Test Split
    df_test = load_test_split()
    
    # 2. Check for Resumability (Load existing CSV rows)
    processed_uids = set()
    if PROGRESS_FILE.exists():
        try:
            # We only need the UID column to know what to skip
            df_progress = pd.read_csv(PROGRESS_FILE)
            if 'uid' in df_progress.columns:
                processed_uids = set(df_progress['uid'].astype(str))
            print(f"Resuming: Found {len(processed_uids)} already processed assets.")
        except Exception as e:
            print(f"Warning: Could not read progress file: {e}")

    # 3. Generate Queries
    print(f"Generating adversarial queries for {len(df_test)} assets...")
    
    fieldnames = [
        "uid", 
        "image_path", 
        "original_caption", 
        "lexical_substitution", 
        "syntactic_variation", 
        "semantic_distraction"
    ]

    # Open CSV in append mode
    # newline='' is important for the csv module to handle line endings correctly
    with open(PROGRESS_FILE, 'a', newline='', encoding='utf-8') as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        
        # Write header only if the file is new/empty
        if f_out.tell() == 0:
            writer.writeheader()

        pbar = tqdm(total=len(df_test), desc="Adversarial Gen")
        pbar.update(len(processed_uids))
        
        for index, row in df_test.iterrows():
            uid = str(row['uid'])
            original_caption = row['caption']
            
            if uid in processed_uids:
                continue
            
            # Call API
            adversarial_data = generate_adversarial_queries(original_caption)
            
            if adversarial_data:
                record = {
                    "uid": uid,
                    "image_path": row['image_path'],
                    "original_caption": original_caption,
                    "lexical_substitution": adversarial_data.get('lexical_substitution', ''),
                    "syntactic_variation": adversarial_data.get('syntactic_variation', ''),
                    "semantic_distraction": adversarial_data.get('semantic_distraction', '')
                }
                
                # Write row immediately to CSV
                writer.writerow(record)
                f_out.flush() # Ensure it saves to disk immediately
            
            pbar.update(1)
            time.sleep(0.5) # Rate limit politeness
            
    # 4. Finalize
    print("\nFinalizing adversarial dataset...")
    
    if PROGRESS_FILE.exists():
        # Read the full progress CSV
        df_adv = pd.read_csv(PROGRESS_FILE)
        # Save to the final output path (ensuring it's a clean copy)
        df_adv.to_csv(OUTPUT_ADVERSARIAL_PATH, index=False)
        
        print(f"Success! Saved {len(df_adv)} adversarial test cases to:")
        print(f"  -> {OUTPUT_ADVERSARIAL_PATH}")
        print("\n--- Sample ---")
        print(df_adv[['original_caption', 'lexical_substitution']].head(2))
    else:
        print("Error: No progress file found. Something went wrong.")

if __name__ == "__main__":
    main()