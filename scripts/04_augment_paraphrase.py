import os
import pandas as pd
import sys
from pathlib import Path
import time
import json
import requests 
from tqdm import tqdm
from dotenv import load_dotenv # <-- Import the new library

# --- Configuration ---
SCRIPT_DIR = Path(__file__).resolve().parent 
PROJECT_ROOT = SCRIPT_DIR.parent 

# --- Input/Output Paths ---
INPUT_CSV = PROJECT_ROOT / 'data' / 'processed' / 'final_metadata_10k.csv'
PROGRESS_FILE = PROJECT_ROOT / 'data' / 'processed' / 'augmented_paraphrases.jsonl'

# --- API Configuration ---
# --- THIS IS THE .ENV LOADER ---
# This line looks for a .env file in the project root and loads it.
load_dotenv(PROJECT_ROOT / '.env') 
# Now, os.environ.get() will be able to read the key from the .env file
API_KEY = os.environ.get('GOOGLE_API_KEY')
# -----------------------------

API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={API_KEY}"

# --- USER REQUESTED CHANGES ---
NUM_PARAPHRASES = 10 # Changed from 5 to 10
# ------------------------------

MAX_RETRIES = 5

# This is the prompt we will send.
PROMPT_TEMPLATE = f"""
You are a data augmentation assistant. Your task is to generate {NUM_PARAPHRASES} unique, high-quality paraphrases of a 3D asset description.

Rules:
1. The paraphrases must be semantically identical to the original.
2. Do not add new information, objects, or attributes.
3. Do not remove key information.
4. Vary sentence structure and vocabulary.
5. Output *only* a valid JSON list of {NUM_PARAPHRASES} strings. Do not include any other text, markdown, or explanations.

Original Description:
"{{caption}}"

JSON List:
"""

# --- Safety (JSON Parsing) ---
# We must use a schema to ensure the LLM *only* returns JSON.
GENERATION_CONFIG = {
    "responseMimeType": "application/json",
    "responseSchema": {
        "type": "ARRAY",
        "items": { "type": "STRING" },
        # --- UPDATED TO MATCH NUM_PARAPHRASES ---
        "minItems": NUM_PARAPHRASES,
        "maxItems": NUM_PARAPHRASES
    }
}

def get_paraphrases(caption: str) -> list[str] | None:
    """
    Calls the Gemini API with exponential backoff to get paraphrases.
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
                # Extract text, which is a JSON string, and parse it
                json_string = result['candidates'][0]['content']['parts'][0]['text']
                paraphrases = json.loads(json_string)
                
                if isinstance(paraphrases, list) and len(paraphrases) == NUM_PARAPHRASES:
                    return paraphrases
                else:
                    print(f"\nWarning: LLM returned unexpected data for {caption}. Skipping.", file=sys.stderr)
                    return None

            elif response.status_code == 429:
                # Rate limit hit. Wait and retry.
                wait_time = (2 ** i) * 3 # Exponential backoff (3s, 6s, 12s, ...)
                print(f"\nWarning: Rate limit hit. Waiting {wait_time}s...", file=sys.stderr)
                time.sleep(wait_time)
            
            else:
                # Other error (e.g., 500)
                print(f"\nError: API call failed with status {response.status_code}: {response.text}", file=sys.stderr)
                time.sleep(5) # Wait a bit before retry

        except requests.exceptions.RequestException as e:
            print(f"\nError: Network error {e}. Retrying...", file=sys.stderr)
            time.sleep(5)
        except (KeyError, json.JSONDecodeError) as e:
            print(f"\nError: Failed to parse LLM response: {e}", file=sys.stderr)
            print(f"Response dump: {response.text}", file=sys.stderr)
            return None # Skip this caption
            
    print(f"\nError: Failed to get paraphrases for {caption} after {MAX_RETRIES} retries. Skipping.", file=sys.stderr)
    return None

def main():
    if not API_KEY:
        print("Error: GOOGLE_API_KEY not found.", file=sys.stderr)
        print("Please make sure you have a .env file in the project root (e.g., '3D-Asset-Retrieval/.env')")
        print("and that it contains the line: GOOGLE_API_KEY='your_key_here'", file=sys.stderr)
        sys.exit(1)
        
    if "YOUR_API_KEY" in API_KEY:
        print("Error: Please replace 'YOUR_API_KEY' with your actual API key in the .env file.", file=sys.stderr)
        sys.exit(1)


    print("Loading input CSV...")
    df_in = pd.read_csv(INPUT_CSV)
    
    processed_uids = set()
    
    # --- Resumability Step ---
    if PROGRESS_FILE.exists():
        print("Found existing progress file. Loading...")
        with open(PROGRESS_FILE, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed_uids.add(data['uid'])
                except json.JSONDecodeError:
                    continue # Skip corrupt lines
        print(f"Loaded {len(processed_uids)} already-processed assets. Resuming...")

    # Open progress file in append mode
    with open(PROGRESS_FILE, 'a') as f_out:
        
        # Create a tqdm progress bar
        pbar = tqdm(total=df_in.shape[0], desc="Augmenting captions")
        pbar.update(len(processed_uids)) # Set the progress bar to the saved progress
        
        # Loop through the *input* dataframe
        for index, row in df_in.iterrows():
            uid = row['uid']
            caption = row['caption']
            
            # --- Resumability Check ---
            if uid in processed_uids:
                continue # Skip this one, we've already done it
            
            paraphrases = get_paraphrases(caption)
            
            if paraphrases:
                # Create a single JSON object for this asset
                output_data = {
                    "uid": uid,
                    "image_path": row['image_path'],
                    "original_caption": caption,
                    "paraphrases": paraphrases
                }
                # Write it as one line to the .jsonl file
                f_out.write(json.dumps(output_data) + "\n")
                f_out.flush() # Ensure it's written immediately

            pbar.update(1) # Manually update the progress bar

        pbar.close()

    print(f"\nAugmentation complete! Progress saved to {PROGRESS_FILE}.")
    print("You can now run '05_finalize_augmented_data.py' to create the final CSV.")

if __name__ == "__main__":
    # You will need to install:
    # pip install python-dotenv
    # pip install requests
    main()