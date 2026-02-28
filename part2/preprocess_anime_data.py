import os
import csv
import urllib.request
import numpy as np
from transformers import GPT2TokenizerFast
from array_record.python import array_record_module
from google.cloud import storage

# --- Configuration ---
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
TRAIN_FILE = "avatar_train.arrayrecord"
VAL_FILE = "avatar_val.arrayrecord"
SEQ_LEN = 256

def upload_to_gcs(local_path, bucket_name, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_path)
    print(f"Uploaded {local_path} to gs://{bucket_name}/{destination_blob_name}")

def download_and_format_avatar_script():
    print("Downloading Avatar: The Last Airbender transcript...")
    url = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-08-11/avatar.csv"
    
    # Download and decode the CSV
    response = urllib.request.urlopen(url)
    lines = [l.decode('utf-8') for l in response.readlines()]
    reader = csv.DictReader(lines)
    
    script_lines = []
    for row in reader:
        character = row.get("character", "").strip()
        dialogue = row.get("character_words", "").strip()
        
        if not character or not dialogue:
            continue
            
        if character == "Scene Description":
            script_lines.append(f"[{dialogue}]\n\n")
        else:
            script_lines.append(f"{character}: {dialogue}\n\n")
            
    return "".join(script_lines)

def preprocess_and_upload():
    data = download_and_format_avatar_script()
    
    # Create a standard 90% / 10% train/val split
    n = len(data)
    train_text = data[:int(n * 0.9)]
    val_text = data[int(n * 0.9):]
    
    splits = [("train", TRAIN_FILE, train_text), ("validation", VAL_FILE, val_text)]
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    
    for split, filename, text in splits:
        print(f"\nProcessing {split} split...")
        tokens = tokenizer.encode(text, truncation=False)
        
        # Chunk into sequences
        sequences = [tokens[i : i + SEQ_LEN] for i in range(0, len(tokens) - SEQ_LEN, SEQ_LEN)]
        
        if sequences:
            sample_seq = sequences[0]
            print(f"--- Sample from {split} ---")
            print(f"Decoded text:\n{tokenizer.decode(sample_seq[:50])} ...\n---------------------------")
        
        writer = array_record_module.ArrayRecordWriter(filename, 'zstd,group_size:1')
        for seq in sequences:
            seq_np = np.array(seq, dtype=np.int32)
            writer.write(seq_np.tobytes())
        writer.close()
        
        # Push to GCS
        upload_to_gcs(filename, GCS_BUCKET_NAME, f"toy_llm/data/{filename}")
        os.remove(filename)

if __name__ == "__main__":
    preprocess_and_upload()