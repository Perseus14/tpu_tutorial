import os
import urllib.request
import numpy as np
from transformers import GPT2TokenizerFast
from array_record.python import array_record_module
from google.cloud import storage

# --- Configuration ---
GCS_BUCKET_NAME = "rishabh_tests" # Replace with your bucket
TRAIN_FILE = "shakespeare_train.arrayrecord"
VAL_FILE = "shakespeare_val.arrayrecord"
SEQ_LEN = 256

def upload_to_gcs(local_path, bucket_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_path)
    print(f"Uploaded {local_path} to gs://{bucket_name}/{destination_blob_name}")

def preprocess_and_upload():
    print("Downloading raw dataset and loading tokenizer...")
    # Download directly from Karpathy's original char-rnn repo
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    data = urllib.request.urlopen(url).read().decode('utf-8')
    
    # Create a standard 90% / 10% train/val split
    n = len(data)
    train_text = data[:int(n * 0.9)]
    val_text = data[int(n * 0.9):]
    
    splits = [("train", TRAIN_FILE, train_text), ("validation", VAL_FILE, val_text)]
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    
    for split, filename, text in splits:
        print(f"Processing {split} split...")
        tokens = tokenizer.encode(text, truncation=False)
        
        # Chunk into sequences
        sequences = [tokens[i : i + SEQ_LEN] for i in range(0, len(tokens) - SEQ_LEN, SEQ_LEN)]
        
        # Write to ArrayRecord locally
        writer = array_record_module.ArrayRecordWriter(filename, 'zstd')
        for seq in sequences:
            # ArrayRecord stores raw bytes, so we convert our numpy array of tokens
            seq_np = np.array(seq, dtype=np.int32)
            writer.write(seq_np.tobytes())
        writer.close()
        
        # Push to GCS
        upload_to_gcs(filename, GCS_BUCKET_NAME, f"toy_llm/data/{filename}")
        os.remove(filename) # Clean up local file

if __name__ == "__main__":
    preprocess_and_upload()