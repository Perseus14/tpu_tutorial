import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import numpy as np
from transformers import GPT2TokenizerFast
from datasets import load_dataset
from array_record.python import array_record_module
from google.cloud import storage
import time

# --- Configuration ---
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
TRAIN_FILE = "openwebtext_train_full.arrayrecord"
VAL_FILE = "openwebtext_val_full.arrayrecord"
SEQ_LEN = 1025 # 1024 tokens + 1 for the EOT token
HF_TOKEN = os.getenv("HF_TOKEN") # You can get your own token from https://huggingface.co/settings/tokens

TRAIN_SPLIT = "train[:99%]"
VAL_SPLIT = "train[99%:]"

def upload_to_gcs(local_path, bucket_name, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    print(f"Uploading {local_path} to GCS...")
    blob.upload_from_filename(local_path)
    print(f"Successfully uploaded to gs://{bucket_name}/{destination_blob_name}")

def process_and_write(dataset_split, filename, tokenizer):
    print(f"\nDownloading dataset split: {dataset_split}...")
    dataset = load_dataset("Skylion007/openwebtext", split=dataset_split, token=HF_TOKEN)
    
    print("Tokenizing entire dataset across multiple CPU cores... (This will be fast!)")
    def tokenize_function(examples):
        return {"tokens": tokenizer(examples["text"], truncation=False)["input_ids"]}

    # num_proc=16 will use 16 CPU cores to tokenize in parallel
    dataset = dataset.map(tokenize_function, batched=True, num_proc=16, remove_columns=["text"])
    
    print(f"Packing tokens into {filename}...")
    writer = array_record_module.ArrayRecordWriter(filename, 'zstd,group_size:1')
    eot_token = tokenizer.eos_token_id 
    
    buffer = []
    total_sequences = 0
    start_time = time.time()

    # Now we just iterate over the pre-tokenized integers
    for item in dataset:
        buffer.extend(item['tokens'] + [eot_token])
        
        while len(buffer) >= SEQ_LEN:
            chunk = buffer[:SEQ_LEN]
            buffer = buffer[SEQ_LEN:] # Keep the leftovers for the next chunk
            
            seq_np = np.array(chunk, dtype=np.int32)
            writer.write(seq_np.tobytes())
            total_sequences += 1
            
            if total_sequences % 50000 == 0: # Increased print interval since it's much faster now
                print(f"Packed {total_sequences} sequences of length {SEQ_LEN}...")

    writer.close()
    end_time = time.time()
    print(f"Finished writing {total_sequences} sequences to {filename} in {end_time - start_time:.2f} seconds.")

def main():
    print("Loading GPT-2 Tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    
    # Process Train and Val
    process_and_write(TRAIN_SPLIT, TRAIN_FILE, tokenizer)
    process_and_write(VAL_SPLIT, VAL_FILE, tokenizer)
    
    # Upload to GCS
    upload_to_gcs(TRAIN_FILE, GCS_BUCKET_NAME, f"toy_llm/data/{TRAIN_FILE}")
    upload_to_gcs(VAL_FILE, GCS_BUCKET_NAME, f"toy_llm/data/{VAL_FILE}")
    
    # Clean up local files
    os.remove(TRAIN_FILE)
    os.remove(VAL_FILE)
    print("All done!")

if __name__ == "__main__":
    main()