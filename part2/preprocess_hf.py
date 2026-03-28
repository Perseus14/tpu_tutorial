import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import time
import numpy as np
from transformers import GPT2TokenizerFast
from datasets import load_dataset
from array_record.python import array_record_module
from google.cloud import storage

# --- Configuration ---
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN or not GCS_BUCKET_NAME:
    raise ValueError("Missing credentials! Run 'source env.sh' first.")

TRAIN_FILE = "openwebtext_train_full.arrayrecord"
VAL_FILE = "openwebtext_val_full.arrayrecord"
SEQ_LEN = 1025

def upload_to_gcs(local_path, bucket_name, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    print(f"Uploading {local_path} to GCS...")
    blob.upload_from_filename(local_path)
    print(f"Successfully uploaded to gs://{bucket_name}/{destination_blob_name}")

def main():
    print("Loading GPT-2 Tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    eot_token = tokenizer.eos_token_id

    # 1. LOAD DATASET IN STREAMING MODE (Zero disk caching!)
    print("Connecting to Hugging Face stream...")
    dataset = load_dataset("Skylion007/openwebtext", split="train", streaming=True, token=HF_TOKEN)
    
    # Batch the stream into chunks of 1000 so the Rust tokenizer can use all your CPU cores
    batched_dataset = dataset.batch(batch_size=1000)

    # 2. ROUTING SETUP
    VAL_TARGET_SEQS = 5000  # Approx 5 Million tokens for validation
    print(f"Writing first {VAL_TARGET_SEQS} sequences to {VAL_FILE}...")
    
    val_writer = array_record_module.ArrayRecordWriter(VAL_FILE, 'zstd,group_size:1')
    train_writer = None # We will open this after validation finishes

    buffer = []
    val_sequences = 0
    train_sequences = 0
    start_time = time.time()

    # 3. THE STREAMING LOOP
    for batch in batched_dataset:
        # Tokenize 1000 documents instantly
        tokenized_batch = tokenizer(batch["text"], truncation=False)["input_ids"]
        
        # Flatten and add EOT tokens
        for doc_tokens in tokenized_batch:
            buffer.extend(doc_tokens + [eot_token])

        # Write out perfectly sized 1025-length chunks
        while len(buffer) >= SEQ_LEN:
            chunk = buffer[:SEQ_LEN]
            buffer = buffer[SEQ_LEN:]
            
            seq_np = np.array(chunk, dtype=np.int32)
            
            # Route to Validation first
            if val_sequences < VAL_TARGET_SEQS:
                val_writer.write(seq_np.tobytes())
                val_sequences += 1
                
                # Switch tracks when Validation is full
                if val_sequences == VAL_TARGET_SEQS:
                    val_writer.close()
                    print("Validation set complete! Opening Training file stream...")
                    train_writer = array_record_module.ArrayRecordWriter(TRAIN_FILE, 'zstd,group_size:1')
            
            # Route the rest to Training
            else:
                train_writer.write(seq_np.tobytes())
                train_sequences += 1
                
                if train_sequences % 25000 == 0:
                    print(f"Packed {train_sequences} training sequences...")

    # Close out the final file
    if train_writer is not None:
        train_writer.close()

    end_time = time.time()
    print(f"Finished processing! Total Train Sequences: {train_sequences}. Time: {end_time - start_time:.2f}s")

    # 4. UPLOAD AND CLEANUP
    upload_to_gcs(VAL_FILE, GCS_BUCKET_NAME, f"toy_llm/data/{VAL_FILE}")
    upload_to_gcs(TRAIN_FILE, GCS_BUCKET_NAME, f"toy_llm/data/{TRAIN_FILE}")

    os.remove(VAL_FILE)
    os.remove(TRAIN_FILE)
    print("All done! Your full dataset is waiting in GCS.")

if __name__ == "__main__":
    main()