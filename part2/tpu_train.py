import argparse
import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
import jax
import jax.numpy as jnp
from flax import nnx
import optax
import orbax.checkpoint as ocp
import grain.python as grain
from dataclasses import dataclass
import numpy as np
from transformers import GPT2TokenizerFast
import time

BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "your_bucket_name_here") # Make sure to set this in your env.sh

builtin_print = print
def print(*args, **kwargs):
    if jax.process_index() == 0:
        builtin_print(*args, **kwargs)

# ==========================================
# 1. Configuration
# ==========================================
@dataclass
class TrainConfig:
    batch_size: int = 1024 # Global batch size (will be sharded across devices)
    max_learning_rate: float = 1.5e-4
    min_learning_rate: float = 1.5e-5
    warmup_steps: int = 2500
    total_steps: int = 100000
    
    train_data_path: str = f"gs://{BUCKET_NAME}/toy_llm/data/openwebtext_train_full.arrayrecord"
    val_data_path: str = f"gs://{BUCKET_NAME}/toy_llm/data/openwebtext_val_full.arrayrecord"
    checkpoint_dir: str = f"gs://{BUCKET_NAME}/toy_llm/checkpoints/"
    
    vocab_size: int = 50257
    d_model: int = 1024
    num_heads: int = 16
    num_layers: int = 24
    max_seq_len: int = 1024

    save_every_n_steps: int = 5000
    eval_every_n_steps: int = 2500
    generate_every_n_steps: int = 1000
    max_generate_length: int = 100

    dtype: any = jnp.bfloat16

config = TrainConfig()

# ==========================================
# 2. Grain Data Loader
# ==========================================
def get_dataloader(file_path: str, batch_size: int, is_training: bool):
    data_source = grain.ArrayRecordDataSource(file_path)
    
    class DecodeTokens(grain.MapTransform):
        def map(self, record):
            tokens = np.frombuffer(record, dtype=np.int32)
            return {"x": tokens[:-1], "y": tokens[1:]}

    operations = [
        DecodeTokens(),
        grain.Batch(batch_size=batch_size, drop_remainder=True)
    ]
    
    sampler = grain.IndexSampler(
        num_records=len(data_source),
        num_epochs=None if is_training else 1,
        shard_options=grain.ShardOptions(
            shard_index=jax.process_index(), 
            shard_count=jax.process_count(), 
            drop_remainder=True
        ),
        shuffle=is_training,
        seed=42
    )

    fast_network_options = grain.ReadOptions(
        num_threads=16,             # Use 16 background threads to pull from GCS
        prefetch_buffer_size=3000    # Keep 3 full batches of raw record ready in RAM at all times
    )
    
    return grain.DataLoader(
        data_source=data_source,
        sampler=sampler,
        operations=operations,
        worker_count=4,
        worker_buffer_size=5,
        read_options=fast_network_options
    )

# ==========================================
# 3. NNX NanoGPT Model
# ==========================================
class KVCache(nnx.Variable):
    """A custom variable type so the optimizer ignores the cache during training."""
    pass

class TransformerBlock(nnx.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, dtype: jnp.dtype, rngs: nnx.Rngs):
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        
        self.ln1 = nnx.LayerNorm(d_model, dtype=jnp.float32, rngs=rngs)
        self.q_proj = nnx.Linear(d_model, d_model, dtype=dtype, use_bias=False, rngs=rngs)
        self.k_proj = nnx.Linear(d_model, d_model, dtype=dtype, use_bias=False, rngs=rngs)
        self.v_proj = nnx.Linear(d_model, d_model, dtype=dtype, use_bias=False, rngs=rngs)
        self.o_proj = nnx.Linear(d_model, d_model, dtype=dtype, use_bias=False, rngs=rngs)
        
        self.ln2 = nnx.LayerNorm(d_model, dtype=jnp.float32, rngs=rngs)
        self.mlp_fc1 = nnx.Linear(d_model, 4 * d_model, dtype=dtype, rngs=rngs)
        self.mlp_fc2 = nnx.Linear(4 * d_model, d_model, dtype=dtype, rngs=rngs)

        # Initialize KV Cache (Shape: Batch=1, MaxSeqLen, Heads, HeadDim)
        self.cache_k = KVCache(jnp.zeros((1, max_seq_len, num_heads, self.head_dim), dtype=dtype))
        self.cache_v = KVCache(jnp.zeros((1, max_seq_len, num_heads, self.head_dim), dtype=dtype))

    def __call__(self, x, mask, cache_index=None):
        residual = x
        x = self.ln1(x)
        
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, L, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, L, self.num_heads, self.head_dim)

        if cache_index is not None:
            # --- DECODE MODE (KV Cache) ---
            # Insert the new token's K and V into the cache array at the current index
            self.cache_k[...] = jax.lax.dynamic_update_slice(self.cache_k[...], k, (0, cache_index, 0, 0))
            self.cache_v[...] = jax.lax.dynamic_update_slice(self.cache_v[...], v, (0, cache_index, 0, 0))
            k_use, v_use = self.cache_k[...], self.cache_v[...]
        else:
            # --- TRAIN MODE ---
            k_use, v_use = k, v

        # Attention: (B, L, H, D) x (B, Seq, H, D) -> (B, H, L, Seq)
        scores = jnp.einsum('bqhd,bkhd->bhqk', q, k_use) / jnp.sqrt(self.head_dim)
        
        if mask is not None:
            scores = jnp.where(mask, scores, -1e5)
            
        attn_weights = jax.nn.softmax(scores, axis=-1)
        attn_out = jnp.einsum('bhqk,bkhd->bqhd', attn_weights, v_use)
        attn_out = attn_out.reshape(B, L, self.d_model)
        
        x = residual + self.o_proj(attn_out)

        mlp_out = self.mlp_fc1(self.ln2(x))
        mlp_out = jax.nn.gelu(mlp_out)
        return x + self.mlp_fc2(mlp_out)

class NanoGPT(nnx.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, num_layers: int, max_seq_len: int, dtype: jnp.dtype,rngs: nnx.Rngs):
        self.max_seq_len = max_seq_len
        self.tok_emb = nnx.Embed(vocab_size, d_model, dtype=dtype, rngs=rngs)
        self.pos_emb = nnx.Embed(max_seq_len, d_model, dtype=dtype, rngs=rngs)
        
        @nnx.vmap(in_axes=0, out_axes=0)
        def create_blocks(key):
            # We create a fresh PRNG wrapper for each layer
            return TransformerBlock(d_model, num_heads, max_seq_len, dtype=dtype, rngs=nnx.Rngs(key))

        keys = jax.random.split(rngs(), num_layers) 
        self.blocks = create_blocks(keys)
        self.ln_f = nnx.LayerNorm(d_model, dtype=jnp.float32, rngs=rngs)
        self.lm_head = nnx.Linear(d_model, vocab_size, dtype=dtype, rngs=rngs)

    def __call__(self, x, cache_index=None):
        batch_size, seq_len = x.shape
        tok_embeddings = self.tok_emb(x)
        
        if cache_index is not None:
            # Generate Mode: Positional index is just the current scalar cache_index
            pos_indices = jnp.expand_dims(cache_index, 0)
            # Mask allows attending to all previous tokens up to current cache_index
            mask = jnp.arange(self.max_seq_len) <= cache_index
            mask = mask[None, None, None, :] 
        else:
            # Train Mode: Full causal masking
            pos_indices = jnp.arange(seq_len)
            mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))[None, None, :, :]
            
        x = tok_embeddings + self.pos_emb(pos_indices)[None, :]

        apply_blocks = nnx.scan(
            nnx.remat(lambda carry_x, blocks, m, c_idx: blocks(carry_x, mask=m, cache_index=c_idx)),
            in_axes=(nnx.Carry, 0, None, None), 
            out_axes=nnx.Carry
        )

        # Execute it
        x = apply_blocks(x, self.blocks, mask, cache_index)

        return self.lm_head(self.ln_f(x))

# ==========================================
# 4. Step Functions & Generation
# ==========================================
def loss_fn(model: NanoGPT, batch: dict):
    logits = model(batch['x'])
    loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=batch['y'])
    return jnp.mean(loss)

@nnx.jit(donate_argnames=("model", "optimizer"))
def train_step(model: NanoGPT, optimizer: nnx.Optimizer, batch: dict):
    loss, grads = nnx.value_and_grad(loss_fn)(model, batch)
    optimizer.update(model, grads)
    return loss

@nnx.jit
def eval_step(model: NanoGPT, batch: dict):
    return loss_fn(model, batch)

@nnx.jit
def generate_token_step(model: NanoGPT, next_token: jax.Array, cache_index: jax.Array, prng_key: jax.Array):
    logits = model(next_token, cache_index=cache_index)
    
    # Grab the logits for the very last token in the sequence
    next_token_logits = logits[:, -1, :] 
    
    # --- ADD TEMPERATURE (Controls creativity. 1.0 is standard, 0.8 is focused) ---
    temperature = 0.8
    next_token_logits = next_token_logits / temperature
    
    # --- ADD TOP-K (Prevents the model from picking total gibberish) ---
    top_k = 40
    # Find the threshold of the 40th best word
    top_k_values, _ = jax.lax.top_k(next_token_logits, top_k)
    min_top_k = top_k_values[:, -1:]
    # Mask out everything else by setting their probability to negative infinity
    next_token_logits = jnp.where(next_token_logits < min_top_k, -1e9, next_token_logits)
    
    # Sample from the remaining probabilities
    new_token = jax.random.categorical(prng_key, next_token_logits)
    return jnp.expand_dims(new_token, axis=-1)

def generate(model: NanoGPT, prompt_token: jax.Array, max_new_tokens: int, seed: int = 42):
    current_token = prompt_token
    generated = [current_token[0, 0].item()]
    key = jax.random.PRNGKey(seed)
    
    # Generate token by token
    for i in range(max_new_tokens):
        key, subkey = jax.random.split(key)
        cache_index = jnp.array(i, dtype=jnp.int32)
        current_token = generate_token_step(model, current_token, cache_index, subkey)
        
        generated.append(current_token[0, 0].item())
        
    return generated

# ==========================================
# 5. Main Execution Loop with DDP
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Train GPT-2 Medium")
    parser.add_argument("--resume_run_id", type=str, default=None, 
                        help="The timestamp ID of a previous run to resume (e.g., 20260301_121500)")
    args = parser.parse_args()

    print(f"JAX running on {jax.device_count()} devices.")
    
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    # 1. Setup DDP Mesh and Sharding specs
    # Create a 1D mesh over all available TPU/GPU devices
    mesh = jax.sharding.Mesh(jax.devices(), ('data',))
    # We will shard the batch dimension of our data across the 'data' axis
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('data', None))
    # We want model weights replicated on all devices
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # 2. Initialize Loaders
    train_loader = get_dataloader(config.train_data_path, config.batch_size, is_training=True)
    val_loader = get_dataloader(config.val_data_path, config.batch_size, is_training=False)
    train_iterator = iter(train_loader)

    # 3. Initialize NNX Model & Optimizer
    rngs = nnx.Rngs(0)
    model = NanoGPT(
        vocab_size=config.vocab_size, 
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        max_seq_len=config.max_seq_len,
        dtype=config.dtype,
        rngs=rngs
    )
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.max_learning_rate,
        warmup_steps=config.warmup_steps,
        decay_steps=config.total_steps, # Total steps including warmup
        end_value=config.min_learning_rate
    )
    optimizer = nnx.Optimizer(
        model, 
        optax.chain(
            optax.clip_by_global_norm(1.0), # Prevents exploding gradients
            optax.adamw(learning_rate=lr_schedule)
        ), 
        wrt=nnx.Param
    )

    # Explicitly place model/optimizer weights on all devices
    # 1. Extract the combined state as a tuple
    state = nnx.state((model, optimizer))
    
    # 2. Map the arrays to the device mesh
    sharded_state = jax.tree.map(lambda x: jax.device_put(x, replicated_sharding), state)
    
    # 3. Update the original objects in-place
    nnx.update((model, optimizer), sharded_state)

    # 4. Initialize Checkpointer
    if args.resume_run_id:
        run_id = args.resume_run_id
        print(f"\n[RESUME MODE] Connecting to existing Run ID: {run_id}")
    else:
        run_id = time.strftime("%Y%m%d_%H%M%S") 
        print(f"\n[NEW RUN] Run ID generated: {run_id}")

    run_checkpoint_dir = os.path.join(config.checkpoint_dir, run_id)
    print(f"Checkpoints directory: {run_checkpoint_dir}\n")

    options = ocp.CheckpointManagerOptions(
        max_to_keep=3, 
        create=True, 
        best_fn=lambda metrics: metrics['val_loss'],
        best_mode='min'
    )
    checkpoint_manager = ocp.CheckpointManager(
        run_checkpoint_dir, options=options, item_names=('state',)
    )

    # --- NEW: Restore the weights if resuming ---
    start_step = 0
    if args.resume_run_id:
        latest_step = checkpoint_manager.latest_step()
        if latest_step is not None:
            print(f"Found saved checkpoint at step {latest_step}. Restoring weights...")
            
            # Restore the raw arrays from GCS
            restored = checkpoint_manager.restore(
                latest_step, 
                args=ocp.args.Composite(
                    state=ocp.args.StandardRestore(state)
                )
            )
            
            # Map the restored CPU arrays back to the TPU device mesh
            sharded_restored_state = jax.tree.map(
                lambda x: jax.device_put(x, replicated_sharding), 
                restored['state']
            )
            
            # Inject the restored, sharded weights into the model/optimizer
            nnx.update((model, optimizer), sharded_restored_state)
            start_step = latest_step
            print("Restore complete! Ready to resume training.\n")
        else:
            print(f"WARNING: No checkpoint found in {run_checkpoint_dir}. Starting from step 0.")

    print("Compiling model (dummy step)...")
    dummy_batch = next(train_iterator)
    jax_dummy_batch = {k: jax.device_put(jnp.array(v), data_sharding) for k, v in dummy_batch.items()}
    
    # Run once to trigger JIT compilation
    dummy_loss = train_step(model, optimizer, jax_dummy_batch)
    dummy_loss.block_until_ready() 
    print("Compilation complete!")

    print("Starting training loop...")
    start_time = time.time()
    for step in range(start_step + 1, config.total_steps + 1):
        # Fetch batch and put it on the device mesh, sharded across devices
        batch = next(train_iterator)
        jax_batch = {k: jax.device_put(jnp.array(v), data_sharding) for k, v in batch.items()}
        
        loss = train_step(model, optimizer, jax_batch)

        if step % 100 == 0:
            loss.block_until_ready()
            end_time = time.time()
            avg_step_time_ms = ((end_time - start_time) / 100) * 1000
            print(f"Step {step} | Train Loss: {float(loss):.4f} | Avg Step Time: {avg_step_time_ms:.2f} ms")
            start_time = time.time()

        # --- Generation ---
        if step % config.generate_every_n_steps == 0:
            rand_token = np.random.randint(0, config.vocab_size)
            prompt = jnp.array([[rand_token]], dtype=jnp.int32) # Random Token
            # Put prompt on replicated sharding so all devices generate identically
            prompt = jax.device_put(prompt, replicated_sharding)
            
            token_list = generate(model, prompt, config.max_generate_length)
            decoded_text = tokenizer.decode(token_list)
            
            print(f"\n--- Step {step} Generation ---")
            print(f"Text:\n{decoded_text}\n------------------------------\n")

        # --- Evaluation ---
        if step % config.eval_every_n_steps == 0:
            val_losses = []
            for val_batch in val_loader:
                jax_val_batch = {k: jax.device_put(jnp.array(v), data_sharding) for k, v in val_batch.items()}
                val_losses.append(float(eval_step(model, jax_val_batch)))
            
            avg_val_loss = np.mean(val_losses)
            print(f"\n--- Step {step} Eval | Val Loss: {float(avg_val_loss):.4f} ---\n")

        # --- Checkpointing ---
        if step % config.save_every_n_steps == 0:
            print(f"Saving checkpoint at step {step} to GCS...")
            # Extract state dict for saving
            _, state = nnx.split((model, optimizer)) 
            checkpoint_manager.save(
                step, 
                args=ocp.args.Composite(
                    state=ocp.args.StandardSave(state)
                ),
                metrics={'val_loss': float(avg_val_loss)} # Every time checkpoint is run, validation is performed on the same state. eval_every_n_steps % save_every_n_steps == 0
            )
    print("Waiting for final checkpoint uploads to finish...")
    checkpoint_manager.wait_until_finished()
    
    print("Training complete!")

if __name__ == "__main__":
    main()