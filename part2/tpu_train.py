import argparse
import os
# --- SILENCE BACKEND LOGS ---
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
# Silences TensorFlow/XLA C++ INFO and WARNING logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" 
# Silences Google Cloud / gRPC C++ INFO logs (like the auth provider)
os.environ["GLOG_minloglevel"] = "2"
os.environ["GRPC_VERBOSITY"] = "ERROR"
# ----------------------------
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


class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_divider(color=Colors.CYAN, char="=", length=60):
    print(f"{color}{char * length}{Colors.RESET}")

# ==========================================
# 1. Configuration
# ==========================================
@dataclass
class TrainConfig:
    batch_size: int = 1024              # Global batch size (will be sharded across devices)
    accumulation_steps: int = 16        # 1024 / 16 = 64 (Our physical micro-batch size)
    max_learning_rate: float = 1.5e-4
    min_learning_rate: float = 1.5e-5
    
    vocab_size: int = 50304             # Padding to nearest multiple of 128
    d_model: int = 1024
    num_heads: int = 16
    num_layers: int = 24
    max_seq_len: int = 1024

    train_data_path: str = f"gs://{BUCKET_NAME}/toy_llm/data/openwebtext_train_full.arrayrecord"
    val_data_path: str = f"gs://{BUCKET_NAME}/toy_llm/data/openwebtext_val_full.arrayrecord"
    checkpoint_dir: str = f"gs://{BUCKET_NAME}/toy_llm/checkpoints/"

    warmup_steps: int = 500         # ~5% of training spent warming up    
    total_steps: int = 10000        # Processes ~5.2 Billion tokens
    
    log_interval: int = 100
    save_every_n_steps: int = 5000
    eval_every_n_steps: int = 2500
    generate_every_n_steps: int = 1000
    max_generate_length: int = 100
    
    '''
    # --- SMOKE TEST OVERRIDES ---
    total_steps: int = 100        # Finish the whole script in minutes
    warmup_steps: int = 10        # Must be lower than total_steps!
    
    log_interval: int = 1
    save_every_n_steps: int = 50  # Forces a save at step 50 and 100
    eval_every_n_steps: int = 25  # Forces eval at 25, 50, 75, 100
    generate_every_n_steps: int = 25 # Forces text generation 4 times
    max_generate_length: int = 20    # Keep generation short so you don't wait
    # ----------------------------
    '''
    
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

    def __call__(self, x, mask):
        residual = x
        x = self.ln1(x)
        
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, L, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, L, self.num_heads, self.head_dim)

        # Look! Just pure, stateless Flash Attention!
        attn_out = jax.nn.dot_product_attention(q, k, v, mask=mask)
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

    def __call__(self, x):
        batch_size, seq_len = x.shape
        tok_embeddings = self.tok_emb(x)
        
        pos_indices = jnp.arange(seq_len)
        mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))[None, None, :, :]
            
        x = tok_embeddings + self.pos_emb(pos_indices)[None, :]

        apply_blocks = nnx.scan(
            nnx.remat(lambda carry_x, blocks, m: blocks(carry_x, mask=m)),
            in_axes=(nnx.Carry, 0, None), 
            out_axes=nnx.Carry
        )

        x = apply_blocks(x, self.blocks, mask)
        x = self.ln_f(x)
        logits = x @ self.tok_emb.embedding[...].T
        return logits

# ==========================================
# 4. Step Functions & Generation
# ==========================================
def loss_fn(model: NanoGPT, batch: dict):
    logits = model(batch['x'])
    logits = logits.astype(jnp.float32)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=batch['y'])
    return jnp.mean(loss)

@nnx.jit(donate_argnames=("model", "optimizer"))
def train_step(model: NanoGPT, optimizer: nnx.Optimizer, micro_batches: dict):
    acc_steps = config.accumulation_steps
    
    # 1. DISMANTLE: Rip the parameters out of the stateful model object
    graphdef, params, rest = nnx.split(model, nnx.Param, ...)

    # Initialize ZEROED accumulators in float32
    zero_grads = jax.tree.map(lambda x: jnp.zeros_like(x, dtype=jnp.float32), params)
    zero_loss = jnp.zeros((), dtype=jnp.float32)

    def micro_step(carry, micro_batch):
        acc_loss, acc_grads = carry
        
        # 2. REASSEMBLE: Define a pure function that temporarily rebuilds the model
        def pure_loss_fn(p):
            m = nnx.merge(graphdef, p, rest)
            return loss_fn(m, micro_batch)

        # 3. PURE MATH: Use JAX's standard autodiff, NOT nnx.value_and_grad
        loss, grads = jax.value_and_grad(pure_loss_fn)(params)
        
        # UPCAST loss and gradients to float32 before doing accumulation math
        loss = loss.astype(jnp.float32)
        grads = jax.tree.map(lambda g: g.astype(jnp.float32), grads)
        
        # Accumulate the loss and gradients
        acc_loss = acc_loss + (loss / acc_steps)
        acc_grads = jax.tree.map(lambda a, g: a + (g / acc_steps), acc_grads, grads)
        
        return (acc_loss, acc_grads), None

    # Run the XLA-optimized loop
    (final_loss, final_grads), _ = jax.lax.scan(
        micro_step, 
        (zero_loss, zero_grads), 
        micro_batches
    )

    # DOWNCAST the final accumulated gradients back to bfloat16
    final_grads = jax.tree.map(lambda orig, g: g.astype(orig.dtype), params, final_grads)

    # 4. UPDATE: Apply the accumulated gradients back to the original model
    optimizer.update(model, final_grads)
    return final_loss

@nnx.jit
def eval_step(model: NanoGPT, batch: dict):
    return loss_fn(model, batch)

@nnx.jit
def generate_token_step(model: NanoGPT, current_sequence: jax.Array, prng_key: jax.Array):
    # Just pass the whole sequence through the stateless model
    logits = model(current_sequence)
    next_token_logits = logits[:, -1, :] 
    
    temperature = 0.8
    next_token_logits = next_token_logits / temperature
    
    top_k = 40
    top_k_values, _ = jax.lax.top_k(next_token_logits, top_k)
    min_top_k = top_k_values[:, -1:]
    next_token_logits = jnp.where(next_token_logits < min_top_k, -1e9, next_token_logits)
    
    new_token = jax.random.categorical(prng_key, next_token_logits)
    return jnp.expand_dims(new_token, axis=-1)

def generate(model: NanoGPT, prompt_token: jax.Array, max_new_tokens: int, seed: int = 42):
    current_sequence = prompt_token
    key = jax.random.PRNGKey(seed)
    
    for _ in range(max_new_tokens):
        key, subkey = jax.random.split(key)
        # Generate the new token
        new_token = generate_token_step(model, current_sequence, subkey)
        # Append it to the sequence and feed the whole thing back in
        current_sequence = jnp.concatenate([current_sequence, new_token], axis=1)
        
    return np.array(current_sequence[0]).tolist()

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
    # Accumulation sharding. We shard Axis 1 (MicroBatch), not Axis 0 (Steps)
    acc_data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, 'data', None))
    
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
        best_mode='min',
        enable_async_checkpointing=True
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

    def run_validation():
        val_losses = []
        for val_batch in val_loader:
            jax_val_batch = jax.tree.map(lambda x: jax.device_put(x, data_sharding), val_batch)
            val_losses.append(eval_step(model, jax_val_batch))
        
        # Average on TPU, then bring to CPU
        return float(jnp.mean(jnp.array(val_losses)))

    print("Compiling model (dummy step)...")
    dummy_batch = next(train_iterator)
    
    # --- NEW: Reshape the dummy batch exactly like the training loop ---
    micro_b = config.batch_size // config.accumulation_steps
    dummy_batch_reshaped = jax.tree.map(
        lambda x: x.reshape(config.accumulation_steps, micro_b, *x.shape[1:]), 
        dummy_batch
    )
    
    # Push to TPU using the correct accumulation sharding spec
    jax_dummy_batch = jax.tree.map(lambda x: jax.device_put(x, acc_data_sharding), dummy_batch_reshaped)
    
    # Run once to trigger JIT compilation
    dummy_loss = train_step(model, optimizer, jax_dummy_batch)
    dummy_loss.block_until_ready() 
    print("Compilation complete!")

    print("Starting training loop...")
    start_time = time.time()
    for step in range(start_step + 1, config.total_steps + 1):
        
        batch = next(train_iterator)
        micro_b = config.batch_size // config.accumulation_steps
        
        # 1. Reshape the raw NumPy array: (1024, Seq) -> (16, 64, Seq)
        batch_reshaped = jax.tree.map(
            lambda x: x.reshape(config.accumulation_steps, micro_b, *x.shape[1:]), 
            batch
        )
        
        # 2. Push to TPU using the Axis-1 sharding spec
        jax_batch = jax.tree.map(lambda x: jax.device_put(x, acc_data_sharding), batch_reshaped)
        
        loss = train_step(model, optimizer, jax_batch)

        if step % config.log_interval  == 0:
            loss.block_until_ready()
            end_time = time.time()
            avg_step_time_ms = ((end_time - start_time) / config.log_interval) * 1000
            print(f"{Colors.BOLD}Step {step:05d}{Colors.RESET} | "
                  f"Train Loss: {Colors.RED}{float(loss):.4f}{Colors.RESET} | "
                  f"Time/Step: {avg_step_time_ms:.1f}ms")
            start_time = time.time()
    
        # --- Generation ---
        if step % config.generate_every_n_steps == 0:
            rand_token = np.random.randint(0, config.vocab_size)
            prompt = jnp.array([[rand_token]], dtype=jnp.int32) # Random Token
            # Put prompt on replicated sharding so all devices generate identically
            prompt = jax.device_put(prompt, replicated_sharding)
            
            token_list = generate(model, prompt, config.max_generate_length)
            decoded_text = tokenizer.decode(token_list)
            
            print()
            print_divider(Colors.CYAN, "-")
            print(f"{Colors.CYAN}{Colors.BOLD}🤖 GENERATION (Step {step}){Colors.RESET}")
            print(f"{decoded_text}")
            print_divider(Colors.CYAN, "-")
            print()

        # --- Evaluation ---
        # Avoid double eval on save steps
        if step % config.eval_every_n_steps == 0 and step % config.save_every_n_steps != 0:
            avg_val_loss_cpu = run_validation()
            print(f"\n{Colors.GREEN}{Colors.BOLD}✔ EVALUATION (Step {step}){Colors.RESET} | "
                  f"Val Loss: {Colors.GREEN}{avg_val_loss_cpu:.4f}{Colors.RESET}\n")

        # --- Checkpointing ---
        if step % config.save_every_n_steps == 0:
            print_divider(Colors.YELLOW)
            print(f"{Colors.YELLOW}{Colors.BOLD}💾 CHECKPOINT (Step {step}){Colors.RESET}")
            
            current_val_loss = run_validation()
            print(f"   ↳ Val Loss: {Colors.GREEN}{current_val_loss:.4f}{Colors.RESET}")
            print(f"   ↳ Uploading to GCS...")
            # Extract state dict for saving
            _, state = nnx.split((model, optimizer)) 
            checkpoint_manager.save(
                step, 
                args=ocp.args.Composite(
                    state=ocp.args.StandardSave(state)
                ),
                metrics={'val_loss': float(current_val_loss)}
            )
            print_divider(Colors.YELLOW)
            print()
    print("Waiting for final checkpoint uploads to finish...")
    checkpoint_manager.wait_until_finished()
    
    print("Training complete!")

if __name__ == "__main__":
    main()

'''
python tpu_train.py \
    --resume_run_id "<run_id>" # Optional, only if you want to resume from a previous checkpoint
'''