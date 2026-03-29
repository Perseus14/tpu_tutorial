import argparse
import os

# --- SILENCE BACKEND LOGS ---
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" 
os.environ["GLOG_minloglevel"] = "2"
os.environ["GRPC_VERBOSITY"] = "ERROR"
# ----------------------------

import jax
import jax.numpy as jnp
from flax import nnx
import optax
import orbax.checkpoint as ocp
from transformers import GPT2TokenizerFast
import numpy as np

# Import your model directly from your training script!
from tpu_train import NanoGPT

BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "your_bucket_name_here") # Make sure to set this in your env.sh
CHECKPOINT_DIR = f"gs://{BUCKET_NAME}/toy_llm/checkpoints"

# ==========================================
# Generation Logic
# ==========================================
@nnx.jit(static_argnames=("temperature", "top_k"))
def generate_token_step(model: NanoGPT, padded_sequence: jax.Array, current_idx: int, prng_key: jax.Array, temperature: float, top_k: int):
    # 1. Forward pass on the ENTIRE fixed-size array (Only compiles ONCE!)
    logits = model(padded_sequence)
    
    # 2. Extract the logit specifically for the last valid token we inputted
    next_token_logits = logits[0, current_idx - 1, :] 
    
    # Apply temperature
    next_token_logits = next_token_logits / temperature
    
    # Apply Top-K filtering
    top_k_values, _ = jax.lax.top_k(next_token_logits, top_k)
    min_top_k = top_k_values[-1:]
    next_token_logits = jnp.where(next_token_logits < min_top_k, -1e9, next_token_logits)
    
    # Sample from the filtered distribution
    new_token = jax.random.categorical(prng_key, next_token_logits)
    return new_token

def generate(model: NanoGPT, prompt_tokens: jax.Array, max_new_tokens: int, temperature: float, top_k: int, seed: int):
    prompt_len = prompt_tokens.shape[1]
    total_len = prompt_len + max_new_tokens
    
    # Create a fixed-size array padded with zeros
    padded_sequence = jnp.zeros((1, total_len), dtype=jnp.int32)
    
    # Inject the prompt into the beginning of the array
    padded_sequence = padded_sequence.at[0, :prompt_len].set(prompt_tokens[0])
    
    key = jax.random.PRNGKey(seed)
    
    print(f"Generating {max_new_tokens} tokens")
    
    for i in range(prompt_len, total_len):
        key, subkey = jax.random.split(key)
        
        # Pass the current length 'i' so the model knows which logit to sample from
        new_token = generate_token_step(model, padded_sequence, i, subkey, temperature, top_k)
        
        # Insert the newly generated token into the padded array at the correct index
        padded_sequence = padded_sequence.at[0, i].set(new_token)
        
    return np.array(padded_sequence[0]).tolist()

# ==========================================
# Main Execution
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Run Inference on trained NanoGPT")
    parser.add_argument("--run_id", type=str, required=True, help="ID of the specific run")
    parser.add_argument("--step", type=int, default=None, help="Specific step to load. Defaults to latest.")
    parser.add_argument("--prompt", type=str, default="The future of artificial intelligence is", help="Starting text for the model.")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Number of tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Creativity vs strictness (lower = safer).")
    parser.add_argument("--top_k", type=int, default=40, help="Limits sampling to top K tokens.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation.")
    
    args = parser.parse_args()

    # Model Hyperparameters (Must match training!)
    vocab_size = 50304
    d_model = 1024
    num_heads = 16
    num_layers = 24
    max_seq_len = 1024
    dtype = jnp.bfloat16

    print("Initializing model architecture...")
    model = NanoGPT(vocab_size, d_model, num_heads, num_layers, max_seq_len, dtype, nnx.Rngs(0))
    
    # 1. Create a dummy schedule so Optax includes the 'count' variable in its state tree
    dummy_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=1.5e-4,
        warmup_steps=500,
        decay_steps=10000, 
        end_value=1.5e-5
    )
    
    # 2. Pass the schedule into the optimizer
    dummy_optimizer = nnx.Optimizer(
        model, 
        optax.chain(
            optax.clip_by_global_norm(1.0), 
            optax.adamw(learning_rate=dummy_schedule)
        ), 
        wrt=nnx.Param
    )
    
    state = nnx.state((model, dummy_optimizer))
    checkpoint_path = os.path.join(CHECKPOINT_DIR, args.run_id)

    print(f"Connecting to Orbax Checkpoint Manager at: {checkpoint_path}")
    options = ocp.CheckpointManagerOptions(create=False)
    checkpoint_manager = ocp.CheckpointManager(checkpoint_path, options=options, item_names=('state',))
    
    step_to_load = args.step if args.step is not None else checkpoint_manager.latest_step()
    if step_to_load is None:
        raise ValueError(f"No checkpoint found in {checkpoint_path}")

    print(f"Downloading and restoring weights from step {step_to_load}...")
    restored = checkpoint_manager.restore(
        step_to_load, 
        args=ocp.args.Composite(
            state=ocp.args.StandardRestore(state)
        )
    )
    
    nnx.update((model, dummy_optimizer), restored['state'])
    print("Weights loaded successfully!\n")

    print(f"Input Prompt: {args.prompt}")

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    encoded_prompt = tokenizer.encode(args.prompt)
    prompt_array = jnp.array([encoded_prompt], dtype=jnp.int32)
    prompt_array = jax.device_put(prompt_array)

    output_tokens = generate(
        model=model, 
        prompt_tokens=prompt_array, 
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed
    )

    decoded_text = tokenizer.decode(output_tokens)
    print("=" * 60)
    print("🤖 GENERATED OUTPUT:")
    print("=" * 60)
    print(decoded_text)
    print("=" * 60)

if __name__ == "__main__":
    main()

'''
python tpu_inference.py \
    --run_id "<run_id>" \
    --prompt "JAX and TPUs are incredibly powerful because" \
    --max_new_tokens 150 \
    --temperature 0.7
'''