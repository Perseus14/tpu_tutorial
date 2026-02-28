import jax
import jax.numpy as jnp

def main():
    devices = jax.devices()
    print(f"JAX is running on: {devices[0].device_kind}")
    print(f"Total TPU cores detected: {jax.device_count()}")
    
    print("\nCreating a random matrix...")
    key = jax.random.PRNGKey(0)
    # JIT-compile the generation func for speed
    @jax.jit
    def generate_data(k):
        return jax.random.normal(k, (50000, 50000))
        
    x = generate_data(key).block_until_ready()
    
    # Wrap the math in a JIT-compiled function
    @jax.jit
    def compute_dot(arr):
        return jnp.dot(arr, arr.T)
        
    print("Performing matrix multiplication...")
    
    # The first run compiles the function, subsequent runs would be lightning fast
    result = compute_dot(x).block_until_ready() 
    
    print(f"Success! Result shape: {result.shape}")

if __name__ == "__main__":
    main()
