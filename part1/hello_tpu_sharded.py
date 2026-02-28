import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

def main():
    devices = jax.devices()
    print(f"JAX is running on: {devices[0].device_kind}")
    print(f"Total TPU cores detected: {len(devices)}")
    
    # 1. Create a 2D Mesh
    # We reshape the flat list of 4 devices into a 2x2 grid.
    # We name the grid dimensions 'x' and 'y'.
    mesh = Mesh(np.array(devices).reshape(2, 2), axis_names=('x', 'y'))
    
    # 2. Define the 2D sharding spec
    # P('x', 'y') tells JAX to split the rows across the 'x' axis 
    # and the columns across the 'y' axis of our mesh.
    sharding = NamedSharding(mesh, P('x', 'y'))
    
    N = 50000
    
    @jax.jit
    def generate_distributed_data(key):
        return jax.random.normal(key, (N, N))
    
    # Apply the 2D sharding specification to the generation function
    generate_distributed_data = jax.jit(
        generate_distributed_data, 
        out_shardings=sharding
    )

    key = jax.random.PRNGKey(0)
    print(f"Creating distributed data...")
    x_sharded = generate_distributed_data(key).block_until_ready() 
    
    print(f"Data is now natively 2D distributed: {x_sharded.sharding}\n")

    @jax.jit
    def distributed_dot(arr):
        return jnp.dot(arr, arr.T)
    
    print("Performing 2D distributed matrix multiplication...")
    
    # Block until ready ensures we wait for the async TPU execution to finish
    result = distributed_dot(x_sharded).block_until_ready() 
    
    print("\n--- Results ---")
    print(f"Success! Result shape: {result.shape}")
    print(f"Result distribution: {result.sharding}")

if __name__ == "__main__":
    main()
