from moo import nested_pade_sim
from optimization_functions import rosenbrock
import jax.numpy as jnp
import jax
import time
import numpy as np

# Assuming your nested_pade_sim function is already defined somewhere
# from your_module import nested_pade_sim

# Define the number of repetitions
num_repetitions = 10

# Create a random JAX array with two values
random_array = jax.random.uniform(jax.random.PRNGKey(0), (2,), minval=-5.0, maxval=5.0)

# Set the parameter to pass to the function
param_value = 5

# Function to run the test
def test_nested_pade_sim():
    results = []
    for _ in range(num_repetitions):
        start_time = time.time()
        
        # Call the nested_pade_sim function with the Rosenbrock function, random array, and parameter
        result = nested_pade_sim(rosenbrock, random_array, param_value)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        results.append((result, elapsed_time))
        
        print(f"Elapsed Time: {elapsed_time:.6f} seconds")
    
    return results

# Run the test
results = test_nested_pade_sim()

# Calculate average time
average_time = np.mean([elapsed_time for _, elapsed_time in results])
print(f"\nAverage Elapsed Time over {num_repetitions} runs: {average_time:.6f} seconds")

