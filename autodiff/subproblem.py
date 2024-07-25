import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from jax import grad
from optimization_functions import (
    exp, ackley, sphere, rastrigin, rosenbrock, beale, goldstein_price, levi, bukin_n6, booth, matyas, three_hump_camel, easom, mccormick, styblinski_tang, schaffer_n2
)


function_list = [
    exp,
    ackley,
    sphere,
    rastrigin,
    rosenbrock,
    beale,
    goldstein_price,
    levi,
    bukin_n6,
    booth,
    matyas,
    three_hump_camel,
    easom,
    mccormick,
    styblinski_tang,
    schaffer_n2
]



def optimize_objective(obj, delta, x, method='L-BFGS-B'):
    """Optimize the objective function with given bounds using the specified method."""
    x = np.asarray(x)
    delta = np.asarray(delta)

    lb = x - delta
    ub = x + delta
    bounds = [(lb_i, ub_i) for lb_i, ub_i in zip(lb, ub)]

    def jax_objective(x):
        """Objective function using JAX."""
        return obj(*x)

    def jax_gradient(x):
        """Compute the gradient of the objective function using JAX."""
        x_jnp = jnp.array(x)
        return grad(jax_objective)(x_jnp).tolist()  # Convert JAX array to list for compatibility

    def jax_objective_np(x):
        """Convert JAX objective to a NumPy-compatible function."""
        return float(jax_objective(x))

    # Convert gradient to a function compatible with scipy.optimize.minimize
    def objective_np(x_np):
        return jax_objective_np(x_np)
    
    def gradient_np(x_np):
        return jax_gradient(x_np)
    
    # Use scipy.optimize.minimize with the 'L-BFGS-B' method for bounded optimization
    result = minimize(objective_np, x, jac=gradient_np, bounds=bounds, method=method, options={'disp': True})
    if not result.success:
        print("Optimization failed:", result.message)
    x_opt = result.x

    return x_opt

if __name__ == "__main__":
    x_initial = np.array([1.0, 5.0])
    delta = np.array([4.0, 4.0])
    
    x_opt = optimize_objective(rosenbrock, delta, x_initial, method='trust-constr')
    print("Optimal solution:", x_opt)
