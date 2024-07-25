import jax
import numpy as np
from jax import grad
import jax.numpy as jnp
from quad_tay import quadratic_taylor_series  # Ensure this path is correct
from subproblem import optimize_objective  # Import the function from subproblem.py
from optimization_functions import siner, rosenbrock
from nested_pade import nested_pade_sim
import pdb
def trust_region_proper(f, x0, delta, max_iter, tol, hat, thresh, method, opt_method='trust-constr'):
    x = x0
    history = [x0]
    print(delta)

    for ind in range(max_iter):
        print(ind)
        if method == "tay":
            m = quadratic_taylor_series(f, x)
        else:
            print(x)
            m = nested_pade_sim(f, x, 5)

        f_handle = lambda v: m(v)
        initial_guess = x

        def objective(*v):
            v = jnp.array(v)
            return f_handle(v)
        
        # Compute the optimized x using the abstracted function
        x_opt = optimize_objective(f, delta, x, method=opt_method)
        print("SPOT",x, x_opt)
        ip = f(*x)
        fp = f(*x_opt)
        im = m(x)
        fm = m(x_opt)
        
        print(ip)
        print(fp)
        print(im)
        print(fm)

        num = ip - fp
        den = im - fm
    
        direction = x_opt - x
        direction = direction / jnp.linalg.norm(direction)

        if jnp.abs(num) < 1e-9 and jnp.abs(den) < 1e-9:
            return x, history

        if jnp.abs(den) < 1e-9:
            if num > 0:
                ratio = 1
            else:
                ratio = 0
        else:
            ratio = num / den

        if ratio < 0.25:
            delta = delta / 4
        else:
            if ratio > 0.75 and delta == jnp.linalg.norm(x_opt - x, jnp.inf):
                delta = min(2 * delta, hat)

        if jnp.linalg.norm(x_opt - x, jnp.inf) < tol:
            return x, history

        if ratio > thresh:
            x = x_opt

        history.append(x)

    return x, history

# Example usage:
# Define a sample function to minimize
def sample_function(x):
    return jnp.sum(x ** 2)  # This returns a scalar

if __name__=="__main__":
    # Initial parameters
    x0 = jnp.array((2.0, 1.0))
    delta = 1.0
    max_iter = 100
    tol = 1e-6
    hat = 2.0
    thresh = 0.1
    method = "pad"
    
    result, history = trust_region_proper(rosenbrock, x0, delta, max_iter, tol, hat, thresh, method)
    print("MOOOOOOOO")
    print("Result:", result)
    print("History:", history)
