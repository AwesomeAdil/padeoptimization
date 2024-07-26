import jax.numpy as jnp
import jax
from jax import grad
from math import factorial
import numpy as np
from jax.tree_util import Partial
from functools import partial
from optimization_functions import ackley, exp, siner, sin, fr, rosenbrock
from optimization_functions import (
    fr, sin, exp, ackley, sphere, rastrigin, rosenbrock, beale, goldstein_price, levi, bukin_n6,
    booth, matyas, three_hump_camel, easom, mccormick, styblinski_tang, schaffer_n2
)
import pdb
from grapher import plot
import scipy.linalg
import time
t_time = 0
o_time = 0
p_time = 0

# Define a small epsilon value for numerical stability
EPSILON = 1e-8
# Regularization term to stabilize the solver
REGULARIZATION_TERM = 1e-5
# Clipping threshold for extreme values
CLIP_THRESHOLD = 1e3

def divide_by_fact(d, deg, *args):
    return d(*args) / factorial(deg)

def taylor_series_multidim(f, a, degree):
    coeffs = []
    derivative = f
    for n in range(degree):
        if n == 0:
            # Directly append the function with the point 'a'
            coeffs.append(Partial(f, a[0]))
        else:
            # Compute the n-th derivative
            derivative = grad(lambda *args: derivative(*args).sum(), argnums=0)
            # Create a lambda function for the n-th coefficient
            x = Partial(derivative, a[0])
            coeff_func = Partial(divide_by_fact, x, n)
            coeffs.append(coeff_func)

    # Print the types and contents of the coefficients
    #print([cf(*(a[1:])) for cf in coeffs])
    return coeffs

def make_pade_approximation(coeffs, m, n, a):
    global p_time
    start = time.time()
    # Print and check the types of the results
    P = []
    vals = [cf(*[a]) for cf in coeffs]
    #print('vals', vals)

    # Continue with the rest of your code
    C = np.zeros((n, n))
    for i in range(n):
        C[i, :] = [cf for cf in vals[m-n+i+1:m+i+1]]

    # Check for singularity and condition number
    cond_number = np.linalg.cond(C)
    if cond_number > 1 / EPSILON:
        print("Warning: Matrix is nearly singular or ill-conditioned, cond number:", cond_number)
        # Apply regularization to stabilize
        C += np.eye(n) * REGULARIZATION_TERM

    try:
        Q = jnp.zeros(n+1, dtype=jnp.float64)  # Use float64 for higher precision
        Q = Q.at[1:].set(-jnp.linalg.solve(C, np.array([cf for cf in vals[m+1:m+n+1]])))
        Q = jnp.concatenate((jnp.array((Q[0],)), jnp.array(Q[-1:0:-1])), axis=0)
        Q = Q.at[0].set(1.0)
    except jnp.linalg.LinAlgError as e:
        print("Linear algebra error during solve:", e)
        return None

    Q = jnp.where(jnp.isnan(Q), 0, Q)
    Q = jnp.clip(Q, -CLIP_THRESHOLD, CLIP_THRESHOLD)  # Clip extreme values

    for i in range(m+1):
        res = vals[i]
        for j in range(i):
            res += vals[i-j-1] * Q[j+1]
        P.append(res)
    
    P = jnp.array(P, dtype=jnp.float64)  # Convert to a JAX array after confirming numeric results
    
    def res(x0):
        x0 = jnp.array(x0, dtype=jnp.float64)  # Use float64 for higher precision
        # Compute the numerator
        num = (P[0] + P[1] * x0 + P[2] * x0**2)

        # Compute the denominator
        denom = (Q[0] + Q[1] * x0 + Q[2] * x0**2) + EPSILON
        
        return num / denom
    end = time.time()
    p_time += end-start
    print(end-start)
    return res

def B1(f1, f2, f3, f4, *args):
    s1 = f1(*args) if callable(f1) else f1
    s2 = f2(*args) if callable(f2) else f2
    s3 = f3(*args) if callable(f3) else f3
    s4 = f4(*args) if callable(f4) else f4
    
    denom = (s3 * s1 - s2 * s2)
    if jnp.abs(denom) < EPSILON:
        return jnp.inf  # or handle accordingly
    return (s3 * s2 - s4 * s1) / denom

def B2(f1, f2, f3, f4, *args):
    s1 = f1(*args) if callable(f1) else f1
    s2 = f2(*args) if callable(f2) else f2
    s3 = f3(*args) if callable(f3) else f3
    s4 = f4(*args) if callable(f4) else f4

    denom = (s2 * s2 - s1 * s3)
    if jnp.abs(denom) < EPSILON:
        return jnp.inf  # or handle accordingly
    return (s3 * s3 - s4 * s2) / denom

def res(P, Q, *args):
    if len(args) == 0:
        raise ValueError("At least one argument is required.")

    args = args[0]
    x0 = args[0]  # Extract the first argument
    remaining_args = args[1:]  # The rest of the arguments
    
    num = (P[0](*remaining_args) +
            P[1](*remaining_args) * x0 +
            P[2](*remaining_args) * x0**2)
            
    # Compute the denominator
    denom = (Q[0] +
            Q[1](*remaining_args) * x0 +
            Q[2](*remaining_args) * x0**2) + EPSILON

    return num / denom

def addon(f, q, *args):
    x = tuple(args)
    return f(x) + q(x)

def multon(f, q, *args):
    x = tuple(args)
    return f(x) * q(x)

def nested_pade_sim(f, a, deg):
    global t_time
    a = jnp.array(a, dtype=jnp.float64)  # Use float64 for higher precision
    start = time.time()
    coeffs = taylor_series_multidim(f, a, deg)
    end = time.time()
    t_time += end-start

    if len(a) == 1:
        return make_pade_approximation(coeffs, 2, 2, a[0])
    else:
        a = a[1:]  # Update to handle the dimension reduction
        Q = []
        Q.append(1.0)
        Q.append(nested_pade_sim(Partial(B1, coeffs[1], coeffs[2], coeffs[3], coeffs[4]), a, deg))
        Q.append(nested_pade_sim(Partial(B2, coeffs[1], coeffs[2], coeffs[3], coeffs[4]), a, deg))
        B1C0 = Partial(multon, Q[1], coeffs[0])
        B1C1 = Partial(multon, Q[1], coeffs[1])
        B2C0 = Partial(multon, Q[2], coeffs[0])
        
        # Remove the debugger stop
        # pdb.set_trace()

        # Handle Partial function wrapping for multi-dimensionality
        P = [
            nested_pade_sim(coeffs[0], a, deg), 
            nested_pade_sim(Partial(addon, coeffs[1], B1C0), a, deg),
            nested_pade_sim(Partial(addon, coeffs[2], Partial(addon, B1C1, B2C0)), a, deg)
        ]

        return Partial(res, P, Q)

if __name__ == "__main__":
    # Define the point of expansion and degrees
    a = (0.0, 0.0)  # Example point in n-dimensional space
    deg = 5  # Example degree for Taylor series expansion

    for i in range(5):
        t_time = 0
        p_time = 0
        o_time = 0
        start = time.time()
        
        # Call the nested Pade simulation
        result = nested_pade_sim(exp, a, deg)
        
        end = time.time()
        o_time = end - start
        
        print(f"TIMES, T-Time: {t_time}, O-Time: {o_time}, P-Time: {p_time}")
