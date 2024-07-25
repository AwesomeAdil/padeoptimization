import jax.numpy as jnp
import jax
jax.config.update('jax_enable_x64', True)
def extract_primal(value):
    if hasattr(value, 'primal'):
        return value.primal
    return value

def fr(*x):
    return jnp.sin(jnp.sum(jnp.array(x)))

def sin(*x):
    y = jnp.array(x)
    return jnp.sum(jnp.sin(y[0]) + jnp.cos(y[1]))

def f(*args):
    return jnp.exp(jnp.sum(jnp.array(args)))  # e^(x+y)

def exp(*args):
    try:
        x = jnp.array(args)
        return jnp.exp(jnp.sum(x))
    except Exception as e:
        print(e)
        return jnp.inf

def siner(*args):
    x = jnp.array(args)
    return jnp.sum(jnp.sin(x[0]) + jnp.cos(x[1]))

def ackley(*args):
    x = jnp.array(args)
    val = -20.0 * jnp.exp(-0.2 * jnp.sqrt(0.5 * (x[0]**2 + x[1]**2))) - jnp.exp(0.5 * (jnp.cos(2.0 * jnp.pi * x[0]) + jnp.cos(2.0 * jnp.pi * x[1]))) + jnp.e + 20
    return val

def sphere(*args):
    try:
        x = jnp.array(args)
        return jnp.sum(x**2)
    except Exception as e:
        print(e)
        return jnp.inf

def rastrigin(*args):
    try:
        x = jnp.array(args)
        return 10 * len(x) + jnp.sum(x**2 - 10 * jnp.cos(2 * jnp.pi * x))
    except Exception as e:
        print(e)
        return jnp.inf

def rosenbrock(*args):
    try:
        x = jnp.array(args)
        return jnp.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
    except Exception as e:
        print(e)
        return jnp.inf

def beale(*args):
    try:
        if len(args) < 2:
            raise ValueError(f"Beale function requires exactly 2 dimensions, but {len(args)} were provided.")
        x = jnp.array(args)
        return (1.5 - x[0] + x[0] * x[1])**2 + (2.25 - x[0] + x[0] * x[1]**2)**2 + (2.625 - x[0] + x[0] * x[1]**3)**2
    except ValueError as e:
        print(e)
        return jnp.inf

def goldstein_price(*args):
    try:
        if len(args) < 2:
            raise ValueError(f"Goldstein-Price function requires exactly 2 dimensions, but {len(args)} were provided.")
        x = jnp.array(args)
        return (1 + (x[0] + x[1] + 1)**2 * (19 - 14 * x[0] + 3 * x[0]**2 - 14 * x[1] + 6 * x[0] * x[1] + 3 * x[1]**2)) * (30 + (2 * x[0] - 3 * x[1])**2 * (18 - 32 * x[0] + 12 * x[0]**2 + 48 * x[1] - 36 * x[0] * x[1] + 27 * x[1]**2))
    except ValueError as e:
        print(e)
        return jnp.inf

def levi(*args):
    try:
        x = jnp.array(args)
        return jnp.sin(3 * jnp.pi * x[0])**2 + (x[0] - 1)**2 * (1 + jnp.sin(3 * jnp.pi * x[1])**2) + (x[1] - 1)**2 * (1 + jnp.sin(2 * jnp.pi * x[1])**2)
    except Exception as e:
        print(e)
        return jnp.inf

def bukin_n6(*args):
    try:
        if len(args) < 2:
            raise ValueError(f"Bukin function N. 6 requires exactly 2 dimensions, but {len(args)} were provided.")
        x = jnp.array(args)
        return 100 * jnp.sqrt(jnp.abs(x[1] - 0.01 * x[0]**2)) + 0.01 * jnp.abs(x[0] + 10)
    except ValueError as e:
        print(e)
        return jnp.inf

def booth(*args):
    try:
        if len(args) < 2:
            raise ValueError(f"Booth function requires exactly 2 dimensions, but {len(args)} were provided.")
        x = jnp.array(args)
        return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2
    except ValueError as e:
        print(e)
        return jnp.inf

def matyas(*args):
    try:
        if len(args) < 2:
            raise ValueError(f"Matyas function requires exactly 2 dimensions, but {len(args)} were provided.")
        x = jnp.array(args)
        return 0.26 * (x[0]**2 + x[1]**2) - 0.48 * x[0] * x[1]
    except ValueError as e:
        print(e)
        return jnp.inf

def three_hump_camel(*args):
    try:
        if len(args) < 2:
            raise ValueError(f"Three-Hump Camel function requires exactly 2 dimensions, but {len(args)} were provided.")
        x = jnp.array(args)
        return 2 * x[0]**2 - 1.05 * x[0]**4 + x[0]**6 / 6 + x[0] * x[1] + x[1]**2
    except ValueError as e:
        print(e)
        return jnp.inf

def easom(*args):
    try:
        if len(args) < 2:
            raise ValueError(f"Easom function requires exactly 2 dimensions, but {len(args)} were provided.")
        x = jnp.array(args)
        return -jnp.cos(x[0]) * jnp.cos(x[1]) * jnp.exp(-((x[0] - jnp.pi)**2 + (x[1] - jnp.pi)**2))
    except ValueError as e:
        print(e)
        return jnp.inf

def mccormick(*args):
    try:
        if len(args) < 2:
            raise ValueError(f"McCormick function requires exactly 2 dimensions, but {len(args)} were provided.")
        x = jnp.array(args)
        return jnp.sin(x[0] + x[1]) + (x[0] - x[1])**2 - 1.5 * x[0] + 2.5 * x[1] + 1
    except ValueError as e:
        print(e)
        return jnp.inf

def styblinski_tang(*args):
    try:
        x = jnp.array(args)
        return 0.5 * jnp.sum(x**4 - 16 * x**2 + 5 * x)
    except Exception as e:
        print(e)
        return jnp.inf

def schaffer_n2(*args):
    try:
        if len(args) < 2:
            raise ValueError(f"Schaffer Function N. 2 requires exactly 2 dimensions, but {len(args)} were provided.")
        x = jnp.array(args)
        return 0.5 + (jnp.sin(x[0]**2 - x[1]**2)**2 - 0.5) / (1 + 0.001 * (x[0]**2 + x[1]**2))**2
    except ValueError as e:
        print(e)
        return jnp.inf
