import jax
import jax.numpy as jnp

def quadratic_taylor_series(func, x0):
    """
    Returns a function that computes the quadratic Taylor series expansion of `func` at `x0`.
    
    Parameters:
    func: callable
        The multivariate function to expand.
    x0: array-like
        The point at which to expand the Taylor series.
        
    Returns:
    taylor_series: callable
        A function that computes the Taylor series expansion at any given point.
    """
    x0 = jnp.asarray(x0)
    
    # Compute the function value at x0
    f0 = func(x0)
    
    # Compute the gradient at x0
    grad_f = jax.grad(func)
    grad_f0 = grad_f(x0)
    
    # Compute the Hessian at x0
    hessian_f = jax.jacfwd(jax.grad(func))
    hessian_f0 = hessian_f(x0)
    
    # Define the quadratic Taylor series expansion
    def taylor_series(x):
        dx = jnp.asarray(x) - x0
        return f0 + jnp.dot(grad_f0, dx) + 0.5 * jnp.dot(dx, jnp.dot(hessian_f0, dx))
    
    return taylor_series

if __name__ == "__main__":
    # Example usage
    def f(x):
        return jnp.sin(x[0]) * jnp.cos(x[1]) + x[0]**2 + x[1]**2

    x0 = jnp.array([1.0, 2.0])
    taylor_series_f = quadratic_taylor_series(f, x0)

    # Test the Taylor series expansion
    x_test = jnp.array([1.1, 2.1])
    print("Function value at x_test:", f(x_test))
    print("Taylor series approximation at x_test:", taylor_series_f(x_test))
