import jax.numpy as jnp
from jax.scipy.stats import norm


def gaussian_kernel(sigma, size):
    """
    Returns a Gaussian kernel of given size and standard deviation.
    :param sigma: standard deviation of the Gaussian kernel
    :param size: size of the kernel
    """
    if size % 2 == 0:
        raise ValueError("Kernel size must be odd")

    n = size // 2
    x = jnp.linspace(-n, n, n * 2 + 1)
    kernel_1d = norm.pdf(x, 0, sigma)
    kernel = jnp.outer(kernel_1d, kernel_1d)
    kernel /= jnp.sum(kernel)
    return kernel


def sobel_kernel(axis):
    """
    Returns a Sobel kernel of given size.
    """
    if axis in (0, 'y', 'i'):
        kernel = jnp.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    elif axis in (1, 'x', 'j'):
        kernel = jnp.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    else:
        raise ValueError("Invalid axis for Sobel kernel")

    return kernel
