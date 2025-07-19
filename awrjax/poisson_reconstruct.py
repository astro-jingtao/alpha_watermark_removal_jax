import jax.numpy as jnp
import numpy as np
from jax import Array
from scipy import fftpack
from poissonpy.RGB import solve_RGB_region
from poissonpy.functional import get_np_gradient

from .jaxcv.filter import convolve_with_kernel, grad, grad_np, sobel, sobel_cv2


def poisson_reconstruct(gradx,
                        grady,
                        boundary_image=None,
                        boundary_zero=True,
                        input_scale=255,
                        output_scale=None):
    """
    gradx, grady: gradient images, dx should be considered, i.e. soble()/8

	Iterative algorithm for Poisson reconstruction. 
	Given the gradx and grady values, find laplacian, and solve for image
	Also return the squared difference of every step.
	h = convergence rate
	"""

    gradx = np.asarray(gradx)
    grady = np.asarray(grady)

    if input_scale == 255:
        gradx = gradx / 255.0
        grady = grady / 255.0
    elif input_scale != 1:
        raise ValueError("input_scale must be 1 or 255")

    if output_scale is None:
        output_scale = input_scale
    elif not output_scale in (255, 1):
        raise ValueError("output_scale must be 1 or 255")

    # fxx = sobel_cv2(gradx, axis='x').astype(np.float32) / 8
    # fyy = sobel_cv2(grady, axis='y').astype(np.float32) / 8
    fxx, _ = get_np_gradient(gradx, forward=False, batch=True, padding=True)
    _, fyy = get_np_gradient(grady, forward=False, batch=True, padding=True)
    # fxx = jnp.gradient(gradx, axis=1)
    # fyy = jnp.gradient(grady, axis=0)
    laplacian = fxx + fyy  # type: ignore

    mask = np.zeros_like(gradx[:, :, 0])
    mask[1:-1, 1:-1] = 1.

    if boundary_zero:
        boundary_image = np.zeros(laplacian.shape)
    elif boundary_image is None:
        raise ValueError(
            "Boundary image must be provided if boundary_zero is False")
    elif boundary_image.shape != laplacian.shape:
        raise ValueError(
            "Boundary image must have the same shape as the laplacian")

    return solve_RGB_region(mask, laplacian, boundary_image) * output_scale
