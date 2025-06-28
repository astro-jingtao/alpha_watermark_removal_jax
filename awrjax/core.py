import jax.numpy as jnp
from jax import jit
from jax import Array
import numpy as np
import cv2

from tqdm import tqdm

from .jaxcv.filter import sobel, convolve_with_kernel
from .utils import binarize_img, normalize_img
# jax version can not work, let's use numpy version
from .ref.alpha_matte_np import closed_form_matte

# pylint: disable=pointless-string-statement
""" 
------------> (j, x)
|
|
|
|
|
v
(i, y)
"""


def estimate_watermark(J, verbose=False):
    """
    Input:
        J: list of images
    Output:
        gradx_med: median of gradients in x-direction
        grady_med: median of gradients in y-direction
        gradx_arr: gradients in x-direction
        grady_arr: gradients in y-direction
    """

    # Ensure images are on the device (e.g., GPU) if JAX is configured to use one
    J = [jnp.asarray(img) for img in J]

    # Compute gradients
    if verbose:
        print(f"Computing gradients of {len(J)} images")

    # Define a jitted function to compute Sobel gradients
    @jit
    def compute_gradients(J):
        gradx = [sobel(j, axis='x') for j in J]
        grady = [sobel(j, axis='y') for j in J]
        return jnp.array(gradx), jnp.array(grady)

    # @jit
    # def compute_gradients(J):
    #     gradx = [jnp.gradient(j, axis=1) for j in J]
    #     grady = [jnp.gradient(j, axis=0) for j in J]
    #     return jnp.array(gradx), jnp.array(grady)

    gradx_arr, grady_arr = compute_gradients(J)

    # Compute median of grads
    if verbose:
        print("Computing median gradients.")
    gradx_med = jnp.median(gradx_arr, axis=0)
    grady_med = jnp.median(grady_arr, axis=0)

    return gradx_med, grady_med, gradx_arr, grady_arr


def box_watermark(gradx, grady, threshold=0.5, boundary_size=2):
    """
	gradx: median of gradients in x-direction
    grady: median of gradients in y-direction
	Assumes the gradx and grady to be in 3 channels
	@param: threshold - gives the threshold param
	@param: boundary_size - boundary around cropped image
	"""
    grad_mod = jnp.sqrt(jnp.square(gradx) + jnp.square(grady))
    grad_mod = normalize_img(jnp.mean(grad_mod, axis=2))
    grad_bin = binarize_img(grad_mod, threshold=threshold)
    i, j = jnp.where(grad_bin == 1)

    ni, nj = grad_mod.shape

    i_min = max(jnp.min(i) - boundary_size, 0)
    i_max = min(jnp.max(i) + boundary_size + 1, ni)
    j_min = max(jnp.min(j) - boundary_size, 0)
    j_max = min(jnp.max(j) + boundary_size + 1, nj)

    return i_min, i_max, j_min, j_max


def crop_watermark(gradx, grady, threshold=0.5, boundary_size=2):

    i_min, i_max, j_min, j_max = box_watermark(gradx,
                                               grady,
                                               threshold=threshold,
                                               boundary_size=boundary_size)

    return gradx[i_min:i_max, j_min:j_max, :], grady[i_min:i_max,
                                                     j_min:j_max, :]


def poisson_reconstruct(gradx,
                        grady,
                        num_iters=100,
                        h=0.1,
                        boundary_image=None,
                        boundary_zero=True):
    """
	Iterative algorithm for Poisson reconstruction. 
	Given the gradx and grady values, find laplacian, and solve for image
	Also return the squared difference of every step.
	h = convergence rate
	"""
    fxx = sobel(gradx, axis='x')
    fyy = sobel(grady, axis='y')
    # fxx = jnp.gradient(gradx, axis=1)
    # fyy = jnp.gradient(grady, axis=0)
    laplacian: Array = fxx + fyy  # type: ignore
    m, n, p = laplacian.shape

    if boundary_zero:
        est = jnp.zeros(laplacian.shape)
    elif boundary_image is None:
        raise ValueError(
            "Boundary image must be provided if boundary_zero is False")
    elif boundary_image.shape != laplacian.shape:
        raise ValueError(
            "Boundary image must have the same shape as the laplacian")
    else:
        est = boundary_image.copy()

    est = est.at[1:-1,
                 1:-1, :].set(jnp.array(np.random.random((m - 2, n - 2, p))))
    loss = []

    for _ in range(num_iters):
        old_est = est.copy()
        est = est.at[1:-1, 1:-1, :].set(
            0.25 *
            (est[0:-2, 1:-1, :] + est[1:-1, 0:-2, :] + est[2:, 1:-1, :] +
             est[1:-1, 2:, :] - h * h * laplacian[1:-1, 1:-1, :]))
        error = np.sum(np.square(est - old_est))
        loss.append(error)

    return est, loss


def detect_watermark(img,
                     gradx,
                     grady,
                     thresh_low=200,
                     thresh_high=220,
                     printval=False):
    """
	Compute a verbose edge map using Canny edge detector, take its magnitude.
	Assuming cropped values of gradients are given.
	Returns image, start and end coordinates
	"""
    gradm = np.average(np.sqrt(np.square(gradx) + np.square(grady)), axis=2)

    img_edgemap = jnp.asarray(cv2.Canny(np.asarray(img, dtype=np.uint8),
                                        thresh_low, thresh_high),
                              dtype=jnp.float32)

    chamfer_dist = convolve_with_kernel(img_edgemap,
                                        gradm[::-1, ::-1],
                                        method='fft')

    # return chamfer_dist, img_edgemap, gradm

    rect = gradm.shape
    index = jnp.unravel_index(jnp.argmax(chamfer_dist), img.shape[:-1])
    if printval:
        print(index)

    i, j = (index[0] - rect[0] / 2), (index[1] - rect[1] / 2)
    return i, j, rect[0], rect[1]


def estimate_normalized_alpha(J, Wm, threshold=0.5):
    Wm = binarize_img(normalize_img(np.average(Wm, axis=-1)),
                      threshold=threshold)
    Wm = (Wm * 255).astype(np.uint8)
    Wm = np.stack([np.asarray(Wm)] * 3, axis=-1)

    num = len(J)
    alpha_lst = []

    print(f"Estimating normalized alpha using {num} images.")
    # for all images, calculate alpha
    for idx in tqdm(range(num)):
        alpha = closed_form_matte(np.asarray(J[idx]), Wm)
        alpha_lst.append(alpha)

    alpha = np.median(alpha_lst, axis=0)
    return jnp.asarray(alpha)
