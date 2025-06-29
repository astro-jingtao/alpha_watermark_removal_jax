import jax.numpy as jnp
from jax import jit
from jax import Array
import numpy as np
import cv2
# from jax.experimental import sparse
from scipy.sparse import coo_matrix, diags
from scipy.sparse import vstack as sp_vstack
from scipy.sparse import hstack as sp_hstack
from scipy.sparse.linalg import spsolve

from tqdm import tqdm

from .jaxcv.filter import sobel, convolve_with_kernel, grad
from .utils import binarize_img, normalize_img, spdiag, COO_spsolve
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


def estimate_normalized_alpha(J,
                              Wm,
                              _lambda=100,
                              threshold=0.7,
                              std_wm_thresh=0.2,
                              std_bg_thresh=0.4):

    prior, prior_confidence = get_matte_prior(J, Wm, _lambda, threshold,
                                              std_wm_thresh, std_bg_thresh)

    num = len(J)
    alpha_lst = []

    print(f"Estimating normalized alpha using {num} images.")
    # for all images, calculate alpha
    for idx in tqdm(range(num)):
        alpha = closed_form_matte(
            np.asarray(J[idx]),
            prior=np.asarray(prior),
            prior_confidence=np.asarray(prior_confidence))
        alpha_lst.append(alpha)

    alpha = np.median(alpha_lst, axis=0)
    return jnp.asarray(alpha)


def get_matte_prior(J, Wm, _lambda, threshold, std_wm_thresh, std_bg_thresh):
    Wm = binarize_img(jnp.mean(normalize_img(Wm, axis=(0, 1)), axis=-1),
                      threshold=threshold)
    std_map = normalize_img(jnp.std(jnp.asarray(J), axis=0).mean(axis=-1))
    is_watermark = (std_map < std_wm_thresh) | (Wm == 1)
    is_background = (std_map > std_bg_thresh) & (Wm == 0)

    prior = is_watermark.astype(np.float64)
    prior_confidence = np.ones_like(prior) * _lambda
    prior_confidence[~is_watermark & ~is_background] = 0
    return prior, prior_confidence


def estimate_blend_factor(J, Wm, alpha_norm):
    J = jnp.asarray(J)
    K, m, n, p = J.shape
    Jm = (J - Wm)
    gx_jm = []
    gy_jm = []

    for i in range(K):
        gx_jm.append(sobel(Jm[i], axis='x'))
        gy_jm.append(sobel(Jm[i], axis='y'))

    gx_jm = jnp.array(gx_jm)
    gy_jm = jnp.array(gy_jm)

    Jm_grad = jnp.sqrt(gx_jm**2 + gy_jm**2)

    est_Ik = alpha_norm[..., jnp.newaxis] * jnp.median(J, axis=0)
    # est_Ik = jnp.median(J, axis=0)
    gx_estIk = sobel(est_Ik, axis='x')
    gy_estIk = sobel(est_Ik, axis='y')
    estIk_grad = np.sqrt(gx_estIk**2 + gy_estIk**2)

    C = []
    for i in range(3):
        c_i = jnp.sum(Jm_grad[:, :, :, i] * estIk_grad[:, :, i]) / jnp.sum(
            jnp.square(estIk_grad[:, :, i])) / K
        C.append(c_i)

    return C, est_Ik


# def estimate_blend_factor(J, Wm, alpha_norm):
#     J = jnp.asarray(J)
#     K, m, n, p = J.shape
#     Jm = (J - Wm)

#     est_Ik = jnp.median(Jm, axis=0)

#     return C, est_Ik


def solve_images_jax(J,
                     Wm,
                     alpha,
                     W_init,
                     gamma=1,
                     beta=1,
                     lambda_w=0.005,
                     lambda_i=1,
                     lambda_a=0.01,
                     iters=4):
    '''
    Master solver, follows the algorithm given in the supplementary.
    W_init: Initial value of W
    Step 1: Image Watermark decomposition
    '''
    # prepare variables
    J = np.asarray(J)
    K, m, n, p = J.shape
    size = m * n * p

    sobelx = get_xSobel_matrix(m, n, p)
    sobely = get_ySobel_matrix(m, n, p)
    Ik = []
    Wk = []
    for i in range(K):
        Ik.append(J[i] - Wm)
        Wk.append(W_init.copy())

    # This is for median images
    W = W_init.copy()

    Wm_old = Wm.copy()

    Wm_gx = sobel(Wm, axis='x')
    Wm_gy = sobel(Wm, axis='y')

    # Iterations
    for ii in range(iters):

        print("------------------------------------")
        print(f"Iteration:{ii}")

        # Step 1
        print("Step 1")
        alpha_gx = sobel(alpha, axis='x')
        alpha_gy = sobel(alpha, axis='y')

        

        cx = diags(np.abs(alpha_gx).flatten())
        cy = diags(np.abs(alpha_gy).flatten())

        alpha_diag = diags(alpha.flatten())
        alpha_bar_diag = diags((1 - alpha).flatten())

        for i in range(K):
            # prep vars
            Wkx = sobel(Wk[i], axis='x')
            Wky = sobel(Wk[i], axis='y')

            Ikx = sobel(Ik[i], axis='x')
            Iky = sobel(Ik[i], axis='y')

            alphaWk = alpha * Wk[i]
            alphaWk_gx = sobel(alphaWk, axis='x')
            alphaWk_gy = sobel(alphaWk, axis='y')

            phi_data = diags(
                func_phi_deriv(
                    np.square(alpha * Wk[i] + (1 - alpha) * Ik[i] -
                              J[i]).reshape(-1)))
            phi_f = diags(
                func_phi_deriv(((Wm_gx - alphaWk_gx)**2 +
                                (Wm_gy - alphaWk_gy)**2).reshape(-1)))
            phi_aux = diags(func_phi_deriv(np.square(Wk[i] - W).reshape(-1)))
            phi_rI = diags(
                func_phi_deriv(
                    np.abs(alpha_gx) * (Ikx**2) + np.abs(alpha_gy) *
                    (Iky**2)).reshape(-1))
            phi_rW = diags(
                func_phi_deriv(
                    np.abs(alpha_gx) * (Wkx**2) + np.abs(alpha_gy) *
                    (Wky**2)).reshape(-1))

            L_i = sobelx.T @ (cx * phi_rI) @ (sobelx) + sobely.T @ (
                cy * phi_rI) @ (sobely)
            L_w = sobelx.T @ (cx * phi_rW) @ (sobelx) + sobely.T @ (
                cy * phi_rW) @ (sobely)
            L_f = sobelx.T @ (phi_f) @ (sobelx) + sobely.T @ (phi_f) @ (sobely)
            A_f = alpha_diag.T @ (L_f) @ (alpha_diag) + gamma * phi_aux

            bW = alpha_diag @ (phi_data) @ (J[i].reshape(-1)) + beta * L_f @ (
                Wm.reshape(-1)) + gamma * phi_aux @ (W.reshape(-1))
            bI = alpha_bar_diag @ (phi_data) @ (J[i].reshape(-1))

            A = sp_vstack([sp_hstack([(alpha_diag**2)*phi_data + lambda_w*L_w + beta*A_f, alpha_diag*alpha_bar_diag*phi_data]), \
                         sp_hstack([alpha_diag*alpha_bar_diag*phi_data, (alpha_bar_diag**2)*phi_data + lambda_i*L_i])]).tocsr()

            b = np.hstack([bW, bI])
            # return A, b
            x = spsolve(A, b)

            Wk[i] = np.clip(x[:size].reshape(m, n, p), 0, 255)
            Ik[i] = np.clip(x[size:].reshape(m, n, p), 0, 255)

            print(i)

        # Step 2
        print("Step 2")
        W = np.median(np.asarray(Wk), axis=0)

        # Step 3
        print("Step 3")
        W_diag = diags(W.reshape(-1))

        for i in range(K):
            alphaWk = alpha * Wk[i]
            alphaWk_gx = sobel(alphaWk, axis='x')
            alphaWk_gy = sobel(alphaWk, axis='y')
            phi_f = diags(
                func_phi_deriv(((Wm_gx - alphaWk_gx)**2 +
                                (Wm_gy - alphaWk_gy)**2).reshape(-1)))

            phi_kA = diags(((func_phi_deriv(
                (((alpha * Wk[i] + (1 - alpha) * Ik[i] - J[i])**2)))) *
                            ((W - Ik[i])**2)).reshape(-1))
            phi_kB = (((func_phi_deriv(
                (((alpha * Wk[i] + (1 - alpha) * Ik[i] - J[i])**2)))) *
                       (W - Ik[i]) * (J[i] - Ik[i])).reshape(-1))

            phi_alpha = diags(
                func_phi_deriv(alpha_gx**2 + alpha_gy**2).reshape(-1))
            L_alpha = sobelx.T @ (phi_alpha @ (sobelx)) + sobely.T @ (
                phi_alpha @ (sobely))

            L_f = sobelx.T @ (phi_f) @ (sobelx) + sobely.T @ (phi_f) @ (sobely)
            A_tilde_f = W_diag.T @ (L_f) @ (W_diag)
            # Ax = b, setting up A
            if i == 0:
                A1 = phi_kA + lambda_a * L_alpha + beta * A_tilde_f
                b1 = phi_kB + beta * W_diag @ (L_f) @ (Wm.reshape(-1))
            else:
                A1 += (phi_kA + lambda_a * L_alpha + beta * A_tilde_f)
                b1 += (phi_kB + beta * W_diag.T @ (L_f) @ (Wm.reshape(-1)))

        alpha = spsolve(A1, b1).reshape(m, n, p)
        alpha = np.clip(np.stack([np.mean(alpha, axis=-1)] * 3, axis=-1), 0,
                        1)
        Wm = alpha * W
        print(np.linalg.norm(Wm - Wm_old))
        Wm_old = Wm.copy()

    return (Wk, Ik, W, alpha)


def func_phi(X, epsilon=1e-3):
    return np.sqrt(X + epsilon**2)


def func_phi_deriv(X, epsilon=1e-3):
    return 0.5 / func_phi(X, epsilon)


# get sobel coordinates for y
def _get_ysobel_coord(coord, shape):
    i, j, k = coord
    m, n, p = shape
    # return [(i - 1, j, k, -2), (i - 1, j - 1, k, -1), (i - 1, j + 1, k, -1),
    #         (i + 1, j, k, 2), (i + 1, j - 1, k, 1), (i + 1, j + 1, k, 1)]
    return [(i - 1, j, k, -1), (i + 1, j, k, 1)]


# get sobel coordinates for x
def _get_xsobel_coord(coord, shape):
    i, j, k = coord
    m, n, p = shape
    # return [(i, j - 1, k, -2), (i - 1, j - 1, k, -1), (i - 1, j + 1, k, -1),
    #         (i, j + 1, k, 2), (i + 1, j - 1, k, 1), (i + 1, j + 1, k, 1)]
    return [(i, j - 1, k, -1), (i, j + 1, k, 1)]


# filter
def _filter_list_item(coord, shape):
    i, j, k, v = coord
    m, n, p = shape
    if i >= 0 and i < m and j >= 0 and j < n:
        return True


# Change to ravel index
# also filters the wrong guys
def _change_to_ravel_index(li, shape):
    li = filter(lambda x: _filter_list_item(x, shape), li)
    i, j, k, v = zip(*li)
    return zip(np.ravel_multi_index((i, j, k), shape), v)


def get_ySobel_matrix(m, n, p):
    size = m * n * p
    shape = (m, n, p)
    i, j, k = np.unravel_index(np.arange(size), (m, n, p))
    ijk = zip(list(i), list(j), list(k))
    ijk_nbrs = map(lambda x: _get_ysobel_coord(x, shape), ijk)
    ijk_nbrs_to_index = map(lambda l: _change_to_ravel_index(l, shape),
                            ijk_nbrs)
    # we get a list of idx, values for a particular idx
    # we have got the complete list now, map it to actual index
    actual_map = []
    for i, list_of_coords in enumerate(ijk_nbrs_to_index):
        for coord in list_of_coords:
            actual_map.append((i, coord[0], coord[1]))

    i, j, vals = zip(*actual_map)
    # return sparse.BCOO((jnp.asarray(vals), jnp.c_[i, j]), shape=(size, size))
    return coo_matrix((vals, (i, j)), shape=(size, size))


# get Sobel sparse matrix for X
def get_xSobel_matrix(m, n, p):
    size = m * n * p
    shape = (m, n, p)
    i, j, k = np.unravel_index(np.arange(size), (m, n, p))
    ijk = zip(list(i), list(j), list(k))
    ijk_nbrs = map(lambda x: _get_xsobel_coord(x, shape), ijk)
    ijk_nbrs_to_index = map(lambda l: _change_to_ravel_index(l, shape),
                            ijk_nbrs)
    # we get a list of idx, values for a particular idx
    # we have got the complete list now, map it to actual index
    actual_map = []
    for i, list_of_coords in enumerate(ijk_nbrs_to_index):
        for coord in list_of_coords:
            actual_map.append((i, coord[0], coord[1]))

    i, j, vals = zip(*actual_map)
    # return sparse.BCOO((jnp.asarray(vals), jnp.c_[i, j]), shape=(size, size))
    return coo_matrix((vals, (i, j)), shape=(size, size))
