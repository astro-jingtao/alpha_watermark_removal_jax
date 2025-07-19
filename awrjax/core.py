from functools import partial

import cv2
import jax.numpy as jnp
import numpy as np
from ait.conv_to_matrix import kernel_to_matrix
from jax import Array, jit
from joblib import Parallel, delayed
from poissonpy.functional import get_np_gradient
# from jax.experimental import sparse
from scipy.sparse import coo_matrix, diags
from scipy.sparse import hstack as sp_hstack
from scipy.sparse import vstack as sp_vstack
from scipy.sparse.linalg import spsolve
from tqdm import tqdm

from .jaxcv.filter import convolve_with_kernel, grad, grad_np, sobel, sobel_cv2
from .post_process import alpha_W_correction, clean_inpaint, edge_correction
# jax version can not work, let's use numpy version
from .ref.alpha_matte_np import closed_form_matte
from .utils import COO_spsolve, binarize_img, normalize_img, spdiag

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
    # @jit
    def compute_gradients(J):
        # gradx = [sobel(j, axis='x') for j in J]
        # grady = [sobel(j, axis='y') for j in J]
        # gradx = [grad(j, axis='x') for j in J]
        # grady = [grad(j, axis='y') for j in J]
        gradx = []
        grady = []
        for j in J:
            gx, gy = get_np_gradient(j, forward=True, batch=True, padding=True)
            gradx.append(gx)
            grady.append(gy)
        return jnp.array(gradx), jnp.array(grady)

    gradx_arr, grady_arr = compute_gradients(J)

    # Compute median of grads
    if verbose:
        print("Computing median gradients.")
    gradx_med = jnp.median(gradx_arr, axis=0)
    grady_med = jnp.median(grady_arr, axis=0)

    return gradx_med, grady_med, gradx_arr, grady_arr


def estimate_watermark_laplacian(J):
    """
    Input:
        J: list of images
    Output:
        gradx_med: median of gradients in x-direction
        grady_med: median of gradients in y-direction
        gradx_arr: gradients in x-direction
        grady_arr: gradients in y-direction
    """
    J = [np.asarray(img) for img in J]

    lap = [cv2.Laplacian(j.astype(np.float32), cv2.CV_32F) for j in J]

    lap_med = np.median(lap, axis=0)

    return lap_med


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
                              std_bg_thresh=0.4,
                              n_jobs=1):

    prior, prior_confidence = get_matte_prior(J, Wm, _lambda, threshold,
                                              std_wm_thresh, std_bg_thresh)
    prior = np.asarray(prior)
    prior_confidence = np.asarray(prior_confidence)

    num = len(J)
    alpha_lst = []

    print(f"Estimating normalized alpha using {num} images.")
    # for all images, calculate alpha
    # for idx in tqdm(range(num)):
    #     alpha = closed_form_matte(np.asarray(J[idx]),
    #                               prior=prior,
    #                               prior_confidence=prior_confidence)
    #     alpha_lst.append(alpha)

    alpha_lst = Parallel(n_jobs=n_jobs)(delayed(closed_form_matte)(
        np.asarray(J[idx]), prior=prior, prior_confidence=prior_confidence)
                                        for idx in range(num))

    alpha = np.median(alpha_lst, axis=0)
    return jnp.asarray(alpha)


def get_matte_prior(J, Wm, _lambda, threshold, std_wm_thresh, std_bg_thresh):
    Wm = binarize_img(jnp.mean(normalize_img(Wm, axis=(0, 1)), axis=-1),
                      threshold=threshold)
    std_map = normalize_img(jnp.std(jnp.asarray(J), axis=0).mean(axis=-1))
    is_watermark = (std_map < std_wm_thresh) | (Wm == 1)
    is_background = (std_map > std_bg_thresh) & (Wm == 0)

    prior = np.asarray(is_watermark, dtype=np.float64)
    prior_confidence = np.ones_like(prior) * _lambda
    prior_confidence[~is_watermark & ~is_background] = 0
    return prior, prior_confidence


def estimate_blend_factor(J, alpha_n, Wm, edge_threshold=0.001):
    _std = jnp.std(jnp.asarray(J), axis=0)
    std_I = jnp.median(_std[alpha_n == 0], axis=0)
    C = 1 - jnp.median(_std[alpha_n == 1], axis=0) / std_I
    alpha = alpha_n[..., None] * jnp.array(C)
    est_Ik = jnp.median(jnp.asarray(J) - Wm, axis=0)
    alpha_std = 1 - (_std / std_I)
    alpha_n_gx = grad_operator(np.asarray(alpha_n, dtype=np.float32), axis='x')
    alpha_n_gy = grad_operator(np.asarray(alpha_n, dtype=np.float32), axis='y')
    is_edge = (alpha_n_gx**2 + alpha_n_gy**2) > edge_threshold
    alpha_mix = np.array(alpha.copy())
    alpha_mix[is_edge] = alpha_std[is_edge]
    return C, est_Ik, alpha_mix


# NOTE: grad_np lead to grid like artifacts, should use sobel
# grad_operator = grad_np
grad_operator = partial(sobel_cv2, norm=True)
# grad_operator = partial(grad_np, mode='forward')
# grad_operator = partial(grad_np, mode='center')
# grad_kernel = 'gradf' # it can not work due to numerical instability?
grad_kernel = 'sobel'


def solve_images_jax(J,
                     Wm,
                     alpha,
                     W_init,
                     alpha_max=1,
                     gamma=1,
                     beta=1,
                     lambda_w=0.005,
                     lambda_i=1,
                     lambda_a=0.01,
                     iters=4,
                     decompose_iters=3,
                     Wm_iters=20,
                     alpha_inters=3,
                     n_jobs=4,
                     tol=0.05,
                     Wm_new_tol=1e-3,
                     step_3_update_alpha=False,
                     step_3_update_W=False,
                     step_4=True):
    '''
    gamma: phi_aux, Wk looks like W
    beta: phi_f, alpha W have similar gradient as Wm
    lambda_w: W smoothness in large alpha gradient
    lambda_i: I smoothness in large alpha gradient
    lambda_a: alpha smoothness

    W_init: Initial value of W
    Step 1: Image Watermark decomposition
    '''
    # prepare variables

    if (not step_4) and (not step_3_update_alpha):
        print("Alpha should be updated in step 3 if step 4 is skipped")
        print("Set `step_3_update_alpha = True` automatically")
        step_3_update_alpha = True

    J = np.asarray(J)
    K, m, n, p = J.shape

    sobelx = kernel_to_matrix(m,
                              n,
                              p,
                              kernel_type=f'x{grad_kernel}',
                              normalize=True)
    sobely = kernel_to_matrix(m,
                              n,
                              p,
                              kernel_type=f'y{grad_kernel}',
                              normalize=True)
    Ik = []
    Wk = []

    _Wk = W_init.copy()
    for i in range(K):
        Ik.append(np.clip((J[i] - Wm) / (1 - alpha), 0, 255))
        Wk.append(_Wk)

    # This is for median images
    W = _Wk.copy()

    # Wm_old = Wm.copy()
    alpha_old = alpha.copy()

    Wm_gx = grad_operator(Wm, axis='x')
    Wm_gy = grad_operator(Wm, axis='y')

    # Iterations
    for ii in range(iters):

        print("------------------------------------")
        print(f"Iteration:{ii}")

        # Step 1
        print("Step 1")

        alpha_gx, alpha_gy, cx, cy, alpha_diag, alpha_bar_diag = prepare_alpha_related_parameters(
            alpha)

        results = Parallel(n_jobs=n_jobs)(
            delayed(decompose_watermark_image)(J_i,
                                               Wk_i,
                                               Ik_i,
                                               alpha,
                                               alpha_gx,
                                               alpha_gy,
                                               alpha_diag,
                                               alpha_bar_diag,
                                               Wm,
                                               Wm_gx,
                                               Wm_gy,
                                               W,
                                               sobelx,
                                               sobely,
                                               cx,
                                               cy,
                                               gamma,
                                               beta,
                                               lambda_w,
                                               lambda_i,
                                               m,
                                               n,
                                               p,
                                               decompose_iters,
                                               verbose=False,
                                               tol=tol)
            for J_i, Wk_i, Ik_i in zip(J, Wk, Ik))

        for i, (Wk_i, Ik_i) in enumerate(results):
            Wk[i] = Wk_i
            Ik[i] = Ik_i

        # res = (np.median(Ik_f, axis=0) - EIk)

        # Step 2
        print("Step 2")

        # Parallelize alpha_W_correction and edge_correction using joblib
        def step2_wrapper(J_i, Wk_i, alpha):
            I_corr, res_x = alpha_W_correction(J_i, Wk_i, alpha)
            alpha_scaler = res_x[:3]
            W_scaler = res_x[3:]
            Ik_new = edge_correction(clean_inpaint(I_corr),
                                     alpha).astype(np.float32)
            Wk_new = Wk_i * W_scaler
            return alpha_scaler, Ik_new, Wk_new

        results = Parallel(n_jobs=n_jobs)(
            delayed(step2_wrapper)(J[i], Wk[i], alpha) for i in range(K))

        alpha_scaler_lst = []
        for i, (alpha_scaler, Ik_new, Wk_new) in enumerate(results):
            alpha_scaler_lst.append(alpha_scaler)
            Ik[i] = Ik_new
            Wk[i] = Wk_new

        print(np.median(alpha_scaler_lst, axis=0))

        alpha = alpha * np.median(alpha_scaler_lst, axis=0)
        W = np.median(np.asarray(Wk), axis=0)

        # Step 3
        print("Step 3")
        alpha_step3 = alpha.copy()
        is_step3_converge = False
        for j in range(Wm_iters):
            Wm_new, alpha_new = update_Wm_alpha(J, Ik, Wm, alpha_step3, K, m,
                                                n, p)
            Wm_rdiff = np.linalg.norm(Wm_new - Wm) / np.linalg.norm(Wm)
            alpha_rdiff = np.linalg.norm(
                alpha_new - alpha_step3) / np.linalg.norm(alpha_step3)
            Wm = Wm_new
            alpha_step3 = alpha_new

            if Wm_rdiff < Wm_new_tol:
                is_step3_converge = True
                break

        if is_step3_converge:
            print(f"Wm converge at {j+1}/{Wm_iters}")
        else:
            print(f"Warning: Wm not converge with {Wm_iters} iterations")

        print(f"Wm_rdiff last iteration: {Wm_rdiff}")  # type: ignore
        if step_3_update_alpha:
            alpha = alpha_step3
            print(f"alpha_rdiff last iteration: {alpha_rdiff}")  # type: ignore

        if step_3_update_W:
            W = np.zeros_like(Wm)
            alpha_nonzero = ~(alpha == 0)
            W[alpha_nonzero] = Wm[alpha_nonzero] / alpha[alpha_nonzero]

        if step_4:
            # Step 4
            print("Step 4")

            step3_alpha_old = alpha.copy()
            first_rdiff = None
            if_converge = False

            for j in range(alpha_inters):

                alpha = update_alpha(
                    J=J,
                    Ik=Ik,
                    W=W,
                    Wm_gx=Wm_gx,
                    Wm_gy=Wm_gy,
                    Wm=Wm,
                    alpha=alpha,
                    #  alpha_gx_abs=alpha_gx_abs,
                    #  alpha_gy_abs=alpha_gy_abs,
                    sobelx=sobelx,
                    sobely=sobely,
                    beta=beta,
                    lambda_a=lambda_a,
                    K=K,
                    m=m,
                    n=n,
                    p=p,
                    alpha_max=alpha_max)

                rdiff_alpha = np.linalg.norm(
                    alpha - step3_alpha_old) / np.linalg.norm(alpha)
                # print(rdiff_alpha)
                step3_alpha_old = alpha.copy()

                if first_rdiff is None:
                    first_rdiff = rdiff_alpha
                else:
                    if rdiff_alpha < 0.5 * first_rdiff + 1e-3:
                        if_converge = True
                        print(f"alpha converge at {j+1}/{alpha_inters}")
                        print(f"first_rdiff: {first_rdiff}")
                        print(f"rdiff_alpha: {rdiff_alpha}")
                        break

                # print(f"alpha_inters: {j+1}/{alpha_inters}")

            if not if_converge:
                print("Warning: alpha not converge")
                print(f"first_rdiff: {first_rdiff}")
                print(f"rdiff_alpha: {rdiff_alpha}")  # type: ignore

            print(np.linalg.norm(alpha - alpha_old))
            alpha_old = alpha.copy()

            # Wm = alpha * W
            # print(np.linalg.norm(Wm - Wm_old))
            # Wm_old = Wm.copy()

    return (Wk, Ik, W, alpha)


def just_decompose(J,
                   Wm,
                   alpha,
                   W_init,
                   gamma=1,
                   beta=1,
                   lambda_w=0.005,
                   lambda_i=1,
                   decompose_iters=3,
                   n_jobs=4,
                   verbose=False,
                   tol=0.05,
                   I_ref=None,
                   r_gamma_I=1,
                   I_ref_weight='alpha_grad'):
    '''
    gamma: phi_aux, Wk looks like W
    beta: phi_f, alpha W have similar gradient as Wm
    lambda_w: W smoothness in large alpha gradient
    lambda_i: I smoothness in large alpha gradient

    W_init: Initial value of W
    '''

    # prepare variables
    J = np.asarray(J)
    K, m, n, p = J.shape

    sobelx = kernel_to_matrix(m,
                              n,
                              p,
                              kernel_type=f'x{grad_kernel}',
                              normalize=True)
    sobely = kernel_to_matrix(m,
                              n,
                              p,
                              kernel_type=f'y{grad_kernel}',
                              normalize=True)
    Ik = []
    Wk = []

    _Wk = W_init.copy()
    for i in range(K):
        Ik.append(np.clip((J[i] - Wm) / (1 - alpha), 0, 255))
        Wk.append(_Wk)

    # This is for median images
    W = _Wk.copy()

    Wm_gx = grad_operator(Wm, axis='x')
    Wm_gy = grad_operator(Wm, axis='y')

    alpha_gx, alpha_gy, cx, cy, alpha_diag, alpha_bar_diag = prepare_alpha_related_parameters(
        alpha)

    results = Parallel(n_jobs=n_jobs)(
        delayed(decompose_watermark_image)(J_i,
                                           Wk_i,
                                           Ik_i,
                                           alpha,
                                           alpha_gx,
                                           alpha_gy,
                                           alpha_diag,
                                           alpha_bar_diag,
                                           Wm,
                                           Wm_gx,
                                           Wm_gy,
                                           W,
                                           sobelx,
                                           sobely,
                                           cx,
                                           cy,
                                           gamma,
                                           beta,
                                           lambda_w,
                                           lambda_i,
                                           m,
                                           n,
                                           p,
                                           decompose_iters,
                                           verbose=verbose,
                                           tol=tol,
                                           I_ref=I_ref,
                                           r_gamma_I=r_gamma_I,
                                           I_ref_weight=I_ref_weight)
        for J_i, Wk_i, Ik_i in zip(J, Wk, Ik))

    for i, (Wk_i, Ik_i) in enumerate(results):
        Wk[i] = Wk_i
        Ik[i] = Ik_i

    return Wk, Ik


def prepare_alpha_related_parameters(alpha):
    alpha_gx = grad_operator(alpha, axis='x')
    alpha_gy = grad_operator(alpha, axis='y')

    cx = diags(np.abs(alpha_gx).flatten())
    cy = diags(np.abs(alpha_gy).flatten())

    alpha_diag = diags(alpha.flatten())
    alpha_bar_diag = diags((1 - alpha).flatten())

    return alpha_gx, alpha_gy, cx, cy, alpha_diag, alpha_bar_diag


def update_alpha(
        J,
        Ik,
        W,
        Wm_gx,
        Wm_gy,
        Wm,
        alpha,
        #  alpha_gx_abs,
        #  alpha_gy_abs,
        sobelx,
        sobely,
        beta,
        lambda_a,
        K,
        m,
        n,
        p,
        alpha_max=1):

    alpha_gx_abs = np.abs(grad_operator(alpha, axis='x'))
    alpha_gy_abs = np.abs(grad_operator(alpha, axis='y'))

    alphaWk = alpha * W
    alphaWk_gx = grad_operator(alphaWk, axis='x')
    alphaWk_gy = grad_operator(alphaWk, axis='y')

    W_gx = grad_operator(W, axis='x')
    W_gy = grad_operator(W, axis='y')

    W_diag = diags(W.flatten())
    W_gx_diag = diags(W_gx.flatten())
    W_gy_diag = diags(W_gy.flatten())

    phi_alpha = diags(
        func_phi_deriv(alpha_gx_abs**2 + alpha_gy_abs**2).flatten())
    phi_f = diags(
        func_phi_deriv(
            ((Wm_gx - alphaWk_gx)**2 + (Wm_gy - alphaWk_gy)**2).flatten()))

    L_alpha = sobelx @ (phi_alpha @ (sobelx)) + sobely @ (phi_alpha @ (sobely))

    K_fx = sobelx @ phi_f @ W_diag
    K_fy = sobely @ phi_f @ W_diag
    A_f = K_fx @ (W_gx_diag + W_diag @ sobelx) + K_fy @ (W_gy_diag +
                                                         W_diag @ sobely)

    A1 = -lambda_a * L_alpha - beta * A_f
    b1 = -beta * (K_fx @ sobelx + K_fy @ sobely) @ (Wm.flatten())

    for i in range(K):

        # paper use W, while original code use Wk[i]
        A_k = func_phi_deriv(
            (alpha * W + (1 - alpha) * Ik[i] - J[i])**2) * (W - Ik[i])

        phi_kA = diags((A_k * (W - Ik[i])).reshape(-1))
        phi_kB = (A_k * (J[i] - Ik[i])).reshape(-1)

        A1 += phi_kA
        b1 += phi_kB

    alpha = spsolve(A1, b1).reshape(m, n, p)
    alpha = np.clip(alpha, 0, alpha_max)
    # alpha = np.clip(np.stack([np.mean(alpha, axis=-1)] * 3, axis=-1), 0,
    #                 alpha_max)

    return alpha


def update_Wm_alpha(J, Ik, Wm, alpha, K, m, n, p):

    size = m * n * p

    A_ul = diags((np.zeros((size))).flatten())
    A_ur = diags((np.zeros((size))).flatten())
    b_u = np.zeros((size))
    A_ll = diags((np.zeros((size))).flatten())
    A_lr = diags((np.zeros((size))).flatten())
    b_l = np.zeros((size))

    for i in range(K):

        phi_data = func_phi_deriv((J[i] - Wm - (1 - alpha) * Ik[i])**2)

        A_ul += diags((phi_data * Ik[i]).flatten())
        A_ur += diags(-(phi_data * Ik[i] * Ik[i]).flatten())
        b_u += (phi_data * Ik[i] * (J[i] - Ik[i])).flatten()

        A_ll += diags((phi_data).flatten())
        A_lr += diags(-(phi_data * Ik[i]).flatten())
        b_l += (phi_data * (J[i] - Ik[i])).flatten()

    A = sp_vstack([sp_hstack([A_ul, A_ur]), sp_hstack([A_ll, A_lr])]).tocsr()
    b = np.hstack([b_u, b_l])

    x = spsolve(A, b)

    Wm_new = np.clip(x[:size].reshape(m, n, p), 0, 255)  # type: ignore
    alpha_new = np.clip(x[size:].reshape(m, n, p), 0, 255)  # type: ignore

    return Wm_new, alpha_new


def func_phi(X, epsilon=1e-3):
    return np.sqrt(X + epsilon**2)


def func_phi_deriv(X, epsilon=1e-3):
    return 0.5 / func_phi(X, epsilon)


def decompose_watermark_image(J_i,
                              Wk_i,
                              Ik_i,
                              alpha,
                              alpha_gx,
                              alpha_gy,
                              alpha_diag,
                              alpha_bar_diag,
                              Wm,
                              Wm_gx,
                              Wm_gy,
                              W,
                              sobelx,
                              sobely,
                              cx,
                              cy,
                              gamma,
                              beta,
                              lambda_w,
                              lambda_i,
                              m,
                              n,
                              p,
                              decompose_iters,
                              tol=0.05,
                              verbose=False,
                              I_ref=None,
                              r_gamma_I=1,
                              I_ref_weight='alpha_grad'):

    Wk_old = Wk_i.copy()
    Ik_old = Ik_i.copy()

    for j in range(decompose_iters):

        Wk_i, Ik_i = _decompose_watermark_image_single(
            J_i=J_i,
            Wk_i=Wk_i,
            Ik_i=Ik_i,
            alpha=alpha,
            alpha_gx=alpha_gx,
            alpha_gy=alpha_gy,
            alpha_diag=alpha_diag,
            alpha_bar_diag=alpha_bar_diag,
            Wm=Wm,
            Wm_gx=Wm_gx,
            Wm_gy=Wm_gy,
            W=W,
            sobelx=sobelx,
            sobely=sobely,
            cx=cx,
            cy=cy,
            gamma=gamma,
            beta=beta,
            lambda_w=lambda_w,
            lambda_i=lambda_i,
            m=m,
            n=n,
            p=p,
            I_ref=I_ref,
            r_gamma_I=r_gamma_I,
            I_ref_weight=I_ref_weight)

        rdiff_Wk = np.linalg.norm(Wk_i - Wk_old) / np.linalg.norm(Wk_old)
        rdiff_Ik = np.linalg.norm(Ik_i - Ik_old) / np.linalg.norm(Ik_old)

        if (rdiff_Wk < tol) and (rdiff_Ik < tol):
            if verbose:
                print(f"Converged in {j+1} iterations")
                print(f"Final rdiff_Wk: {rdiff_Wk}")
                print(f"Final rdiff_Ik: {rdiff_Ik}")
            return Wk_i, Ik_i

        Wk_old = Wk_i.copy()
        Ik_old = Ik_i.copy()

        if verbose:
            print(f"{j+1}/{decompose_iters}")
            print(f"rdiff_Wk: {rdiff_Wk}")
            print(f"rdiff_Ik: {rdiff_Ik}")

    if verbose:
        print("Not converged")
        print(f"Final rdiff_Wk: {rdiff_Wk}")  # type: ignore
        print(f"Final rdiff_Ik: {rdiff_Ik}")  # type: ignore

    return Wk_i, Ik_i


def _decompose_watermark_image_single(J_i,
                                      Wk_i,
                                      Ik_i,
                                      alpha,
                                      alpha_gx,
                                      alpha_gy,
                                      alpha_diag,
                                      alpha_bar_diag,
                                      Wm,
                                      Wm_gx,
                                      Wm_gy,
                                      W,
                                      sobelx,
                                      sobely,
                                      cx,
                                      cy,
                                      gamma,
                                      beta,
                                      lambda_w,
                                      lambda_i,
                                      m,
                                      n,
                                      p,
                                      I_ref=None,
                                      r_gamma_I=1,
                                      I_ref_weight='alpha_grad'):

    size = m * n * p

    Wkx = grad_operator(Wk_i, axis='x')
    Wky = grad_operator(Wk_i, axis='y')

    Ikx = grad_operator(Ik_i, axis='x')
    Iky = grad_operator(Ik_i, axis='y')

    alpha_gx_abs = np.abs(alpha_gx)
    alpha_gy_abs = np.abs(alpha_gy)

    alpha_gx_diag = diags(alpha_gx.flatten())
    alpha_gy_diag = diags(alpha_gy.flatten())

    alphaWk = alpha * Wk_i
    alphaWk_gx = grad_operator(alphaWk, axis='x')
    alphaWk_gy = grad_operator(alphaWk, axis='y')

    phi_data = diags(
        func_phi_deriv(
            np.square(alpha * Wk_i + (1 - alpha) * Ik_i - J_i).flatten()))
    phi_f = diags(
        func_phi_deriv(
            ((Wm_gx - alphaWk_gx)**2 + (Wm_gy - alphaWk_gy)**2).flatten()))
    phi_aux = diags(func_phi_deriv(np.square(Wk_i - W).flatten()))
    phi_rI = diags(
        func_phi_deriv(alpha_gx_abs * (Ikx**2) + alpha_gy_abs *
                       (Iky**2)).flatten())
    phi_rW = diags(
        func_phi_deriv(alpha_gx_abs * (Wkx**2) + alpha_gy_abs *
                       (Wky**2)).flatten())

    L_i = sobelx @ (cx * phi_rI) @ (sobelx) + sobely @ (cy * phi_rI) @ (sobely)
    L_w = sobelx @ (cx * phi_rW) @ (sobelx) + sobely @ (cy * phi_rW) @ (sobely)
    K_fx = sobelx @ phi_f @ alpha_diag
    K_fy = sobely @ phi_f @ alpha_diag
    A_f = K_fx @ (alpha_gx_diag + alpha_diag @ sobelx) + K_fy @ (
        alpha_gy_diag + alpha_diag @ sobely)

    A_ul = (alpha_diag**
            2) * phi_data + gamma * phi_aux - lambda_w * L_w - beta * A_f

    if I_ref is not None:

        if I_ref_weight == 'alpha_grad':
            scaler = np.sqrt(alpha_gx_abs**2 + alpha_gy_abs**2)
        elif I_ref_weight == 'alpha_non_zero':
            scaler = (alpha > 0).astype(np.float32)
        elif I_ref_weight == 'same':
            scaler = np.ones_like(alpha_gx_abs)
        else:
            raise ValueError(f"Unknown I_ref_weight: {I_ref_weight}")
        phi_I = diags(
            func_phi_deriv((np.square(Ik_i - I_ref) * scaler).flatten()))

        K_scaler = diags(scaler.flatten()) @ phi_I

        A_lr = (alpha_bar_diag**
                2) * phi_data - lambda_i * L_i + gamma * r_gamma_I * K_scaler

        bI = alpha_bar_diag @ (phi_data) @ (
            J_i.flatten()) + gamma * r_gamma_I * K_scaler @ (I_ref.flatten())
    else:
        A_lr = (alpha_bar_diag**2) * phi_data - lambda_i * L_i

        bI = alpha_bar_diag @ (phi_data) @ (J_i.flatten())

    A = sp_vstack([sp_hstack([A_ul, alpha_diag*alpha_bar_diag*phi_data]), \
                    sp_hstack([alpha_diag*alpha_bar_diag*phi_data, A_lr])]).tocsr()

    bW = alpha_diag @ (phi_data) @ (J_i.flatten()) + gamma * phi_aux @ (
        W.flatten()) - beta * (K_fx @ sobelx + K_fy @ sobely) @ (Wm.flatten())

    b = np.hstack([bW, bI])
    # return A, b
    x = spsolve(A, b)

    Wk_i_new = np.clip(x[:size].reshape(m, n, p), 0, 255)  # type: ignore
    Ik_i_new = np.clip(x[size:].reshape(m, n, p), 0, 255)  # type: ignore

    # Wk_i_new = x[:size].reshape(m, n, p)  # type: ignore
    # Ik_i_new = x[size:].reshape(m, n, p)  # type: ignore

    return Wk_i_new, Ik_i_new
