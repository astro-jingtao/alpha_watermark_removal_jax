from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax.experimental import sparse
from jax import jit, vmap

from .utils import binarize_img, spdiag, COO_to_CSR_info

jax.config.update("jax_enable_x64", True)


def closed_form_matte(img, Wm, mylambda=100):
    h, w, c = img.shape
    is_diff = binarize_img(jnp.sum(jnp.abs(img - Wm[..., jnp.newaxis]),
                                   axis=-1),
                           threshold=1e-3).astype(jnp.float32)
    Ds = is_diff.ravel()
    consts_vals = Wm * is_diff
    b_s = consts_vals.ravel()
    # print("Computing Matting Laplacian")
    L = compute_laplacian(img)

    sD_s = spdiag(Ds)
    # print("Solving for alpha")
    A = L + mylambda * sD_s
    data, cols, indptr = COO_to_CSR_info(A)
    # return data.astype(jnp.float32), cols, indptr, mylambda * b_s

    x = sparse.linalg.spsolve(data.astype(jnp.float32), cols, indptr,
                              mylambda * b_s)
    return x
    alpha = jnp.clip(x.reshape(h, w), 0, 1)
    return alpha


@partial(jit, static_argnums=(1, ))
def rolling_block(matrix, window_shape=(3, 3)):
    matrix_width = matrix.shape[1]
    matrix_height = matrix.shape[0]

    window_width = window_shape[0]
    window_height = window_shape[1]

    n_w = matrix_width - window_width + 1
    n_h = matrix_height - window_height + 1

    startsx = jnp.arange(n_w)
    startsy = jnp.arange(n_h)
    starts_xy = jnp.dstack(jnp.meshgrid(startsx, startsy)).reshape(
        -1, 2)  # cartesian product => [[x,y], [x,y], ...]

    array_our = vmap(lambda start: jax.lax.dynamic_slice(
        matrix, (start[1], start[0]), (window_height, window_width)))(
            starts_xy)

    return array_our.reshape((n_h, n_w) + window_shape)


def compute_laplacian(img, eps=1e-7, win_rad=1):
    win_size = (win_rad * 2 + 1)**2
    h, w, d = img.shape
    # Number of window centre indices in h, w axes
    c_h, c_w = h - 2 * win_rad, w - 2 * win_rad
    win_diam = win_rad * 2 + 1

    indsM = jnp.arange(h * w).reshape((h, w))
    ravelImg = img.reshape(h * w, d)
    win_inds = rolling_block(indsM, window_shape=(win_diam, win_diam))

    win_inds = win_inds.reshape(c_h, c_w, win_size)
    # should convert to float64 to avoid precision issues
    winI = ravelImg[win_inds].astype(jnp.float64)

    win_mu = jnp.mean(winI, axis=2, keepdims=True)
    # if winI is unit8
    # jnp.einsum('...ji,...jk ->...ik', winI, winI) give incorrect result
    win_var = jnp.einsum('...ji,...jk ->...ik',
                         winI, winI) / win_size - jnp.einsum(
                             '...ji,...jk ->...ik', win_mu, win_mu)

    inv = jnp.linalg.pinv(win_var + (eps / win_size) * jnp.eye(3))

    X = jnp.einsum('...ij,...jk->...ik', winI - win_mu, inv)
    vals = jnp.eye(win_size) - (1 / win_size) * (
        1 + jnp.einsum('...ij,...kj->...ik', X, winI - win_mu))

    nz_indsCol = jnp.tile(win_inds, win_size).ravel()
    nz_indsRow = jnp.repeat(win_inds, win_size).ravel()
    nz_indsVal = vals.ravel()
    L = sparse.BCOO((nz_indsVal, jnp.c_[nz_indsRow, nz_indsCol]),
                    shape=(h * w, h * w))
    return L
