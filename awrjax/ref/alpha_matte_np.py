from __future__ import division

import numpy as np
import scipy.sparse
import scipy
from scipy.sparse.linalg import spsolve
from numpy.lib.stride_tricks import as_strided


def rolling_block(A, block=(3, 3)):
    shape = (A.shape[0] - block[0] + 1, A.shape[1] - block[1] + 1) + block
    strides = (A.strides[0], A.strides[1]) + A.strides
    return as_strided(A, shape=shape, strides=strides)


# Returns sparse matting laplacian
def computeLaplacian(img, eps=10**(-7), win_rad=1):
    win_size = (win_rad * 2 + 1)**2
    h, w, d = img.shape
    # Number of window centre indices in h, w axes
    c_h, c_w = h - 2 * win_rad, w - 2 * win_rad
    win_diam = win_rad * 2 + 1  # radius for window

    indsM = np.arange(h * w).reshape((h, w))
    ravelImg = img.reshape(h * w, d)
    win_inds = rolling_block(indsM, block=(win_diam, win_diam))

    win_inds = win_inds.reshape(c_h, c_w, win_size)
    # original code do not convert to float64
    # it makes win_var wrong, noted later
    winI = ravelImg[win_inds].astype(np.float64)

    win_mu = np.mean(winI, axis=2, keepdims=True)

    # np.einsum('...ji,...jk ->...ik', winI, winI) give incorrect result if winI is unit8
    win_var = np.einsum('...ji,...jk ->...ik',
                        winI, winI) / win_size - np.einsum(
                            '...ji,...jk ->...ik', win_mu, win_mu)

    inv = np.linalg.pinv(win_var + (eps / win_size) * np.eye(3))

    X = np.einsum('...ij,...jk->...ik', winI - win_mu, inv)
    vals = np.eye(win_size) - (1 / win_size) * (
        1 + np.einsum('...ij,...kj->...ik', X, winI - win_mu))

    nz_indsCol = np.tile(win_inds, win_size).ravel()
    nz_indsRow = np.repeat(win_inds, win_size).ravel()
    nz_indsVal = vals.ravel()
    # return nz_indsVal, nz_indsRow, nz_indsCol
    L = scipy.sparse.coo_matrix((nz_indsVal, (nz_indsRow, nz_indsCol)),
                                shape=(h * w, h * w))
    return L


def closed_form_matte(img, prior, prior_confidence):
    h, w, c = img.shape
    # print("Computing Matting Laplacian")
    L = computeLaplacian(img)

    confidence = scipy.sparse.diags(prior_confidence.ravel())
    # print("Solving for alpha")
    x = spsolve(L + confidence, prior_confidence.ravel() * prior.ravel())
    alpha = np.clip(x.reshape(h, w), 0, 1)
    return alpha


def closed_form_matte_prepare(img, prior, prior_confidence):
    L = computeLaplacian(img)
    confidence = scipy.sparse.diags(prior_confidence.ravel())
    return L + confidence, prior_confidence.ravel() * prior.ravel()