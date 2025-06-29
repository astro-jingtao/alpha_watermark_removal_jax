import os

from jax.experimental import sparse
import jax.numpy as jnp
from ait.utils import read_img


def normalize_img(img, axis=None):
    return (img - jnp.min(img, axis=axis)) / (jnp.max(img, axis=axis) -
                                              jnp.min(img, axis=axis))


def binarize_img(img, threshold=0.5):
    return jnp.where(img > threshold, 1, 0)


def load_imgs(folder):
    '''
    load all images in a folder and return a list of numpy arrays
    assuming all files are images
    '''

    all_imgs = []
    for file in os.listdir(folder):
        img = read_img(os.path.join(folder, file))
        all_imgs.append(img)
    return all_imgs


def spdiag(x):
    '''
    create a sparse diagonal matrix from a vector
    '''
    n = len(x)
    return sparse.BCOO((x, jnp.c_[jnp.arange(n), jnp.arange(n)]), shape=(n, n))


def COO_spsolve(A, b):
    data, cols, indptr = COO_to_CSR_info(A)
    return sparse.linalg.spsolve(data, cols, indptr, b)


def COO_to_CSR_info(X):
    # Sort COO data by row and then column
    data = X.data
    rows, cols = X.indices.T
    nrows = X.shape[0]

    sorted_args = jnp.lexsort((cols, rows))
    sorted_rows = rows[sorted_args]
    sorted_cols = cols[sorted_args]
    sorted_data = data[sorted_args]

    # Count non-zero elements per row
    # should give length=nrows, or we will lose trailing all zero rows
    row_counts = jnp.bincount(sorted_rows, length=nrows)

    # Create indptr by concatenating 0 and the cumulative sum of row counts
    indptr = jnp.concatenate(
        [jnp.array([0], dtype=row_counts.dtype),
         jnp.cumsum(row_counts)])

    return sorted_data, sorted_cols, indptr
