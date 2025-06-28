import os

os.environ['CUDA_VISIBLE_DEVICES'] = '8'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '1'

# pylint: disable=wrong-import-position

from itertools import product

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.sparse import coo_matrix
from jax.experimental import sparse
from awrjax.utils import COO_to_CSR_info

np.random.seed(0)

class TestSparseUtils:

    @pytest.mark.parametrize(
        'i_size, j_size, threshold',
        product([30, 50, 70], [30, 50, 70], [0.85, 0.90, 0.95]))
    def test_COO_to_CSR_info(self, i_size, j_size, threshold):
        ii, jj = np.where(np.random.uniform(size=(i_size, j_size)) > threshold)
        X_coo = coo_matrix((np.ones_like(ii, dtype=np.float32), (ii, jj)),
                           shape=(i_size, j_size))
        X_coo_jax = sparse.BCOO(
            (jnp.ones_like(ii, jnp.float32), jnp.c_[ii, jj]),
            shape=(i_size, j_size))

        data, indices, indptr = COO_to_CSR_info(X_coo_jax)

        assert jnp.allclose(data, X_coo.tocsr().data)
        assert jnp.allclose(indices, X_coo.tocsr().indices)
        assert jnp.allclose(indptr, X_coo.tocsr().indptr)

    def test_COO_to_CSR_info_specific(self):

        ii = np.arange(1)
        jj = np.arange(1)
        i_size, j_size = 2, 2
        X_coo = coo_matrix((np.ones_like(ii, dtype=np.float32), (ii, jj)),
                           shape=(i_size, j_size))
        X_coo_jax = sparse.BCOO(
            (jnp.ones_like(ii, jnp.float32), jnp.c_[ii, jj]),
            shape=(i_size, j_size))

        data, indices, indptr = COO_to_CSR_info(X_coo_jax)

        assert jnp.allclose(data, X_coo.tocsr().data)
        assert jnp.allclose(indices, X_coo.tocsr().indices)
        assert jnp.allclose(indptr, X_coo.tocsr().indptr)
