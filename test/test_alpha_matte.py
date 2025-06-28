import os

os.environ['CUDA_VISIBLE_DEVICES'] = '8'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '1'

# pylint: disable=wrong-import-position

from itertools import product

import jax.numpy as jnp
import numpy as np
import pytest

from awrjax.alpha_matte import compute_laplacian, rolling_block
from awrjax.ref.alpha_matte_np import computeLaplacian as compute_laplacian_np
from awrjax.ref.alpha_matte_np import rolling_block as rolling_block_np

np.random.seed(1997)


class TestRollingBlock:

    @pytest.mark.parametrize('i_size, j_size',
                             product([3, 5, 7, 9], [3, 5, 7, 9]))
    def test_rolling_block(self, i_size, j_size):
        X = np.arange(i_size * j_size).reshape(i_size, j_size)
        X_roll_np = rolling_block_np(X)
        X_roll = rolling_block(X)
        assert jnp.allclose(X_roll, X_roll_np)


class TestLaplacian:

    @pytest.mark.parametrize('i_size, j_size',
                             product([5, 7, 9], [5, 7, 9]))
    def test_rolling_block(self, i_size, j_size):
        X = np.random.randint(0, 256, size=(i_size, j_size, 3))
        L_np = compute_laplacian_np(X)
        L = compute_laplacian(X)
        assert jnp.allclose(L.todense(), L_np.todense(), atol=1e-5)