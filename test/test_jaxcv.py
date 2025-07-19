import os

os.environ['CUDA_VISIBLE_DEVICES'] = '8'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '1'

# pylint: disable=wrong-import-position

from itertools import product

import cv2
import jax.numpy as jnp
import numpy as np
import pytest

from awrjax.jaxcv.filter import gaussian, sobel, sobel_cv2

np.random.seed(1997)


class TestFilter:

    @pytest.mark.parametrize("sigma, size", product([1, 2, 3], [3, 5, 7]))
    def test_gaussian(self, sigma, size):
        img = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
        img_jax = jnp.array(img, dtype=jnp.float32)

        img_g_cv2 = cv2.GaussianBlur(img, (size, size), sigma)
        img_g_jax = gaussian(img_jax, sigma, size)

        # print(jnp.min(jnp.abs(img_g_cv2 - img_g_jax)))
        # print(jnp.max(jnp.abs(img_g_cv2 - img_g_jax)))
        assert jnp.all(jnp.abs(img_g_cv2 - img_g_jax) < 2)

    @pytest.mark.parametrize("dummy", range(9))
    def test_sobel(self, dummy):

        img = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
        img_jax = jnp.array(img, dtype=jnp.float32)

        img_sobel_x_cv2_raw = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        img_sobel_y_cv2_raw= cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)


        img_sobel_x_jax = sobel(img_jax, 'x')
        img_sobel_y_jax = sobel(img_jax, 'y')
        assert jnp.allclose(img_sobel_x_cv2_raw, img_sobel_x_jax)
        assert jnp.allclose(img_sobel_y_cv2_raw, img_sobel_y_jax)


        img_sobel_x_cv2 = sobel_cv2(img.astype(np.float32), 'x')
        img_sobel_y_cv2 = sobel_cv2(img.astype(np.float32), 'y')
        assert jnp.allclose(img_sobel_x_cv2_raw, img_sobel_x_cv2)
        assert jnp.allclose(img_sobel_y_cv2_raw, img_sobel_y_cv2)

    
