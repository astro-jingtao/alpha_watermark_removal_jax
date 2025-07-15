import cv2
import jax.numpy as jnp
import numpy as np
from jax import vmap
from jax.scipy.signal import convolve
from poissonpy.functional import get_np_gradient

from .kernel import gaussian_kernel, grad_kernel, sobel_kernel


def convolve_with_kernel(image, kernel, boundary='reflect', method='auto'):

    npad_i = kernel.shape[0] // 2
    npad_j = kernel.shape[1] // 2

    _pad_width = ((npad_i, npad_i), (npad_j, npad_j))
    if len(image.shape) == 3:
        _pad_width += ((0, 0), )

    image_ext = jnp.pad(image, pad_width=_pad_width, mode=boundary)

    if len(image.shape) == 2:
        image_conv = convolve(image_ext, kernel, mode='same', method=method)
    else:
        image_conv = jnp.zeros_like(image_ext)
        for i in range(image.shape[-1]):
            image_conv = image_conv.at[:, :, i].set(
                convolve(image_ext[:, :, i],
                         kernel,
                         mode='same',
                         method=method))

    return image_conv[npad_i:-npad_i, npad_j:-npad_j]


def gaussian(image, sigma, size, boundary='reflect'):
    return convolve_with_kernel(image,
                                gaussian_kernel(sigma, size),
                                boundary=boundary)


def sobel(image, axis, boundary='reflect'):
    # https://docs.opencv.org/4.11.0/d2/d2c/tutorial_sobel_derivatives.html
    return convolve_with_kernel(image, sobel_kernel(axis), boundary=boundary)


def sobel_cv2(image, axis, norm=False):
    original_dtype = image.dtype
    image = image.astype(np.float32)

    if norm:
        scaler = 1/8
    else:
        scaler = 1

    if axis in (0, 'y', 'i'):
        return cv2.Sobel(image, cv2.CV_32F, 0, 1,
                         ksize=3).astype(original_dtype) * scaler
    elif axis in (1, 'x', 'j'):
        return cv2.Sobel(image, cv2.CV_32F, 1, 0,
                         ksize=3).astype(original_dtype) * scaler
    else:
        raise ValueError('Invalid axis')


def grad(image, axis, boundary='reflect'):
    return convolve_with_kernel(image, grad_kernel(axis), boundary=boundary)


def grad_np(image, axis, mode='center', boundary='reflect'):

    _pad_width = ((1, 1), (1, 1))
    if len(image.shape) == 3:
        batch = True
        _pad_width += ((0, 0), )
    else:
        batch = False

    if mode == 'center':

        image_padded = np.pad(image, pad_width=_pad_width, mode=boundary) # type: ignore

        if axis in (0, 'y', 'i'):
            return (image_padded[2:, :, :] - image_padded[:-2, :, :])[:, 1:-1, :]
        elif axis in (1, 'x', 'j'):
            return (image_padded[:, 2:, :] - image_padded[:, :-2, :])[1:-1, :, :]
        else:
            raise ValueError('Invalid axis')

    elif mode in ['forward', 'backward']:
        forward = True if mode == 'forward' else False
        gx, gy = get_np_gradient(image, forward=forward, batch=batch, padding=True)
        if axis in (0, 'y', 'i'):
            return gy
        elif axis in (1, 'x', 'j'):
            return gx
