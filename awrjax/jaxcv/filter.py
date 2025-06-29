import jax.numpy as jnp
from jax.scipy.signal import convolve
from jax import vmap
from .kernel import gaussian_kernel, sobel_kernel, grad_kernel


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


def grad(image, axis, boundary='reflect'):
    return convolve_with_kernel(image, grad_kernel(axis), boundary=boundary)


# def canny(img, low_thresh=50, high_thresh=100, sigma=1.0):
#     # https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
#     img = rgb2gray(img)
#     img = gaussian(img, sigma, 5)

#     magnitude, direction = gradient_magnitude_direction(img)

#     print('non-maximum suppression')
#     nms = non_maximum_suppression(magnitude, direction)

#     print('double threshold')
#     strong, weak = double_threshold(nms, low_thresh, high_thresh)

#     print('edge tracking by hysteresis')
#     edges = edge_tracking_by_hysteresis(strong, weak)
#     return edges.astype(jnp.float32)

# def rgb2gray(img):
#     return jnp.dot(img[..., :3], jnp.array([0.2989, 0.5870, 0.1140]))

# def gradient_magnitude_direction(img):
#     Gx = sobel(img, axis='x')
#     Gy = sobel(img, axis='y')
#     magnitude = jnp.sqrt(jnp.square(Gx) + jnp.square(Gy))
#     direction = jnp.arctan2(Gy, Gx)
#     return magnitude, direction

# def non_maximum_suppression(magnitude, direction):
#     H, W = magnitude.shape
#     direction = jnp.rad2deg(direction) % 180

#     def suppress_pixel(i, j):
#         angle = direction[i, j]
#         m = magnitude[i, j]

#         cond1 = ((0 <= angle) & (angle < 22.5)) | ((157.5 <= angle) &
#                                                    (angle < 180))

#         before, after = jnp.where(
#             cond1,
#             x=jnp.array((magnitude[i, j - 1], magnitude[i, j + 1])),
#             y=jnp.where(
#                 (22.5 <= angle) & (angle < 67.5),
#                 jnp.array((magnitude[i - 1, j + 1], magnitude[i + 1, j - 1])),
#                 jnp.where(
#                     (67.5 <= angle) & (angle < 112.5),
#                     jnp.array((magnitude[i - 1, j], magnitude[i + 1, j])),
#                     jnp.array(
#                         (magnitude[i - 1,
#                                    j - 1], magnitude[i + 1,
#                                                      j + 1]))  # else condition
#                 )))

#         return jnp.where((m >= before) & (m >= after), m, 0.0)

#     # Vectorized over image except borders
#     vmapped_suppress = vmap(vmap(suppress_pixel, in_axes=(0, None)),
#                             in_axes=(None, 0))

#     result = vmapped_suppress(jnp.arange(1, H - 1), jnp.arange(1, W - 1))

#     return jnp.pad(result, pad_width=1, mode='constant', constant_values=0)

# def double_threshold(img, low_thresh, high_thresh):
#     strong = img > high_thresh
#     weak = (img >= low_thresh) & ~strong
#     return strong, weak

# def edge_tracking_by_hysteresis(strong, weak):
#     H, W = strong.shape
#     edges = strong.copy()

#     def grow(i, j):
#         if not weak[i, j]:
#             return False
#         for di in [-1, 0, 1]:
#             for dj in [-1, 0, 1]:
#                 if di == 0 and dj == 0:
#                     continue
#                 if edges[i + di, j + dj]:
#                     return True
#         return False

#     changed = True
#     while changed:
#         changed = False
#         new_edges = edges.copy()
#         for i in range(1, H - 1):
#             for j in range(1, W - 1):
#                 if weak[i, j] and grow(i, j):
#                     new_edges = new_edges.at[i, j].set(True)
#                     changed = True
#         edges = new_edges

#     return edges
