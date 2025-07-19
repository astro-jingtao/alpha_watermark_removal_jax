# pylint: disable=no-member
from functools import partial

import cv2
import numpy as np
from scipy.optimize import minimize

from .jaxcv.filter import sobel_cv2
from .utils import normalize_img_np


def clean_median(I, ksize=5, threshold=5):
    """
    Clean the image using median filter and thresholding.
    
    params:
    I: input image, [0, 255]
    ksize: kernel size of median filter
    threshold: threshold for pixel difference between median and original image
    
    """
    I_median = cv2.medianBlur(I.astype(np.uint8), ksize)
    is_noisy = (np.abs(I_median - I) > threshold).any(axis=-1)
    I_cleaned = I.copy()
    I_cleaned[is_noisy] = I_median[is_noisy]
    return I_cleaned


def clean_inpaint(I,
                  ksize=5,
                  threshold=5,
                  inpaint_radius=3,
                  inpaint_method='NS'):
    """
    Inpaint the image, use median filter and thresholding to construct the mask.
    
    params:
    I: input image, [0, 255]
    ksize: kernel size of median filter
    threshold: threshold for pixel difference between median and original image
    inpaint_radius: radius of inpainting
    inpaint_method: method for inpainting, 'NS' or 'TELEA'
    
    """

    inpaint_flag = get_inpaint_flag(inpaint_method)

    I = np.clip(I, 0, 255)

    I_median = cv2.medianBlur(I.astype(np.uint8), ksize)
    is_noisy = (np.abs(I_median - I) > threshold).any(axis=-1)
    is_noisy = is_noisy.astype(np.uint8) * 255
    I_cleaned = cv2.inpaint(I.astype(np.uint8), is_noisy, inpaint_radius,
                            inpaint_flag)
    return I_cleaned


def expand_mask(mask, k=15, n=1):
    """把二值 mask 往外扩大 n 次，核大小 k"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask, kernel, iterations=n)


def shrink_mask(mask, k=3, n=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.erode(mask, kernel, iterations=n)


def direct_remove(j, W, alpha):
    return (j - W * alpha) / (1 - alpha)


def alpha_W_correction(J,
                       W,
                       alpha,
                       verbose=False,
                       Wm_threshold=1,
                       mask_expand_k=3,
                       median_blur_k=15,
                       inpaint_radius=3,
                       inpaint_method='NS',
                       maxiter=1000,
                       clean_kwargs=None):

    if clean_kwargs is None:
        clean_kwargs = {}

    inpaint_flag = get_inpaint_flag(inpaint_method)

    def loss(x, I_ref, j, W, alpha, mask):
        alpha_scaler = x[:3]
        W_scaler = x[3:]
        Ik_this = direct_remove(j, W * W_scaler, alpha * alpha_scaler)

        Ik_this_blur = cv2.medianBlur(clean_inpaint(Ik_this, **clean_kwargs),
                                      median_blur_k)

        return np.square(Ik_this_blur[mask == 255] - I_ref[mask == 255]).mean()

    mask = ((W * alpha) > Wm_threshold).any(axis=-1).astype(np.uint8) * 255
    # expand_mask to avoid the edge effect
    I_ref = cv2.inpaint(clean_inpaint(J, **clean_kwargs),
                        expand_mask(mask, k=mask_expand_k), inpaint_radius,
                        inpaint_flag)
    I_ref = cv2.medianBlur(I_ref, median_blur_k)

    res = minimize(loss,
                   np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
                   args=(I_ref, J, W, alpha, mask),
                   method='Nelder-Mead',
                   options={
                       'maxiter': maxiter,
                       'disp': verbose
                   })

    if verbose:
        print(res.x)
        print(
            f'No correction loss: {loss(np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), I_ref, J, W, alpha, mask)}'
        )
        print(f'Best correction loss: {loss(res.x, I_ref, J, W, alpha, mask)}')

    alpha_scaler = res.x[:3]
    W_scaler = res.x[3:]

    Ik_this = direct_remove(J, W * W_scaler, alpha * alpha_scaler)

    return Ik_this, res.x


def get_inpaint_flag(inpaint_method):
    inpaint_flag = {
        'NS': cv2.INPAINT_NS,
        'TELEA': cv2.INPAINT_TELEA
    }[inpaint_method]

    return inpaint_flag


grad_operator = partial(sobel_cv2, norm=True)


def edge_correction(I,
                    alpha,
                    grad_threshold=0.1,
                    expand_k=0,
                    inpaint_radius=3,
                    inpaint_method='NS'):

    inpaint_flag = get_inpaint_flag(inpaint_method)

    grad_mask = (normalize_img_np(
        grad_operator(alpha, axis='x')**2 + grad_operator(alpha, axis='y')**2)
                 > grad_threshold).any(axis=-1)

    grad_mask = np.asarray(grad_mask).astype(np.uint8) * 255

    grad_mask_exp = expand_mask(grad_mask,
                                k=expand_k) if expand_k > 0 else grad_mask

    I_ecorr = cv2.inpaint(clean_inpaint(I),
                          grad_mask_exp,
                          inpaint_radius,
                          flags=inpaint_flag)

    return I_ecorr
