from .core import estimate_watermark, poisson_reconstruct, crop_watermark, detect_watermark, box_watermark
from .utils import load_imgs
import jax.numpy as jnp

J = load_imgs(
    '/home/jingtao/software/workspace/myrepos/alpha_watermark_removal_jax/data/awr_test'
)
J = [j[1500:2500, 2000:4000, :] for j in J]

gx, gy, gx_arr, gy_arr = estimate_watermark(J)

# est = poisson_reconstruct(gx, gy, np.zeros(gx.shape)[:,:,0])
cropped_gx, cropped_gy = crop_watermark(gx, gy)

Wm, loss = poisson_reconstruct(cropped_gx, cropped_gy)

# im, start, end = detect_watermark(J[10], cropped_gx, cropped_gy)

i_min, i_max, j_min, j_max = box_watermark(gx, gy)

J_crop = [j[i_min:i_max, j_min:j_max, :] for j in J]


alph_est = estimate_normalized_alpha(J_crop, Wm)