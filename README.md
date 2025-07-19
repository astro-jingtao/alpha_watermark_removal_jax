
# Alpha Watermark Removal

This repository provides tools for alpha watermark removal, based on the algorithm presented in [Dekel et al. 2017](https://openaccess.thecvf.com/content_cvpr_2017/papers/Dekel_On_the_Effectiveness_CVPR_2017_paper.pdf) and the Python implementation by [rohitrango](https://github.com/rohitrango/automatic-watermark-detection).

This implementation introduces modifications to the original algorithm and provides additional algorithms for watermark removal.  Key features and differences from the original implementation include:

- **Parallelization:** Utilizes `joblib` for parallelization of time-consuming operations, significantly improving performance.
- **Bug Fixes:** Addresses bugs found in both the original implementation and the original paper.
- **Improved Blend Factor Estimation:**  Implements a novel approach to estimate the blend factor using the standard deviation of multiple watermarked images (`estimate_blend_factor()` in `core.py`).
- **Post-Processing:** Includes post-processing techniques (`post_process.py`) to minimize watermark residuals and enhance the quality of the resulting image.
- **Simultaneous Alpha Matte and Watermark Update:** Updates the alpha matted watermark `Wm` and `alpha` concurrently (`update_Wm_alpha()` in `core.py`) for improved accuracy.


This project relies on two other GitHub projects:
- [Astro Image Toolkits](https://github.com/astro-jingtao/astro_image_toolkits)
- [Modified `poissonpy`](https://github.com/astro-jingtao/poissonpy)

## Current State of the Project

This project is currently under development. While the code has passed testing in limited cases, further refinement and optimization are ongoing.  The codebase may undergo significant changes.



## Plan

- Provide an example notebook.
- Further optimization.


## Disclaimer

I do not encourage or endorse piracy by making this project public. This algorithm can be easily protected by using opaque watermarks. The code is free for academic/research purposes.


