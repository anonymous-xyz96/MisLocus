"""Visualization utilities (stubbed for MisLocus fine-tuning).

Original requires cv2, matplotlib, umap, seaborn, colorcet, sklearn — not
installed in the subcell env. These functions are only called by
result_callback.py during validation visualization, which we disable.
"""


def save_feat_nmf(*args, **kwargs):
    pass


def save_overlay_attn(*args, **kwargs):
    pass


def save_recon(*args, **kwargs):
    pass
