import numpy as np


def match_mean_std(source, j, mask):
    j_source = source.reshape(j.shape)
    j_source_mean = np.mean(j_source[mask])
    j_source_std = np.std(j_source[mask])
    j_mean = np.mean(j[~mask])
    j_std = np.std(j[~mask])
    j_source_adjusted = j.copy()
    j_source_adjusted[mask] = (j_source[mask] -
                               j_source_mean) / j_source_std * j_std + j_mean
    return j_source_adjusted
