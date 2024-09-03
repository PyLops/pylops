import numpy as np
import pytest

# from pylops.signalprocessing import UDCT

from ucurv import udct, ucurvfwd, ucurvinv, bands2vec, vec2bands

eps = 1e-6
shapes = [
    [[256, 256], ],
    [[32, 32, 32], ],
    [[16, 16, 16, 16], ]
]

configurations = [
    [[[3, 3]],
     [[6, 6]],
     [[12, 12]],
     [[12, 12], [24, 24]],
     [[12, 12], [3, 3], [6, 6]],
     [[12, 12], [3, 3], [6, 6], [24, 24]]],
    [[[3, 3, 3]],
     [[6, 6, 6]],
     [[12, 12, 12]],
     [[12, 12, 12], [24, 24, 24]]],
    # [[12, 12, 12], [3, 3, 3], [6, 6, 6]],
    # [[12, 12, 12], [3, 3, 3], [6, 6, 6], [12, 24, 24]],

    [[[3, 3, 3, 3]]],
    #  [[6, 6, 6, 6]],
    #  [[12, 12, 12, 12]],
    #  [[12, 12, 12, 12], [24, 24, 24, 24]],
    #  [[12, 12, 12, 12], [3, 3, 3, 3], [6, 6, 6, 6]],
    #  [[12, 12, 12, 12], [3, 3, 3, 3], [6, 6, 6, 6], [12, 24, 24, 24]],
]

combinations = [
    (shape, config)
    for shape_list, config_list in zip(shapes, configurations)
    for shape in shape_list
    for config in config_list]


@pytest.mark.parametrize("shape, cfg", combinations)
def test_ucurv(shape, cfg):
    data = np.random.rand(*shape)
    tf = udct(shape, cfg)
    band = ucurvfwd(data, tf)
    recon = ucurvinv(band, tf)
    are_close = np.all(np.isclose(data, recon, atol=eps))
    assert are_close


@pytest.mark.parametrize("shape, cfg", combinations)
def test_vectorize(shape, cfg):
    data = np.random.rand(*shape)
    tf = udct(shape, cfg)
    band = ucurvfwd(data, tf)
    flat = bands2vec(band)
    unflat = vec2bands(flat, tf)
    recon = ucurvinv(unflat, tf)
    are_close = np.all(np.isclose(data, recon, atol=eps))
    assert are_close
