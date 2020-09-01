import numpy as np
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from feature_extraction.texture_feature import compute_texture_pd
from feature_extraction.venation_feature import compute_venation_pd
from feature_extraction.shape_feature import compute_shape_pd
import os
import homcloud.interface as hc
from config.model_configuration import view_combination, pht_threshold_shape, pht_threshold_texture, pht_threshold_vein
import cv2
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.transform import rescale

# pd transform
class PHT:
    def __init__(self, v):
        self.b_1 = np.reshape(np.array([1, 1]) / np.sqrt(2), [1, 2])
        self.b_2 = np.reshape(np.array([-1, 1]) / np.sqrt(2), [1, 2])
        self.v = v

    def __call__(self, pds):
        x = pds * np.repeat(self.b_1, pds.shape[0], axis=0)
        x = np.sum(x, axis=1)
        y = pds * np.repeat(self.b_2, pds.shape[0], axis=0)
        y = np.sum(y, axis=1)
        i = [y <= self.v]
        y[i] = np.log(y[i] / self.v) + self.v
        ret = np.stack([x, y], axis=1)
        return ret


pht = {}
pht['shape'] = PHT(pht_threshold_shape)
pht['texture'] = PHT(pht_threshold_texture)
pht['venation'] = PHT(pht_threshold_vein)


def get_persistence(pd, N):
    persistence = pd[:, 1] - pd[:, 0]
    index = np.argsort(persistence)[::-1]
    if len(index) > N:
        return index[0:N]
    else:
        temp = np.repeat(index[-1], N-len(index))
        index = np.concatenate([index, temp])
        return index


def compute_pds(imgs, isVenation=False, remove_background=False):
    texture_pd = None
    vein_pd = None
    shape_pd = None
    for key in imgs.keys():
        gray = rgb2gray(imgs[key])
        img2 = cv2.cvtColor(imgs[key], cv2.COLOR_BGR2HSV)
        if remove_background:
            S = img2[:, :, 1]
            re, m = cv2.threshold(S, 10, 120, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            mask = S < re
            mask = remove_small_holes(mask, 1000)
            mask = remove_small_objects(mask, 1000)
            mask = ~mask
        else:
            mask = gray > 0
            mask = remove_small_holes(mask, 1000)
            mask = remove_small_objects(mask, 1000)

        gray = mask.astype(int) * gray
        # gray = rescale(gray, 0.5)
        # mask = rescale(mask, 0.5)
        texture_pd_path = ""
        texture_pd = compute_texture_pd(gray, save_path=texture_pd_path)
        if isVenation:
            vein_pd_path = ""
            dt, vein_pd = compute_venation_pd(gray, save_path=vein_pd_path)
        shape_pd_path = ""
        shape_pd = compute_shape_pd(mask=mask, key=key, save_path=shape_pd_path, type="shape")
    return texture_pd, shape_pd, vein_pd