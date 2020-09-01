import homcloud.interface as hc
from skimage.morphology import remove_small_holes
from skimage.filters import threshold_sauvola
import matplotlib.pyplot as plt
import os
from feature_extraction.texture_feature import dump_pd_as_txt

BINARY_SMALL_HOLES_SIZE = 100


# compute pd of binary vein image
def compute_binary_pd(venation, name, save_path):
    dt = hc.distance_transform(venation, signed=True, metric='euclidean')
    mask = remove_small_holes(venation, BINARY_SMALL_HOLES_SIZE)
    dt = mask.astype(int) * dt
    pd = hc.PDList.from_bitmap_levelset(dt, mode='sublevel', type='bitmap')
    dest_pd0_path = os.path.join(save_path, name, '_vein_pd0.txt')
    dest_pd1_path = os.path.join(save_path, name, '_vein_pd1.txt')
    dump_pd_as_txt(pd, dest_pd0_path, dest_pd1_path)
    return dt, pd


# extracting venation from leaf image by adaptive threshold binarization algorithm
def get_leaf_venation(gray_img):
    # using the sauvola algorithm
    threshold = threshold_sauvola(image=gray_img, window_size=11, k=0.04)
    venation_img = gray_img > threshold
    plt.imshow(venation_img)
    plt.show()
    return venation_img


def compute_venation_pd(gray_img, name, save_path):
    venation_img = get_leaf_venation(gray_img)
    dt, pd = compute_binary_pd(venation_img, name, save_path)
    return dt, pd

