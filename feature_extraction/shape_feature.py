import matplotlib.pyplot as plt
import numpy as np
from pershombox import calculate_discrete_NPHT_2d
import warnings
import cv2
import os

warnings.filterwarnings('ignore')
BINARY_THRESHOLD = 0
OBJECT_AREA_THRESHOLD = 100
HOLE_AREA_THRESHOLD = 10000


# 调用pershombox计算形状的PD
def calculate_dgms_of_shape(bin_img, name, save_path, orientation_num=30):
    leaf_dgms = calculate_discrete_NPHT_2d(bin_img, orientation_num)
    dgms = []
    for index, dgm in enumerate(leaf_dgms):
        dgm_arr = np.array(dgm[0])
        dgms.append(dgm_arr)
        # print("dgm shape is: {}".format(dgm_arr.shape))
        file_name = "{}_{}.txt".format(name, index)
        np.savetxt(os.path.join(save_path, file_name), dgm_arr)
    #return dgms


# 填充叶子上的孔洞
def fillHole(im_in):
    im_floodfill = im_in.copy()
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_in.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    # Combine the two images to get the foreground.
    im_out = im_in | im_floodfill_inv
    return im_out


def img_preprocess(gray_img):
    if(np.max(gray_img > 255)):
        maxvalue = 65535
    else:
        maxvalue = np.max(gray_img)
    ret, mask = cv2.threshold(gray_img, 0, maxvalue, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    mask = fillHole(mask)
    # w, h = mask.shape
    # mask = cv2.resize(mask, (int(w*0.5), int(h*0.5)))
    return mask


def compute_shape_pd(mask, name, save_path):
    dgms = calculate_dgms_of_shape(mask, name, save_path)
    #return dgms
