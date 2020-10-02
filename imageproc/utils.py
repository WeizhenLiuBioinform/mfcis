import numpy as np
from skimage.color import rgb2gray
import cv2


def background_remove(img):
    gray = rgb2gray(img)
    if np.max(gray) > 60000:
        mask = gray < 20000
    else:
        mask = img_preprocess(gray)
    mask = mask.astype(int)
    mask = np.dstack([mask, mask, mask])
    img = img * mask
    return img


def background_padding(img, size):
    w, h = size
    dest_size = max(w, h)
    padding_left = (dest_size - w) // 2
    padding_right = dest_size - padding_left - w
    padding_top = (dest_size - h) // 2
    padding_bottom = dest_size - padding_top - h
    img = np.pad(img, (padding_top, padding_bottom), (padding_left, padding_right), 'constant', constant_values=0)
    return img


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


def unit_test():
    pass