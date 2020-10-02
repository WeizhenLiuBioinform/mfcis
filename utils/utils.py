import numpy as np
import os
# from tqdm import *
import re
from skimage import io as skio
from skimage.transform import resize
from config.model_configuration import configs, view_combination


def data_loader_for_combined_model(file_list, dataset, config, isVenation):
    shape_x = []
    texture_x = []
    img_x = []
    vein_x = []
    y = []
    regx_str = config['regx_str']
    regx = re.compile(regx_str)
    for path in file_list:
        path = path[:-1]
        strs = str.split(path, '/')
        f_name = regx.findall(strs[-1])[0]
        d = strs[-2][2:]
        if dataset == 'soybean':
            period = strs[-3]
            shape_parent_path = os.path.join(config['shape_data_path'], d, period)
            texture_parent_path = os.path.join(config['texture_data_path'], d, period)
            vein_parent_path = os.path.join(config['vein_data_path'], d, period)
        else:
            shape_parent_path = os.path.join(config['shape_data_path'], d)
            texture_parent_path = os.path.join(config['texture_data_path'], d)
            if isVenation:
                vein_parent_path = os.path.join(config['vein_data_path'], d)

        shape_multiview_x = []
        texture_multiview_x = []
        if isVenation:
            vein_multiview_x = []
        for i in range(config['shape_views']):
            channel_1 = np.loadtxt(
                os.path.join(shape_parent_path, f_name + '_' + str(view_combination[i][0]) + '.txt'))
            channel_2 = np.loadtxt(
                os.path.join(shape_parent_path, f_name + '_' + str(view_combination[i][1]) + '.txt'))
            channel_3 = np.loadtxt(
                os.path.join(shape_parent_path, f_name + '_' + str(view_combination[i][2]) + '.txt'))
            feature = np.dstack([channel_1, channel_2, channel_3])
            flag = np.sum(np.isinf(feature).astype(int))
            if flag > 0:
                print("Inf Error: {}".format(f_name))
            shape_multiview_x.append(feature)

        for j in range(config['texture_views']):
            texture_pairs = np.loadtxt(os.path.join(texture_parent_path, f_name + 'pd' + str(j) + '.txt'))
            texture_multiview_x.append(texture_pairs)

        if isVenation:
            for m in range(config['vein_views']):
                vein_pairs = np.loadtxt(os.path.join(vein_parent_path, f_name + '-pd' + str(m) + '.txt'))
                vein_multiview_x.append(vein_pairs)

        img_f_path = os.path.join(path)
        img = skio.imread(img_f_path)
        img = resize(img, [config['image_size'][0], config['image_size'][1], 3])

        # print(np.max(img))
        # img = img/255
        shape_x.append(shape_multiview_x)
        texture_x.append(texture_multiview_x)
        img_x.append(img)
        y.append(int(d))
        if isVenation:
            vein_x.append(vein_multiview_x)
            return img_x, shape_x, texture_x, vein_x, y
    return img_x, shape_x, texture_x, y


def data_loader_for_xception_model(file_list, config):
    img_x = []
    y = []
    for path in file_list:
        path = path
        strs = str.split(path, '/')
        if strs[-2].startswith("yd"):
            d = strs[-2][2:]
        else:
            d = strs[-2]
        img_f_path = os.path.join(path)
        img = skio.imread(img_f_path)
        img = resize(img, [config['image_size'][0], config['image_size'][1], 3])
        # print(np.max(img))
        # img = img/255
        img_x.append(img)
        y.append(int(d))
    return img_x, y


def get_dataset_file_list(img_path):
    dirs = os.listdir(img_path)
    x_list = []
    y_list = []
    for d in dirs:
        if d.startswith("yd"):
            cultivar = int(d[2:])
        else:
            cultivar= int(d)
        parent_path = os.path.join(img_path, d)
        files = os.listdir(parent_path)
        for f in files:
            f_path = os.path.join(parent_path, f)
            x_list.append(f_path)
            y_list.append(cultivar)
    return x_list, y_list


def get_dataset_file_list_soybean(img_path, period):
    dirs = os.listdir(img_path)
    x_list = []
    y_list = []
    for d in dirs:
        cultivar = int(d)
        parent_path = os.path.join(img_path, d, period)
        files = os.listdir(parent_path)
        for f in files:
            f_path = os.path.join(parent_path, f)
            x_list.append(f_path)
            y_list.append(cultivar)
    return x_list, y_list


def create_dirs(config, cultivar=None, period=None, isVenation=False):
    if not os.path.exists(config['img_path']):
        os.mkdir(config['img_path'])
    if not os.path.exists(config['shape_data_path']):
        os.mkdir(config['shape_data_path'])
    if not os.path.exists(config['texture_data_path']):
        os.mkdir(config['texture_data_path'])
    if isVenation:
        if not os.path.exists(config['vein_data_path']):
            os.mkdir(config['vein_data_path'])
    types = ['img_path', 'texture_data_path', 'shape_data_path']
    if isVenation:
        types.append('vein_data_path')

    for type in types:
        base_path = config[type]
        if not os.path.exists(base_path):
            os.mkdir(base_path)
        sec_path = os.path.join(base_path, cultivar)
        if not os.path.exists(sec_path):
            os.mkdir(sec_path)
        if period:
            third_path = os.path.join(sec_path, period)
            if not os.path.exists(third_path):
                os.mkdir(third_path)

