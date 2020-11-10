import numpy as np
import os
from tqdm import *
import re
from skimage import io as skio
from skimage.transform import resize
from config.model_configuration import configs, view_combination, pht_threshold_shape, pht_threshold_vein, pht_threshold_texture,shape_point_num, texture_and_vein_point_num
from multiprocessing.pool import Pool
from multiprocessing import Manager
from config.model_configuration import process_number

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


def data_loader_for_combined_model_bat(file_list, dataset, config, isVenation):
    list_size = len(file_list)
    pool = Pool(process_number)
    img_x = []
    shape_x = []
    texture_x = []
    vein_x = []
    y_x = []
    delta = int(list_size / process_number) + 1
    results = []
    for i in range(process_number):
        if (i+1) * delta > list_size:
            task_list = file_list[i*delta:]
        else:
            task_list = file_list[i*delta: (i+1)*delta]
        result_map = pool.apply_async(data_loader_for_combined_model, args=(task_list, dataset, config, isVenation))
        results.append(result_map)

    for res in results:
        data = res.get()
        img_x.append(data['img_x'])
        shape_x.append(data['shape_x'])
        texture_x.append(data['texture_x'])
        y_x.append(data['y'])
        if isVenation:
            vein_x.append(data['vein_x'])

    pool.close()
    pool.join()
    # img_x = np.load("cherry_img_x.npy", allow_pickle=True)
    # shape_x = np.load("cherry_shape_x.npy", allow_pickle=True)
    # texture_x = np.load("cherry_texture_x.npy", allow_pickle=True)
    # vein_x = np.load("cherry_vein_x.npy", allow_pickle=True)
    # y_x = np.load("cherry_y.npy", allow_pickle=True)
    tmp_img_x = np.zeros([list_size, 256, 256, 3])
    tmp_shape_x = np.zeros([list_size, 30, 700, 2, 3])
    tmp_texture_x = np.zeros([list_size, 2, 1000, 2])
    tmp_vein_x = np.zeros([list_size, 2, 1000, 2])
    tmp_y = np.zeros([list_size, 1])
    delta = int(list_size / process_number)+1
    for i in range(process_number):
        for j in range(len(img_x[i])):
            start_index = i * delta
            tmp_img_x[i*delta + j] = img_x[i][j]
            tmp_shape_x[start_index+j] = shape_x[i][j]
            tmp_texture_x[start_index+j] = texture_x[i][j]
            tmp_vein_x[start_index+j] = vein_x[i][j]
            tmp_y[start_index+j] = y_x[i][j]
    # img_x = np.array(tmp_img_x)
    # shape_x = np.array(tmp_shape_x)
    # texture_x = np.array(tmp_texture_x)
    # vein_x = np.array(tmp_vein_x)
    # y_x = np.array(tmp_y)
    img_x = tmp_img_x
    shape_x = tmp_shape_x
    texture_x = tmp_texture_x
    vein_x = tmp_vein_x
    y_x = tmp_y
    if isVenation:
        return img_x, shape_x, texture_x, vein_x, y_x
    else:
        return img_x, shape_x, texture_x, y_x


def data_loader_for_combined_model(file_list, dataset, config, isVenation):
    shape_x = []
    texture_x = []
    img_x = []
    vein_x = []
    y = []
    maxVal = config['max_val']
    regx_str = config['regx_str']
    regx = re.compile(regx_str)
    for path in tqdm(file_list):
        path = path.strip()
        strs = str.split(path, '/')
        f_name = regx.findall(strs[-1])[0]
        if strs[-2].startswith("yd"):
            d = strs[-2][2:]
        else:
            d = strs[-2]
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

            if channel_1.size < 4:
                channel_1 = np.reshape(channel_1, [1, 2])
                channel_1 = np.repeat(channel_1, 100, axis=0)

            index1 = get_persistence(channel_1, shape_point_num)
            vec1 = channel_1[index1]
            vector1 = pht['shape'](vec1)

            channel_2 = np.loadtxt(
                os.path.join(shape_parent_path, f_name + '_' + str(view_combination[i][1]) + '.txt'))

            if channel_2.size < 4:
                channel_2 = np.reshape(channel_2, [1, 2])
                channel_2 = np.repeat(channel_2, 100, axis=0)

            index2 = get_persistence(channel_2, shape_point_num)
            vec2 = channel_2[index2]
            vector2 = pht['shape'](vec2)

            channel_3 = np.loadtxt(
                os.path.join(shape_parent_path, f_name + '_' + str(view_combination[i][2]) + '.txt'))

            if channel_3.size < 4:
                channel_3 = np.reshape(channel_3, [1, 2])
                channel_3 = np.repeat(channel_3, 100, axis=0)

            index3 = get_persistence(channel_3, shape_point_num)
            vec3 = channel_3[index3]
            vector3 = pht['shape'](vec3)

            feature = np.dstack([vector1, vector2, vector3])
            flag = np.sum(np.isinf(feature).astype(int))
            if flag > 0:
                print("Inf Error: {}".format(f_name))
            shape_multiview_x.append(feature)

        for j in range(config['texture_views']):

            if dataset == 'cherry':
                texture_pairs = np.loadtxt(os.path.join(texture_parent_path, f_name + '_pd' + str(j) + '.txt'))
            else:
                texture_pairs = np.loadtxt(os.path.join(texture_parent_path, f_name + '-pd' + str(j) + '.txt'))

            if texture_pairs.size < 4:
                texture_pairs = np.reshape(texture_pairs, [1, 2])
                texture_pairs = np.repeat(texture_pairs, 100, axis=0)

            index4 = get_persistence(texture_pairs, texture_and_vein_point_num)
            vec_texture = texture_pairs[index4]
            vec_texture = pht['texture'](vec_texture) / maxVal
            texture_multiview_x.append(vec_texture)

        if isVenation:
            for m in range(config['vein_views']):

                if dataset == 'cherry':
                    vein_pairs = np.loadtxt(os.path.join(vein_parent_path, f_name + '_pd' + str(m) + '.txt'))
                else:
                    vein_pairs = np.loadtxt(os.path.join(vein_parent_path, f_name + '-pd' + str(m) + '.txt'))

                if vein_pairs.size < 4:
                    vein_pairs = np.reshape(vein_pairs, [1, 2])
                    vein_pairs = np.repeat(vein_pairs, 100, axis=0)

                index5 = get_persistence(vein_pairs, texture_and_vein_point_num)
                vec_vein = vein_pairs[index5]
                vec_vein = pht['venation'](vec_vein) /maxVal
                vein_multiview_x.append(vec_vein)

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

    result_map = dict()
    result_map['img_x'] = img_x
    result_map['shape_x'] = shape_x
    result_map['texture_x'] = texture_x
    if isVenation:
        result_map['vein_x'] = vein_x
    result_map['y'] = y

    return result_map

    # if isVenation:
    #     return img_x, shape_x, texture_x, vein_x, y
    #
    # return img_x, shape_x, texture_x, y


def data_loader_for_xception_model(file_list, config):
    img_x = []
    y = []
    for path in tqdm(file_list):
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
    for d in tqdm(dirs):
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


