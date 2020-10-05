import numpy as np
configs = {}

# default configuration
shape_views = 30
texture_views = 2
vein_views = 2
stage1_kr = 0.3
stage1_dropout = 0.5
stage1_neuron_num = 512
pht_threshold_shape = 0.0001
pht_threshold_texture = 0.0001
pht_threshold_vein = 0.0001
shape_point_num = 700
texture_and_vein_point_num = 1000

point_number_list = list(np.zeros(shape_views) + shape_point_num)
point_number_list.extend(list(np.zeros(texture_views+vein_views) + texture_and_vein_point_num))

sequence = []
sequence.append(shape_views - 1)
sequence.extend(list(range(0, shape_views)))
sequence.append(0)
print(sequence)

# 预定义形状PD各个方向结合的组合
view_combination = []
for i in range(1, len(sequence) - 1):
    view_combination.append(sequence[i - 1:i + 2])
print(view_combination)


configs['soybean_model'] = {
    "texture_data_path": r'/mfcis/dataset/soybean/soybean-dataset-texture-txt',
    "vein_data_path": r"/mfcis/dataset/soybean/soybean-dataset-vein-txt",
    "shape_data_path": r"/mfcis/dataset/soybean/soybean-dataset-contour-pdm",
    "img_path": r"/mfcis/dataset/soybean/soybean_cultivar100_img",
    "views": 34,
    "shape_views": shape_views,
    "texture_views": texture_views,
    "vein_views": vein_views,
    "learning_rate": 0.001,

    "stage1_kr": stage1_kr,
    "stage1_dropout": stage1_dropout,
    "stage1_neuron_num": stage1_neuron_num,

    "pht_threshold_shape": pht_threshold_shape,
    "pht_threshold_texture": pht_threshold_texture,
    "pht_threshold_vein": pht_threshold_vein,

    "N": [int(d) for d in point_number_list],
    "classes": 100,
    "image_size": (256, 256),
    "regx_str": r'\S{1,}_t'
}

configs['cherry_model'] = {
    "texture_data_path": r"/home/zyp/pda-docker-data/final_experiment/cherry_dataset_cultivar88/texture_pd",
    "shape_data_path": r"/home/zyp/pda-docker-data/final_experiment/cherry_dataset_cultivar88/shape_pd",
    "vein_data_path": r"/home/zyp/pda-docker-data/final_experiment/cherry_dataset_cultivar88/vein_pd",
    "img_path": r"/home/zyp/pda-docker-data/final_experiment/cherry_dataset_cultivar88/cherry_jpg256_cultivar88",
    "views": 34,
    "shape_views": shape_views,
    "texture_views": texture_views,
    "vein_views": vein_views,
    "learning_rate": 0.001,

    "stage1_kr": stage1_kr,
    "stage1_dropout": stage1_dropout,
    "stage1_neuron_num": stage1_neuron_num,

    "pht_threshold_shape": pht_threshold_shape,
    "pht_threshold_texture": pht_threshold_texture,
    "pht_threshold_vein": pht_threshold_vein,

    "N": [int(d) for d in point_number_list],
    "classes": 88,
    "image_size": (256, 256),
    "regx_str": r'\S{1,}_d'
}

configs['swedish_model'] = {
    "texture_data_path": r"/mfcis/dataset/swedish/swedish-dataset-texture-txt",
    "shape_data_path": r"/mfcis/dataset/swedish/swedish-dataset-contour-pdm",
    "img_path": r"/mfcis/dataset/swedish/swedish-dataset-png",
    "views": 32,
    "shape_views": shape_views,
    "texture_views": 2,
    "stage1_kr": 0.3,
    "stage1_dropout": 0.5,
    "stage1_neuron_num": 512,
    "pht_threshold_shape": 0.0001,
    "pht_threshold_texture": 0.0001,
    "N": [int(d) for d in point_number_list],
    "classes": 15,
    "image_size": (256, 256),
    "regx_str": r'l\d{1,}\S{1,}\d{1,3}'
}

configs['flavia_model'] = {
    "texture_data_path": r"/home/zyp/public_dataset/flavia-dataset/flavia-dataset-texture-txt",
    "shape_data_path": r"/home/zyp/public_dataset/flavia-dataset/flavia-pdm",
    "img_path": r"/home/zyp/public_dataset/flavia-dataset/flavia",
    "views": 32,
    "shape_views": shape_views,
    "texture_views": 2,
    "stage1_kr": 0.3,
    "stage1_dropout": 0.5,
    "stage1_neuron_num": 512,
    "pht_threshold_shape": 0.0001,
    "pht_threshold_texture": 0.0001,
    "N": [int(d) for d in point_number_list],
    "classes": 32,
    "image_size": (256, 192),
    "regx_str":r'\d{4,}'
}
