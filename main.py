from config.model_configuration import *
from models import BaseModel
from utils import utils
from feature_extraction import extract_features
from keras.utils import to_categorical
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
import re
import json
import os
from multiprocessing import Pool
from skimage import io as skio

iteration = 10


def xception(dataset, config, period=''):
    # config = configs[dataset+"_model"]
    x_list, y_list = utils.get_dataset_file_list(config['img_path'])
    img_x_list, y_list = utils.data_loader_for_xception_model(file_list=x_list, config=config)
    xception_model_training_and_test(img_x_list=img_x_list, y_list=y_list, config=config)


def xception_model_training_and_test(img_x_list, y_list, config):
    x = np.array(img_x_list)
    y = np.array(y_list)
    dataset = config['dataset']
    period = config['period']
    id_map = np.loadtxt(dataset+'_id.txt')
    for index, d in enumerate(y):
        for label in id_map:
            if d == label[0]:
                y[index] = label[1]

    y_one_hot = to_categorical(y)

    lr_adjust = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.5,
                                  patience=5,
                                  min_lr=1e-5)

    result = []
    for i in range(iteration):
        index = np.arange(len(y))
        # print(index)
        X_train_index, X_test_index, y_train_index, y_test_index = train_test_split(index, y,
                                                                        test_size=0.3,
                                                                        random_state=i,
                                                                        shuffle=True,
                                                                        stratify=y)
        print(len(X_train_index))
        print(len(X_test_index))
        np.save('{}_iteration_{}_img_{}_xception_train_index.npy'.format(dataset, i, period), X_train_index)
        np.save('{}_iteration_{}_img_{}_xception_test_index.npy'.format(dataset, i, period), X_test_index)
        X_train = x[X_train_index]
        X_test = x[X_test_index]
        y_train = y_one_hot[y_train_index]
        y_test = y_one_hot[y_test_index]

        save_bset_weight = ModelCheckpoint('xception_img_{}_itreation-{}-{}.hdf5'.format(dataset, i, period),
                                           monitor='val_loss', verbose=1, save_best_only=True, mode='auto',
                                           save_weights_only=True)
        # you can change the parallels to create multi_gpu_model if you have more than one GPU available
        model = BaseModel.Xception_Model(parallels=0, config=config)
        # you should set a smaller batch_size if you GPU memory is limited
        model.fit(X_train, y_train, batch_size=100, epochs=100, validation_split=0.1,
                  callbacks=[lr_adjust, save_bset_weight])
        K.clear_session()

        model2 = BaseModel.Xception_Model(parallels=0, config=config)
        model2.load_weights('xception_img_{}_itreation-{}-{}.hdf5'.format(dataset, i, period))


        score = model2.evaluate(X_test, y_test)
        print(score)

        pre_final = model2.predict(X_test, batch_size=100)
        y_test_label = np.array([np.argmax(d) for d in y_test])
        y_pre_label = np.array([np.argmax(d) for d in pre_final])
        performance = get_performance(y_pre_label, y_test_label)
        performance['test_loss'] = score[0]
        performance['test_acc'] = score[1]
        K.clear_session()
        result.append(performance)
        json_str = json.dumps(performance, indent=4)
        with open('{}_xception-iteration-{}-{}-result.json'.format(dataset, i, period), 'w') as json_file:
            json_file.write(json_str)
        plot_result(result)


def get_performance(y_pre_label, y_test_label):
    performance = {}
    report = classification_report(y_test_label, y_pre_label)
    performance['report'] = report
    performance['precision'] = precision_score(y_test_label, y_pre_label, average='macro')
    performance['recall'] = recall_score(y_test_label, y_pre_label, average='macro')
    performance['accuracy'] = accuracy_score(y_test_label, y_pre_label)
    performance['f1_score'] = f1_score(y_test_label, y_pre_label, average='macro')
    return performance


def plot_result(result):
    precision = []
    recall = []
    f1 = []
    acc = []
    for d in result:
        precision.append(d['precision'])
        recall.append(d['recall'])
        f1.append(d['f1_score'])
        acc.append(d['test_acc'])
    plt.figure(figsize=(10, 6))
    plt.plot(precision, label='precision', marker='*')
    plt.plot(recall, label='recall', marker='o')
    plt.plot(f1, label='f1_score', marker='x')
    plt.plot(acc, label='accuracy', marker='s')
    plt.ylim([0.9, 1.2])
    plt.xlabel('iteration')
    plt.ylabel('score')
    plt.legend()
    plt.show()


def lr_reducer(epoch):
    lr = K.get_value(model.optimizer.lr)
    if epoch < 10:
        lr = epoch/5 * 0.001
    if epoch == 10:
        lr = 0.001
    K.set_value(model.optimizer.lr, lr)
    print("current lr is : {}".format(lr))
    return lr


def tp_xception(dataset, config,  isVenation=False, period='N'):
    # config = configs[dataset + "_model"]

    x_list, y_list = utils.get_dataset_file_list(config['img_path'])
    vein_x = []
    if isVenation:
        img_x_list, shape_x, texture_x, vein_x, y_list = utils.data_loader_for_combined_model(file_list=x_list,
                                                                  dataset=dataset,
                                                                  config=config,
                                                                  isVenation=isVenation)
    else:
        img_x_list, shape_x, texture_x, y_list = utils.data_loader_for_combined_model(file_list=x_list,
                                                                  dataset=dataset,
                                                                  config=config,
                                                                  isVenation=isVenation)

    tp_xception_model_training_and_test(img_x_list, shape_x, texture_x, vein_x, isVenation, y_list, config)


def tp_xception_model_training_and_test(img_x_list, shape_x, texture_x, vein_x, isVenation, y_list, config):
    img_x = np.array(img_x_list)
    shape_x = np.array(shape_x)
    texture_x = np.array(texture_x)
    if isVenation:
        vein_x = np.array(vein_x)
    y = np.array(y_list)
    dataset = config['dataset']
    id_map = np.loadtxt(dataset+'_id.txt')
    for index, d in enumerate(y):
        for label in id_map:
            if d == label[0]:
                y[index] = label[1]
    y_one_hot = to_categorical(y)
    result = []
    lr_adjust = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.5,
                                  patience=5,
                                  min_lr=1e-5)
    dataset = config['dataset']
    period = config['period']

    for i in range(iteration):
        index = np.arange(len(y))
        # print(index)
        X_train_index, X_test_index, y_train_index, y_test_index = train_test_split(index, y_one_hot,
                                                                                    test_size=0.3,
                                                                                    random_state=i,
                                                                                    shuffle=True,
                                                                                    stratify=y)
        print(len(X_train_index))
        print(len(X_test_index))
        np.save('{}_iteration_{}_img_{}_tp_xception_train_index.npy'.format(dataset, i, period), X_train_index)
        np.save('{}_iteration_{}_img_{}_tp_xception_test_index.npy'.format(dataset, i, period), X_test_index)

        shape_x_train = shape_x[X_train_index]
        texture_x_train = texture_x[X_train_index]
        img_x_train = img_x[X_train_index]
        shape_x_train_list = [shape_x_train[:, i, :, :] for i in range(config["shape_views"])]
        texture_x_train_list = [texture_x_train[:, i, :, :] for i in range(config["texture_views"])]
        y_train = y_one_hot[y_train_index]

        if isVenation:
            vein_x_train = vein_x[X_train_index]
            vein_x_train[:, 0, :, 1] = (vein_x_train[:, 0, :, 1] - np.mean(vein_x_train[:, 0, :, 1])) / np.std(
                vein_x_train[:, 0, :, 1])
            vein_x_train[:, 1, :, 1] = (vein_x_train[:, 1, :, 1] - np.mean(vein_x_train[:, 1, :, 1])) / np.std(
                vein_x_train[:, 1, :, 1])
            vein_x_train_list = [vein_x_train[:, i, :, :] for i in range(config["vein_views"])]

        x_train_list = []

        for index, d in enumerate(texture_x_train_list):
            texture_x_train_list[index] = np.reshape(d, [d.shape[0], d.shape[1], d.shape[2], 1])
        if isVenation:
            for index, d in enumerate(vein_x_train_list):
                vein_x_train_list[index] = np.reshape(d, [d.shape[0], d.shape[1], d.shape[2], 1])

        x_train_list.extend(shape_x_train_list)
        x_train_list.extend(texture_x_train_list)
        if isVenation:
            x_train_list.extend(vein_x_train_list)
        x_train_list.append(img_x_train)

        lr_reduce = LearningRateScheduler(lr_reducer)

        y_train_label = [np.argmax(d) for d in y_train]

        class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train_label), y_train_label)
        save_bset_weight = ModelCheckpoint('./{}_pdm_iteration-{}-{}.hdf5'.format(dataset, i, period),
                                           monitor='val_loss', verbose=1, save_best_only=True, mode='auto',
                                           save_weights_only=True)

        model = BaseModel.Combined_Model(parallels=4, config=config)
        model.fit(x_train_list, y_train, batch_size=45, epochs=100, validation_split=0.1, class_weight=class_weights,
                  callbacks=[lr_reduce,
                             ReduceLROnPlateau(monitor='val_loss', patience=3, min_lr=1e-6, factor=0.5),
                             save_bset_weight,
                             lr_adjust
                             ])

        shape_x_test = shape_x[X_test_index]
        texture_x_test = texture_x[X_test_index]

        img_x_test = img_x[X_test_index]
        y_test = y_one_hot[y_test_index]

        shape_x_test_list = [shape_x_test[:, i, :, :] for i in range(config["shape_views"])]
        texture_x_test_list = [texture_x_test[:, i, :, :] for i in range(config["texture_views"])]

        if isVenation:
            vein_x_test = vein_x[X_test_index]
            vein_x_test_list = [vein_x_test[:, i, :, :] for i in range(config["vein_views"])]

        x_test_list = []

        for index, d in enumerate(texture_x_test_list):
            texture_x_test_list[index] = np.reshape(d, [d.shape[0], d.shape[1], d.shape[2], 1])

        x_test_list.extend(shape_x_test_list)
        x_test_list.extend(texture_x_test_list)

        if isVenation:
            for index, d in enumerate(vein_x_test_list):
                vein_x_test_list[index] = np.reshape(d, [d.shape[0], d.shape[1], d.shape[2], 1])
                x_test_list.extend(vein_x_test_list)

        x_test_list.append(img_x_test)

        K.clear_session()

        model2 = BaseModel.Combined_Model(parallels=4, config=config)
        model2.load_weights('./{}_pdm_iteration-{}-{}.hdf5'.format(dataset, i, period))
        score = model2.evaluate(x_test_list, y_test, batch_size=128)
        print(score)
        y_test_label = np.array([np.argmax(d) for d in y_test])
        y_pre = model2.predict(x_test_list)
        y_pre_label = np.array([np.argmax(d) for d in y_pre])

        performance = get_performance(y_pre_label, y_test_label)

        performance['test_acc'] = score[1]
        performance['test_loss'] = score[0]

        result.append(performance)
        K.clear_session()
        print("precision_score: {}".format(performance['precision']))
        print("recall_score: {}".format(performance['recall']))
        print("f1_score: {}".format(performance['f1_score']))
        json_str = json.dumps(performance, indent=4)
        with open('./{}_pd-rgb-combined-iteration-{}-{}.json'.format(dataset, i, period), 'w') as json_file:
            json_file.write(json_str)

        plot_result(result)


def xception_multi_period_score_fusion(dataset, config):
    result = []
    for i in range(iteration):
        pre_list = []
        y_test_list = []
        for period in ['R1', 'R3', 'R4', 'R5', 'R6']:
            model = BaseModel.Xception_Model(parallels=4, config=config)
            model.load_weights('xception_img_{}_itreation-{}-{}.hdf5'.format(dataset, i, period))
            file_list = np.loadtxt('{}-{}_file_list.txt'.format(dataset, period))
            img_x_list, y_list = utils.data_loader_for_xception_model(file_list=file_list, config=config)
            x = np.array(img_x_list)
            y = np.array(y_list)
            id_map = np.loadtxt(dataset + '_id.txt')
            for index, d in enumerate(y):
                for label in id_map:
                    if d == label[0]:
                        y[index] = label[1]

            y_one_hot = to_categorical(y)
            test_index = np.load('{}_iteration_{}_img_{}_xception_test_index.npy'.format(dataset, i, period))
            img_x_test = x[test_index]
            y_test = y_one_hot[test_index]
            y_test_list.append(y_test)
            pre = model.predict(img_x_test)
            pre_list.append(pre)
        pre_final_arr = np.array(pre_list)
        pre_final = np.sum(pre_final_arr, axis=2)
        pre_final_label = [np.argmax(d) for d in pre_final]
        for i in range(1,len(y_test_list)+1):
            if y_test_list[i-1] != y_test_list[i]:
                print("The test label of different period should be the same")
                return -1
        y_test_label = [np.argmax(d) for d in y_test_list[0]]

        performance = get_performance(pre_final_label, y_test_label)
        result.append(performance)

    plot_result(result)


def tp_xception_multi_period_score_fusion(dataset, config):
    result = []
    for i in range(iteration):
        pre_list = []
        y_test_list = []
        for period in ['R1', 'R3', 'R4', 'R5', 'R6']:
            model = BaseModel.Combined_Model(parallels=4, config=config)
            model.load_weights('./{}_pdm_iteration-{}-{}.hdf5'.format(dataset, i, period))
            file_list = np.loadtxt('{}-{}_file_list.txt'.format(dataset, period))
            img_x_list, y_list = utils.data_loader_for_xception_model(file_list=file_list, config=config)
            x = np.array(img_x_list)
            y = np.array(y_list)
            id_map = np.loadtxt(dataset + '_id.txt')
            for index, d in enumerate(y):
                for label in id_map:
                    if d == label[0]:
                        y[index] = label[1]
            y_one_hot = to_categorical(y)
            test_index = np.load('{}_iteration_{}_img_{}_tp_xception_test_index.npy'.format(dataset, i, period))
            img_x_test = x[test_index]
            y_test = y_one_hot[test_index]
            y_test_list.append(y_test)
            pre = model.predict(img_x_test)
            pre_list.append(pre)
        pre_final_arr = np.array(pre_list)
        pre_final = np.sum(pre_final_arr, axis=2)
        pre_final_label = [np.argmax(d) for d in pre_final]
        for i in range(1, len(y_test_list)+1):
            if y_test_list[i-1] != y_test_list[i]:
                print("The test label of different period should be the same")
                return -1
        y_test_label = [np.argmax(d) for d in y_test_list[0]]

        performance = get_performance(pre_final_label, y_test_label)
        result.append(performance)

    plot_result(result)


def mixing_all_period_tp_xception(config):
    x_list = []
    y_list = []
    dataset = 'soybean'
    for period in ['R1', 'R3', 'R4', 'R5', 'R6']:
        tmp_x_list, tmp_y_list = utils.get_dataset_file_list(os.path.join(config['img_path'], period))
        x_list.append(tmp_x_list)
        y_list.append(tmp_y_list)
    img_x_list, shape_x, texture_x, vein_x, y_list = utils.data_loader_for_combined_model(file_list=x_list,
                                                                                          dataset=dataset,
                                                                                          config=config,
                                                                                          isVenation=True)
    tp_xception_model_training_and_test(img_x_list=img_x_list,
                                        shape_x=shape_x,
                                        texture_x=texture_x,
                                        vein_x=vein_x,
                                        isVenation=True,
                                        config=config)


def mixing_all_period_xception(config):
    x_list = []
    y_list = []
    for period in ['R1', 'R3', 'R4', 'R5', 'R6']:
        tmp_x_list, tmp_y_list = utils.get_dataset_file_list(os.path.join(config['img_path'], period))
        x_list.append(tmp_x_list)
        y_list.append(tmp_y_list)

    img_x_list, y_list = utils.data_loader_for_xception_model(file_list=x_list, config=config)
    xception_model_training_and_test(img_x_list=img_x_list, y_list=y_list)


def feature_extraction(dataset, period, config, isVenation=False):
    #utils.create_dirs(config, period, isVenation=False)
    img_path = config['img_path']
    task_list = []
    cultivars = os.listdir(img_path)
    if dataset == 'soybean':
        soybean_regx = re.compile(config['regx_str'])
        for cultivar in cultivars:
            files = os.listdir(os.path.join(img_path, cultivar, period))
            for f in files:
                f_name = soybean_regx.findall(f)[0]
                task = (os.path.join(img_path, cultivar, period, f), f_name, config, cultivar, period, isVenation)
                task_list.append(task)
    else:
        regx = re.compile(config['regx_str'])
        for cultivar in cultivars:
            files = os.listdir(os.path.join(img_path, cultivar))
            for f in files:
                f_name = regx.findall(f)[0]
                task = (os.path.join(img_path, cultivar, f), f_name, config, cultivar, period, isVenation)
                task_list.append(task)

    extract_feature_bat(task_list)
    # You can accelerate the feature extraction process by using multiprocess
    # Uncomment the code below to accelerate the computation, and comment the line 399 'extract_feature_bat()'.
    # p = 20
    # pool = Pool(p)
    # task_size = len(task_list)
    # delta = task_size // p
    # for i in range(p):
    #     tmp_list = []
    #     if (i+1) * delta < task_size:
    #         tmp_list = task_list[i*delta:(i+1)*delta]
    #     else:
    #         tmp_list = task_list[i*delta:]
    #
    #     pool.apply_async(extract_feature_bat, args=(tmp_list))


def extract_feature_bat(task_list):
    for task in task_list:
        img = skio.imread(task[0])
        try:
            extract_features.compute_pds(img, name=task[1], config=task[2], cultivar=task[3], period=task[4], isVenation=task[5])
        except Exception as e:
            print(e)
            continue
            print(task)


if __name__ == "__main__":
    '''
    Xception Model and TP+Xception Model evaluation
    '''
    ''' #############################################################################
        ##### 1.You can uncomment the code of which experiment you want to run ######
        ##### 2.The dataset path config can be found in the config folder,     ######
        #####   and dont't change the directory structure of the datasets.     ######
        ##### 3.The computation of PD rely on HomCloud and Dipha, please       ######
        #####      confirm the environment is properly configured!!!           ######
        #####      Please read the readme.md file first!!!                     ######
        #############################################################################
    '''
    '''
       1. We tested and refactored the code on a linux platform with multi-processors and 4 GPU.
       2.You can change the training parameters according to the performance of your own device.
       3.Because the topological feature extraction is a time consuming process, we recommend 
        you save the result of feature extraction.
       4. The dataset folder structure should be set as below: (you can also use different setting, 
       then you should modify the code)
       5. The code configuration can be found in the file config/model_configuration.py
       Example:
       
       soybean [dataset name]
       |--shape-feature                 [shape feature folder]
          |--class 1                    [the class number e.g. 022]
          |   |--period                 [optional, only the soybean dataset have different growth period, e.g. R1]
          |   |   |--file0_0.txt        [the topological feature of direction 0]
          |   |   |--file0_1.txt
          |   |   |--file0_2.txt
          |   |   |-- ...
          |   |   |--file0_29.txt
          |   |   |-- ...
          |   |   |--filen_m.txt
          |--class 2
          |   |--period                
          |   |   |--file0_0.txt        
          |   |   |--file0_1.txt
          |   |   |--file0_2.txt
          |   |   |-- ...
          |   |   |--file0_29.txt
          |   |   |-- ...
          |   |   |--filen_m.txt
       |--texture-feature
          |--class 1                      [the class number e.g. 022]
          |   |--period                   [optional, only the soybean dataset have different growth period, e.g. R1]
          |   |   |--file0_pd0.txt        [the topological feature of pd0]
          |   |   |--file0_pd1.txt        [the topological feature of pd1]
          |   |   |--file0_2.txt
          |   |   |-- ... 
          |   |   |--filen_pd1.txt
          |--class 2
          |   |--period                
          |   |   |--file0_pd0.txt        
          |   |   |--file0_pd1.txt        
          |   |   |--file0_2.txt
          |   |   |-- ... 
          |   |   |--filen_pd1.txt       
       |--venation-feature
          |--class 1                      [the class number e.g. 022]
          |   |--period                   [optional, only the soybean dataset have different growth period, e.g. R1]
          |   |   |--file0_pd0.txt        [the topological feature of pd0]
          |   |   |--file0_pd1.txt        [the topological feature of pd1]
          |   |   |--file0_2.txt
          |   |   |-- ... 
          |   |   |--filen_pd1.txt
          |--class 2
          |   |--period                
          |   |   |--file0_pd0.txt        
          |   |   |--file0_pd1.txt        
          |   |   |--file0_2.txt
          |   |   |-- ... 
          |   |   |--filen_pd1.txt                           
       |--leaf-image
          |--class 1                      [the class number e.g. 022]
          |   |--period                   [optional, only the soybean dataset have different growth period, e.g. R1]
          |   |   |--file0.png            [the topological feature of pd0]
          |   |   |-- ... 
          |   |   |--filen.png
              |--class 1                  
          |   |--period                   
          |   |   |--file0.png            
          |   |   |-- ... 
          |   |   |--filen.png       
         
    '''


    '''
        # -- Topological Feature Extraction -- #
        # This process is time consuming,so we dump pd as txt files, please ensure there enough disk space to store them.
        # The feature extraction process may take a long time, we recommend you run the code on a multi-core computer,
        # we set the process default process number to 20,
        # you can modify it in the 'feature_extraction' function in this file.
    '''
    # feature_extraction(dataset='soybean', period='R1', config=configs['soybean_model'], isVenation=True)
    # feature_extraction(dataset='soybean', period='R3', config=configs['soybean_model'], isVenation=True)
    # feature_extraction(dataset='soybean', period='R4', config=configs['soybean_model'], isVenation=True)
    # feature_extraction(dataset='soybean', period='R5', config=configs['soybean_model'], isVenation=True)
    # feature_extraction(dataset='soybean', period='R6', config=configs['soybean_model'], isVenation=True)
    #
    # feature_extraction(dataset='cherry', period=None, config=configs['flavia_model'], isVenation=False)
    # feature_extraction(dataset='flavia', period=None, config=configs['flavia_model'], isVenation=False)
    # feature_extraction(dataset='swedish', period=None, config=configs['swedish_model'], isVenation=False)

    '''
    # -- Xception Model --
    # evaluate the xception model on swedish dataset
    '''
    # dataset = 'swedish'
    # config_swedish = configs['swedish_model']
    # config_swedish['dataset'] = dataset
    # xception(dataset)

    # evaluate the xception model on flavia dataset
    # dataset = 'flavia'
    # config_flavia['dataset'] = dataset
    # config_flavia = configs['flavia_model']
    # xception(dataset)


    '''
    # evalute the xception model on soybean dataset
    # R1 period
    '''
    dataset = 'soybean'
    period = 'R1'
    config_soybean_R1 = configs['soybean_model']
    config_soybean_R1['dataset'] = dataset
    config_soybean_R1['period'] = period
    config_soybean_R1['img_path'] = os.path.join(config_soybean_R1['img_path'], period)
    config_soybean_R1['shape_data_path'] = os.path.join(config_soybean_R1['shape_data_path'], period)
    config_soybean_R1['texture_data_path'] = os.path.join(config_soybean_R1['texture_data_path'], period)
    config_soybean_R1['vein_data_path'] = os.path.join(config_soybean_R1['vein_data_path'], period)
    xception(dataset='soybean', config=config_soybean_R1, period='R1')

    # R3 period
    # dataset = 'soybean'
    # period = 'R3'
    # config_soybean_R3 = configs['soybean_model']
    # config_soybean_R3['dataset'] = dataset
    # config_soybean_R3['period'] = period
    # config_soybean_R3['img_path'] = os.path.join(config_soybean_R3['img_path'], period)
    # config_soybean_R3['shape_data_path'] = os.path.join(config_soybean_R3['shape_data_path'], period)
    # config_soybean_R3['texture_data_path'] = os.path.join(config_soybean_R3['texture_data_path'], period)
    # config_soybean_R3['vein_data_path'] = os.path.join(config_soybean_R3['vein_data_path'], period)
    # xception(dataset='soybean', config=config_soybean_R3, period='R3')

    # R4 period
    # dataset = 'soybean'
    # period = 'R4'
    # config_soybean_R4 = configs['soybean_model']
    # config_soybean_R4['dataset'] = dataset
    # config_soybean_R4['period'] = period
    # config_soybean_R4['img_path'] = os.path.join(config_soybean_R4['img_path'], period)
    # config_soybean_R4['shape_data_path'] = os.path.join(config_soybean_R4['shape_data_path'], period)
    # config_soybean_R4['texture_data_path'] = os.path.join(config_soybean_R4['texture_data_path'], period)
    # config_soybean_R4['vein_data_path'] = os.path.join(config_soybean_R4['vein_data_path'], period)
    # xception(dataset='soybean', config=config_soybean_R4, period='R4')

    # R5 period
    # dataset = 'soybean'
    # period = 'R5'
    # config_soybean_R5 = configs['soybean_model']
    # config_soybean_R5['dataset'] = dataset
    # config_soybean_R5['period'] = period
    # config_soybean_R5['img_path'] = os.path.join(config_soybean_R5['img_path'], period)
    # config_soybean_R5['shape_data_path'] = os.path.join(config_soybean_R5['shape_data_path'], period)
    # config_soybean_R5['texture_data_path'] = os.path.join(config_soybean_R5['texture_data_path'], period)
    # config_soybean_R5['vein_data_path'] = os.path.join(config_soybean_R5['vein_data_path'], period)
    # xception(dataset='soybean', config=config_soybean_R5, period='R5')

    # R6 period
    # dataset = 'soybean'
    # period = 'R6'
    # config_soybean_R6 = configs['soybean_model']
    # config_soybean_R6['dataset'] = dataset
    # config_soybean_R6['period'] = period
    # config_soybean_R6['img_path'] = os.path.join(config_soybean_R6['img_path'], period)
    # config_soybean_R6['shape_data_path'] = os.path.join(config_soybean_R6['shape_data_path'], period)
    # config_soybean_R6['texture_data_path'] = os.path.join(config_soybean_R6['texture_data_path'], period)
    # config_soybean_R6['vein_data_path'] = os.path.join(config_soybean_R6['vein_data_path'], period)
    # xception(dataset='soybean', config=config_soybean_R6, period='R6')

    '''
    #evaluate the xception model on cherry dataset
    '''
    # dataset = 'cherry'
    # config_cherry = configs['cherry_model']
    # config_cherry['dataset'] = dataset
    # xception(dataset = dataset, config=config_cherry)

    '''
    # --TP+Xception Model --
    '''
    # tp_xception('swedish', config_swedish, isVenation=False)
    #
    # tp_xception('flavia', config_flavia, isVenation=False)
    #
    # tp_xception('cherry', config_cherry, isVenation=True)
    #
    # tp_xception('soybean', config_soybean_R1, isVenation=True)
    #
    # tp_xception('soybean', config_soybean_R3, isVenation=True)
    #
    # tp_xception('soybean', config_soybean_R4, isVenation=True)
    #
    # tp_xception('soybean', config_soybean_R5, isVenation=True)
    #
    # tp_xception('soybean', config_soybean_R6, isVenation=True)

    '''
    #-- Soybean TP+Xception model score fusion
    '''
    # tp_xception_multi_period_score_fusion(dataset='soybean', config=configs['soybean_model'])

    '''
    #-- Soybean Xception model score fusion
    '''
    # xception_multi_period_score_fusion(dataset='soybean', config=configs['soybean_model'])

    '''
    #-- Soybean TP+Xception Mixing all period
    '''
    # mixing_all_period_tp_xception(config=configs['soybean_model'])

    '''
    #-- Soybean Xception Mxing all period
    '''
    # mixing_all_period_xception(config=configs['soybean_model'])

