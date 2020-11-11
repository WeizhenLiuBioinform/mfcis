import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv1D, Dropout, Dense, BatchNormalization, Input, Concatenate, MaxPooling2D, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.applications.xception import Xception
import os
## set the id of available gpu e.g. "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# PD Layer 定义，用于纹理和叶脉的PD
class PD_Layer(Layer):
    def __init__(self, **kwargs):
        super(PD_Layer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.centers = self.add_weight(name='pd_center',
                                       shape=(input_shape[1], input_shape[2], input_shape[3]),
                                       initializer=keras.initializers.RandomNormal(),
                                       trainable=True)
        self.sharpness = self.add_weight(name='pd_sharpness',
                                         shape=(input_shape[1], input_shape[2], input_shape[3]),
                                         initializer=keras.initializers.Constant(4),
                                         trainable=True)
        super(PD_Layer, self).build(input_shape)

    def call(self, x):
        sharpeness = K.pow(self.sharpness, 2)
        x = x - self.centers
        x = K.pow(x, 2)
        y = K.stack([x[:, :, :, i] * sharpeness[:, :, i] for i in range(x.shape[-1])], axis=-1)
        y = K.sum(y, axis=2)
        y = K.exp(-1 * y)
        return y

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], 1)


# 用于形状的PD Layer的定义
class PD_Layer_Shape(Layer):
    def __init__(self, **kwargs):
        super(PD_Layer_Shape, self).__init__(**kwargs)

    def build(self, input_shape):
        self.centers = self.add_weight(name='pd_center',
                                       shape=(input_shape[1], input_shape[2], input_shape[3]),
                                       initializer=keras.initializers.RandomNormal(),
                                       trainable=True)
        self.sharpness = self.add_weight(name='pd_sharpness',
                                         shape=(input_shape[1], input_shape[2], input_shape[3]),
                                         initializer=keras.initializers.Constant(3),
                                         trainable=True)
        super(PD_Layer_Shape, self).build(input_shape)

    def call(self, x):
        sharpeness = K.pow(self.sharpness, 2)
        x = x - self.centers
        x = K.pow(x, 2)
        y = K.stack([x[:, :, :, i] * sharpeness[:, :, i] for i in range(x.shape[-1])], axis=-1)
        y = K.sum(y, axis=2)
        y = K.exp(-1 * y)
        return y

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], 3)


# 形状PD特征进一步提取网络部分
def stage_1(pd, direction, N, stage_1_kr=0.1, stage_1_drop_out_1=0.5, stage_1_neuron_num_1=512):
    x = PD_Layer_Shape()(pd)
    x = Conv1D(filters=16, kernel_size=1, strides=1, padding='same', activation='relu',
               kernel_regularizer=regularizers.l2(stage_1_kr), name='stage_1_conv_1_' + str(direction))(x)

    x = Conv1D(filters=16, kernel_size=1, strides=1, padding='same', activation='relu',
               kernel_regularizer=regularizers.l2(stage_1_kr), name='stage_1_conv_2_' + str(direction))(x)

    x = Reshape((N, 16, 1))(x)
    x = MaxPooling2D(pool_size=(1, 16))(x)
    x = Reshape((N,))(x)
    x = Dropout(stage_1_drop_out_1)(x)
    x = Dense(stage_1_neuron_num_1)(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    model = Model(inputs=pd, outputs=x)
    return model


# 纹理PD和叶脉PD特征进一步提取部分的网络定义
def stage_2(pd, direction, N, stage_1_kr=0.1, stage_1_drop_out_1=0.5, stage_1_neuron_num_1=512):
    x = PD_Layer()(pd)
    x = Conv1D(filters=16, kernel_size=1, strides=1, padding='same', activation='relu',
               kernel_regularizer=regularizers.l2(stage_1_kr), name='stage_1_conv_1_' + str(direction))(x)

    x = Conv1D(filters=16, kernel_size=1, strides=1, padding='same', activation='relu',
               kernel_regularizer=regularizers.l2(stage_1_kr), name='stage_1_conv_2_' + str(direction))(x)
    x = Reshape((N, 16, 1))(x)
    x = MaxPooling2D(pool_size=(1, 16))(x)
    x = Reshape((N,))(x)
    x = Dropout(stage_1_drop_out_1)(x)
    x = Dense(stage_1_neuron_num_1)(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    model = Model(inputs=pd, outputs=x)
    return model


# 将Slayer提取的PD的特征，于Xception 提取到的图像的特征在FC层处拼接在一起
def Combined_Model(parallels, config):
    # with tf.device('/cpu:0'):
    inputs = [Input(shape=(config['N'][i], 2, 3)) for i in range(config['shape_views'])]
    inputs.extend(Input(shape=(config['N'][j], 2, 1)) for j in range(config['shape_views'], config['views']))
    input_tensor = Input(shape=(config['image_size'][0], config['image_size'][1], 3))
    inputs.append(input_tensor)
    stage_1_outputs = []
    for i in range(config['shape_views']):
        model = stage_1(inputs[i], int(i), int(config['N'][i]), config['stage1_kr'], config['stage1_dropout'],
                        int(config['stage1_neuron_num']))
        stage_1_outputs.append(model.output)

    for j in range(30, 34):
        model2 = stage_2(inputs[j], int(j), int(config['N'][j]), config['stage1_kr'], config['stage1_dropout'],
                         int(config['stage1_neuron_num']))
        stage_1_outputs.append(model2.output)

    model_img = Xception(include_top=False,
                         weights='imagenet',
                         input_tensor=input_tensor,
                         input_shape=(config['image_size'][0], config['image_size'][1], 3),
                         pooling='max')

    stage_1_outputs.append(model_img.output)

    x = Concatenate(axis=1, name='concat')(stage_1_outputs)
    x = Dropout(0.5)(x)
    x = Dense(2048)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(config['classes'], activation='softmax')(x)
    fused_model = Model(inputs=inputs, outputs=x)
    rmsprop = RMSprop(lr=0.001)

    if (parallels > 1):
        parallel_model = multi_gpu_model(fused_model, gpus=parallels)
        parallel_model.compile(optimizer=rmsprop,
                               loss='categorical_crossentropy',
                               metrics=['categorical_accuracy'])
        return parallel_model
    else:
        fused_model.compile(optimizer=rmsprop,
                            loss='categorical_crossentropy',
                            metrics=['categorical_accuracy'])
        return fused_model


def Xception_Model(parallels, config):
    # with tf.device('/cpu:0'):
    input_tensor = Input(shape=(config['image_size'][0], config['image_size'][1], 3))
    base_model = Xception(include_top=False,
                          weights='imagenet',
                          input_tensor=input_tensor,
                          input_shape=(config['image_size'][0], config['image_size'][1], 3),
                          pooling='max')
    x = base_model.output
    print(x.shape)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.5)(x)
    y = Dense(config['classes'], activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=y)
    if parallels > 1:
        parallel_model = multi_gpu_model(model, gpus=parallels)
        parallel_model.compile(optimizer='sgd',
                               loss='categorical_crossentropy',
                               metrics=['categorical_accuracy'])
        return parallel_model
    else:
        model.compile(optimizer='sgd',
                               loss='categorical_crossentropy',
                               metrics=['categorical_accuracy'])
    return model