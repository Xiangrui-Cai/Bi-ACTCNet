import math
import numpy as np
import tensorflow as tf
from keras.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape, multiply
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, AveragePooling2D, MaxPooling2D
from tensorflow.keras.layers import Conv1D, Conv2D, SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization, LayerNormalization, Flatten
from tensorflow.keras.layers import Add, Concatenate, Lambda, Input, Permute
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Permute, Lambda, Dense, Activation, Flatten, MaxPooling2D
from tensorflow.keras.constraints import max_norm
import tensorflow.keras.backend as K
from attention_models import MutualCrossAttention


def etca_attention(input_feature, gama=2, b=3):
    in_channel = input_feature.shape[2]

    kernel_size = int(abs((math.log(in_channel, 2) + b) / gama))
    if kernel_size % 2:
        kernel_size = kernel_size
    else:
        kernel_size = kernel_size + 1
    input_feature = tf.transpose(input_feature, perm=[0, 1, 3, 2])

    # [h,w,c]==>[None,c]
    x = GlobalAveragePooling2D()(input_feature)
    x = Reshape(target_shape=(in_channel, 1))(x)

    # [c,1]==>[c,1]
    block = Conv1D(filters=1, kernel_size=kernel_size, padding='same', use_bias=False)(x)

    x = Conv1D(filters=1, kernel_size=kernel_size, dilation_rate=2, activation='linear',
                   padding='causal', kernel_initializer='he_uniform')(block)
    x = Activation(activation='relu')(x)
    x = Dropout(0.3)(x)

    x = Conv1D(filters=1, kernel_size=kernel_size*2, dilation_rate=2, activation='linear',
               padding='causal', kernel_initializer='he_uniform')(x)
    x = Activation(activation='relu')(x)
    x = Dropout(0.3)(x)
    added = Add()([x, block])
    out = tf.nn.sigmoid(added)
    # [c,1]==>[1,1,c]
    x = Reshape((1, 1, in_channel))(out)
    outputs = multiply([input_feature, x])
    outputs = tf.transpose(outputs, perm=[0, 1, 3, 2])

    return outputs


def TCN_block(input_layer, input_dimension=32, depth=2, kernel_size=4, filters=32, dropout=0.1, activation='elu'):
    block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=1, activation='linear',
                   padding='causal', kernel_initializer='he_uniform')(input_layer)
    block = BatchNormalization()(block)
    block = Activation(activation)(block)
    block = Dropout(dropout)(block)
    block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=1, activation='linear',
                   padding='causal', kernel_initializer='he_uniform')(block)
    block = BatchNormalization()(block)
    block = Activation(activation)(block)
    block = Dropout(dropout)(block)
    if (input_dimension != filters):
        conv = Conv1D(filters, kernel_size=1, padding='same')(input_layer)
        added = Add()([block, conv])
    else:
        added = Add()([block, input_layer])
    out = Activation(activation)(added)

    for i in range(depth - 1):
        block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=2 ** (i + 1), activation='linear',
                       padding='causal', kernel_initializer='he_uniform')(out)
        block = BatchNormalization()(block)
        block = Activation(activation)(block)
        block = Dropout(dropout)(block)
        block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=2 ** (i + 1), activation='linear',
                       padding='causal', kernel_initializer='he_uniform')(block)
        block = BatchNormalization()(block)
        block = Activation(activation)(block)
        block = Dropout(dropout)(block)
        added = Add()([block, out])
        out = Activation(activation)(added)
    return out

def EEGNet_ECA(input_layer, F1=8, kernLength=64, D=2, Chans=64, dropout=0.25):
    F2 = F1 * D
    block1 = Conv2D(F1, (kernLength, 1), padding='same', data_format='channels_last', use_bias=False)(input_layer)
    block1 = BatchNormalization(axis=-1)(block1)
    block1 = etca_attention(block1)
    transposed_tensor = tf.transpose(block1, perm=[0, 2, 1, 3])
    reshaped_tensor = tf.reshape(transposed_tensor, (-1, 64, 640 * 8))
    reshaped_tensor = tf.reduce_sum(reshaped_tensor, axis=2)  # 对最后一个维度求和 -> (None, 59)
    block2 = DepthwiseConv2D((1, Chans), use_bias=False,
                             depth_multiplier=D,
                             data_format='channels_last',
                             depthwise_constraint=max_norm(1.))(block1)
    block2 = BatchNormalization(axis=-1)(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((8, 1), data_format='channels_last')(block2)
    block2 = Dropout(dropout)(block2)
    block3 = SeparableConv2D(F2, (16, 1),
                             data_format='channels_last',
                             use_bias=False, padding='same')(block2)
    block3 = BatchNormalization(axis=-1)(block3)
    block3 = Activation('elu')(block3)
    block3 = AveragePooling2D((4, 1), data_format='channels_last')(block3)
    block3 = Dropout(dropout)(block3)
    return block3, reshaped_tensor

def Bi_ACTCNET(n_classes, Chans=64, Samples=1500, layers=2, kernel_s=4, filt=16, dropout=0.2, activation='elu', F1=8, D=2,
             kernLength=32, dropout_eeg=0.2):

    input1 = Input(shape=(1, Chans, Samples))
    input2 = Permute((3, 2, 1))(input1)
    regRate = .25
    numFilters = F1
    F2 = numFilters * D

    # branch 1
    EEGNet_sep = EEGNet_ECA(input_layer=input2, F1=F1, kernLength=kernLength, D=D, Chans=Chans, dropout=dropout_eeg)
    block2 = Lambda(lambda x: x[0][:, :, -1, :])(EEGNet_sep)

    # branch 2
    input2 = Permute((3, 2, 1))(input1)
    input3 = input2[:, ::-1, :]
    EEGNet_sep = EEGNet_ECA(input_layer=input3, F1=F1, kernLength=kernLength, D=D, Chans=Chans, dropout=dropout_eeg)
    block3 = Lambda(lambda x: x[0][:, :, -1, :])(EEGNet_sep)

    attention_output = MutualCrossAttention(dropout_rate=0.3)(block2, block3)

    outs = TCN_block(input_layer=attention_output, input_dimension=F2, depth=layers, kernel_size=kernel_s, filters=filt,
                     dropout=dropout, activation=activation)
    out2 = Lambda(lambda x: x[:, -1, :])(outs)

    flatten1 = Flatten(name='flatten1')(out2)
    dense = Dense(n_classes, name='dense', kernel_constraint=max_norm(regRate))(flatten1)
    softmax = Activation('softmax', name='softmax')(dense)

    return Model(inputs=input1, outputs=softmax)




