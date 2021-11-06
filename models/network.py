import numpy as np
import tensorflow as tf
from tensorflow.keras.activations import sigmoid, softmax
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D, MaxPooling2D, Dense, GlobalMaxPooling2D, Reshape, multiply, Add
from tensorflow.keras.layers import Input, Dropout, ZeroPadding2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model

def channel_attention(input_feature, ratio=8):
    '''
    Apply channel attention on input feature.
    '''
    channel_axis = -1
    channel = input_feature.shape[channel_axis]
    
    shared_layer_one = Dense(channel//ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    
    avg_pool = GlobalAveragePooling2D()(input_feature)    
    avg_pool = Reshape((1,1,channel))(avg_pool)
    assert avg_pool.shape[1:] == (1,1,channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool.shape[1:] == (1,1,channel//ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool.shape[1:] == (1,1,channel)
    
    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1,1,channel))(max_pool)
    assert max_pool.shape[1:] == (1,1,channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool.shape[1:] == (1,1,channel//ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool.shape[1:] == (1,1,channel)
    
    cbam_feature = Add()([avg_pool,max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)
    
    return multiply([input_feature, cbam_feature])

# network definition
def resnet_layer(inputs,num_filters=16,kernel_size=3,strides=1,learn_bn = True,wd=1e-4,use_relu=True):
    '''
    Resnet layer function
    '''
    x = inputs
    x = Conv2D(num_filters,kernel_size=kernel_size,strides=strides,padding='valid',kernel_initializer='he_normal',
                  kernel_regularizer=l2(wd),use_bias=False)(x)
    x = BatchNormalization(center=learn_bn, scale=learn_bn)(x)
    if use_relu:
        x = Activation('relu')(x)
    return x

def conv_layer1(inputs, num_channels=6, num_filters=14, learn_bn=True, wd=1e-4, use_relu=True):
    '''
    Stack of convolutional layers
    ''' 
    kernel_size1 = [5, 5]
    kernel_size2 = [3, 3]
    strides1 = [2, 2]
    strides2 = [1, 1]
    x = inputs
    x = BatchNormalization(center=learn_bn, scale=learn_bn)(x)
    x = ZeroPadding2D(padding=(2, 2), data_format='channels_last')(x)
    x = Conv2D(num_filters*num_channels, kernel_size=kernel_size1, strides=strides1,
               padding='valid', kernel_initializer='he_normal',
               kernel_regularizer=l2(wd), use_bias=False)(x)
    x = BatchNormalization(center=learn_bn, scale=learn_bn)(x)
    if use_relu:
        x = Activation('relu')(x)

    x = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(x)
    x = Conv2D(num_filters * num_channels, kernel_size=kernel_size2, strides=strides2,
               padding='valid', kernel_initializer='he_normal',
               kernel_regularizer=l2(wd), use_bias=False)(x)
    x = BatchNormalization(center=learn_bn, scale=learn_bn)(x)
    if use_relu:
        x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2),padding='valid')(x)
    return x

def conv_layer2(inputs, num_channels=6, num_filters=28, learn_bn=True, wd=1e-4, use_relu=True):
    '''
    Stack of convolutional layers
    '''
    kernel_size = [3, 3]
    strides = [1, 1]
    x = inputs
    x = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(x)
    x = Conv2D(num_filters*num_channels, kernel_size=kernel_size, strides=strides,
               padding='valid', kernel_initializer='he_normal',
               kernel_regularizer=l2(wd), use_bias=False)(x)
    x = BatchNormalization(center=learn_bn, scale=learn_bn)(x)
    if use_relu:
        x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(x)
    x = Conv2D(num_filters * num_channels, kernel_size=kernel_size, strides=strides,
               padding='valid', kernel_initializer='he_normal',
               kernel_regularizer=l2(wd), use_bias=False)(x)
    x = BatchNormalization(center=learn_bn, scale=learn_bn)(x)
    if use_relu:
        x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='valid')(x)
    return x

def conv_layer3(inputs, num_channels=6, num_filters=56, learn_bn=True, wd=1e-4, use_relu=True):
    '''
    Stack of convolutional layers
    '''
    kernel_size = [3, 3]
    strides = [1, 1]
    x = inputs
    # 1
    x = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(x)
    x = Conv2D(num_filters*num_channels, kernel_size=kernel_size, strides=strides,
               padding='valid', kernel_initializer='he_normal',
               kernel_regularizer=l2(wd), use_bias=False)(x)
    x = BatchNormalization(center=learn_bn, scale=learn_bn)(x)
    if use_relu:
        x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    #2
    x = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(x)
    x = Conv2D(num_filters*num_channels, kernel_size=kernel_size, strides=strides,
               padding='valid', kernel_initializer='he_normal',
               kernel_regularizer=l2(wd), use_bias=False)(x)
    x = BatchNormalization(center=learn_bn, scale=learn_bn)(x)
    if use_relu:
        x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    # 3
    x = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(x)
    x = Conv2D(num_filters*num_channels, kernel_size=kernel_size, strides=strides,
               padding='valid', kernel_initializer='he_normal',
               kernel_regularizer=l2(wd), use_bias=False)(x)
    x = BatchNormalization(center=learn_bn, scale=learn_bn)(x)
    if use_relu:
        x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    #4
    x = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(x)
    x = Conv2D(num_filters*num_channels, kernel_size=kernel_size, strides=strides,
               padding='valid', kernel_initializer='he_normal',
               kernel_regularizer=l2(wd), use_bias=False)(x)
    x = BatchNormalization(center=learn_bn, scale=learn_bn)(x)
    if use_relu:
        x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='valid')(x)
    return x

def model_fcnn(num_classes, input_shape=[None, 128, 60], num_filters=[24, 48, 96], wd=1e-3):
    '''
    Creates fully connected convolutional neural network model.
    :param int num_classes: number of classes
    :param list input_shape: shape of input images
    :param list num_filters: number of convolutional filters for each convolutional path
    :param float wd: factor for kernel regularizor
    :return tf.keras.model model: keras model.
    '''
    
    inputs = Input(shape=input_shape)
    ConvPath1 = conv_layer1(inputs=inputs,
                            num_channels=input_shape[-1],
                            num_filters=num_filters[0],
                            learn_bn=True,
                            wd=wd,
                            use_relu=True)
    ConvPath2 = conv_layer2(inputs=ConvPath1,
                            num_channels=input_shape[-1],
                            num_filters=num_filters[1],
                            learn_bn=True,
                            wd=wd,
                            use_relu=True)
    ConvPath3 = conv_layer3(inputs=ConvPath2,
                            num_channels=input_shape[-1],
                            num_filters=num_filters[2],
                            learn_bn=True,
                            wd=wd,
                            use_relu=True)

    # output layers after last sum
    OutputPath = resnet_layer(inputs=ConvPath3,
                              num_filters=num_classes,
                              strides=1,
                              kernel_size=1,
                              learn_bn=False,
                              wd=wd,
                              use_relu=True)

    OutputPath = BatchNormalization(center=False, scale=False)(OutputPath)
    OutputPath = channel_attention(OutputPath, ratio=2)

    OutputPath = GlobalAveragePooling2D()(OutputPath)
    OutputPath = Activation('softmax')(OutputPath)
    # Instantiate model.
    model = Model(inputs=inputs, outputs=OutputPath)
    return model

def create_model(model_config, data_config, train_config):
    '''
    Creates and compiles neural network model.
    :param class model_config: config file of the model
    :param class data_config: config file for data 
    :param class train_config: config file for training
    :return tf.keras.model model: tensorflow neural network model
    '''
    model = model_fcnn(model_config.num_classes, 
                       input_shape=[data_config.num_freq_bin, None, 3*data_config.num_audio_channel], 
                       num_filters=model_config.num_filters, 
                       wd=model_config.wd) 
    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.SGD(lr=train_config.max_lr, decay=train_config.decay, momentum=train_config.momentum, nesterov=False),
                  #optimizer =tf.keras.optimizers.Adam(learning_rate=train_config.max_lr, decay=train_config.decay),
                  metrics=['accuracy'])
    return model
