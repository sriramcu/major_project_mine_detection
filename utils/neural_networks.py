from keras.layers import Input, merge, Dropout, Dense, Flatten, Activation
from keras.layers.convolutional import MaxPooling2D, Conv2D, AveragePooling2D

from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling2D
from keras.models import Model
from keras import backend as K
from keras.layers import concatenate
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import regularizers
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.nasnet import NASNetLarge
import efficientnet.keras as efn

from constants import *



def conv_block(x, nb_filter, nb_row, nb_col, padding='same', subsample=(1, 1), bias=False):
    if K.image_data_format() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    x = Conv2D(nb_filter, (nb_row, nb_col), strides=subsample, padding=padding, use_bias=bias)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    return x


def inception_stem(input):

    channel_axis = -1

    # Shape 299 x 299 x 3 
    x = conv_block(input, 32, 3, 3, subsample=(2, 2), padding='valid')
    x = conv_block(x, 32, 3, 3, padding='valid')
    x = conv_block(x, 64, 3, 3)

    x1 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(x)
    x2 = conv_block(x, 96, 3, 3, subsample=(2, 2), padding='valid')
    x = concatenate([x1, x2], axis=channel_axis)

    x1 = conv_block(x, 64, 1, 1)
    x1 = conv_block(x1, 96, 3, 3, padding='valid')

    x2 = conv_block(x, 64, 1, 1)
    x2 = conv_block(x2, 64, 1, 7)
    x2 = conv_block(x2, 64, 7, 1)
    x2 = conv_block(x2, 96, 3, 3, padding='valid')
    x = concatenate([x1, x2], axis=channel_axis)

    x1 = conv_block(x, 192, 3, 3, subsample=(2, 2), padding='valid')
    x2 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(x)
    x = concatenate([x1, x2], axis=channel_axis)

    return x


def inception_A(input):
    channel_axis = -1

    a1 = conv_block(input, 96, 1, 1)

    a2 = conv_block(input, 64, 1, 1)
    a2 = conv_block(a2, 96, 3, 3)

    a3 = conv_block(input, 64, 1, 1)
    a3 = conv_block(a3, 96, 3, 3)
    a3 = conv_block(a3, 96, 3, 3)

    a4 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input)
    a4 = conv_block(a4, 96, 1, 1)

    m = concatenate([a1, a2, a3, a4], axis=channel_axis)

    return m


def inception_B(input):
    channel_axis = -1

    b1 = conv_block(input, 384, 1, 1)

    b2 = conv_block(input, 192, 1, 1)
    b2 = conv_block(b2, 224, 1, 7)
    b2 = conv_block(b2, 256, 7, 1)

    b3 = conv_block(input, 192, 1, 1)
    b3 = conv_block(b3, 192, 7, 1)
    b3 = conv_block(b3, 224, 1, 7)
    b3 = conv_block(b3, 224, 7, 1)
    b3 = conv_block(b3, 256, 1, 7)

    b4 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input)
    b4 = conv_block(b4, 128, 1, 1)

    m = concatenate([b1, b2, b3, b4], axis=channel_axis)

    return m


def inception_C(input):
    channel_axis = -1

    c1 = conv_block(input, 256, 1, 1)

    c2 = conv_block(input, 384, 1, 1)
    c2_1 = conv_block(c2, 256, 1, 3)
    c2_2 = conv_block(c2, 256, 3, 1)

    c2 = concatenate([c2_1, c2_2], axis=channel_axis)


    c3 = conv_block(input, 384, 1, 1)
    c3 = conv_block(c3, 448, 3, 1)
    c3 = conv_block(c3, 512, 1, 3)
    c3_1 = conv_block(c3, 256, 1, 3)
    c3_2 = conv_block(c3, 256, 3, 1)

    c3 = concatenate([c3_1, c3_2], axis=channel_axis)

    c4 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input)
    c4 = conv_block(c4, 256, 1, 1)

    m = concatenate([c1, c2, c3, c4], axis=channel_axis)

    return m


def reduction_A(input):
    channel_axis = -1

    r1 = conv_block(input, 384, 3, 3, subsample=(2, 2), padding='valid')

    r2 = conv_block(input, 192, 1, 1)
    r2 = conv_block(r2, 224, 3, 3)
    r2 = conv_block(r2, 256, 3, 3, subsample=(2, 2), padding='valid')

    r3 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(input)

    m = concatenate([r1, r2, r3], axis=channel_axis)

    return m


def reduction_B(input):
    channel_axis = -1

    r1 = conv_block(input, 192, 1, 1)
    r1 = conv_block(r1, 192, 3, 3, subsample=(2, 2), padding='valid')

    r2 = conv_block(input, 256, 1, 1)
    r2 = conv_block(r2, 256, 1, 7)
    r2 = conv_block(r2, 320, 7, 1)
    r2 = conv_block(r2, 320, 3, 3, subsample=(2, 2), padding='valid')

    r3 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(input)

    m = concatenate([r1, r2, r3], axis=channel_axis)

    return m


def create_inception_v4(nb_classes, resume_previous_training, checkpoint_path):

    print("Custom Inceptionv4 model being used...")
    init = Input((299,299, 3))

    x = inception_stem(init)

    # 4 x Inception A
    for i in range(4):
        x = inception_A(x)

    # Reduction A
    x = reduction_A(x)

    # 7 x Inception B
    for i in range(7):
        x = inception_B(x)

    # Reduction B
    x = reduction_B(x)

    # 3 x Inception C
    for i in range(3):
        x = inception_C(x)

    # Average Pooling
    x = AveragePooling2D((8, 8))(x)

    # Dropout - Use 0.2, as mentioned in official paper. 
    x = Dropout(0.8)(x)
    x = Flatten()(x)

    # Output
    out = Dense(nb_classes, activation='softmax')(x)

    model = Model(init, out, name='Inception-v4')

    if resume_previous_training:
        weights = checkpoint_path
        model.load_weights(weights, by_name=True)
        print("Previously trained model weights loaded.")
 
    return model



def create_pretrained_inceptionv3(nb_classes, resume_previous_training, checkpoint_path):
    
    print("Pretrained Inceptionv3 being used...")
    
    
    
    base_model = InceptionV3(input_shape = (299, 299, 3), include_top = False, weights = 'imagenet')
    
    l = len(base_model.layers)
    for i, layer in enumerate(base_model.layers):
        layer.trainable = False
        rev_index = l-i-1
        if rev_index < NUM_TRAINABLE_LAYERS:
            layer.trainable = True
        
    
    x = GlobalAveragePooling2D()(base_model.output) 
    # x = base_model.output
    # x = Flatten()(x)
    x = Dense(30, activation="relu", 
              kernel_regularizer=regularizers.l2(L2_REG),
              activity_regularizer=regularizers.l2(L2_REG),
              bias_regularizer=regularizers.l2(L2_REG)              
              )(x)
    
    x = Dropout(MY_DROPOUT)(x)
    
    x_image_out = Dense(nb_classes, activation="softmax",
                        kernel_regularizer=regularizers.l2(L2_REG),
                        activity_regularizer=regularizers.l2(L2_REG),
                        bias_regularizer=regularizers.l2(L2_REG)                        
                        )(x)

    
    
    model = Model(base_model.input, x_image_out)
    
    if resume_previous_training:
        # weights = checkpoint_path
        # model.load_weights(weights, by_name=True)
        model = load_model(checkpoint_path)
        print("Previously trained model weights loaded.")


    # print(model.summary())
    
    return model
    
    
    
def create_pretrained_efficientnet(nb_classes, resume_previous_training, checkpoint_path):
    
    print("Pretrained EfficientNet B7 being used...")   
    
    
    base_model = efn.EfficientNetB7(input_shape = (299, 299, 3), 
                                    include_top = False, weights = 'imagenet')
    l = len(base_model.layers)
    for i, layer in enumerate(base_model.layers):
        layer.trainable = False
        rev_index = l-i-1
        if rev_index < NUM_TRAINABLE_LAYERS:
            layer.trainable = True
        
    
    x = GlobalAveragePooling2D()(base_model.output) 
    # x = base_model.output
    # x = Flatten()(x)
    x = Dense(30, activation="relu", 
              kernel_regularizer=regularizers.l2(L2_REG),
              activity_regularizer=regularizers.l2(L2_REG),
              bias_regularizer=regularizers.l2(L2_REG)              
              )(x)
    
    x = Dropout(MY_DROPOUT)(x)
    
    x_image_out = Dense(nb_classes, activation="softmax",
                        kernel_regularizer=regularizers.l2(L2_REG),
                        activity_regularizer=regularizers.l2(L2_REG),
                        bias_regularizer=regularizers.l2(L2_REG)                        
                        )(x)

    
    
    model = Model(base_model.input, x_image_out)
    
    if resume_previous_training:
        # weights = checkpoint_path
        # model.load_weights(weights, by_name=True)
        model = load_model(checkpoint_path)
        print("Previously trained model weights loaded.")


    # print(model.summary())
    
    return model



def create_pretrained_nasnet(nb_classes, resume_previous_training, checkpoint_path):
    
    print("Pretrained NASNet being used...")       
    
    base_model = NASNetLarge(input_shape = (331, 331, 3), include_top = False, weights = 'imagenet')
    
    l = len(base_model.layers)
    for i, layer in enumerate(base_model.layers):
        layer.trainable = False
        rev_index = l-i-1
        if rev_index < NUM_TRAINABLE_LAYERS:
            layer.trainable = True
        
    
    x = GlobalAveragePooling2D()(base_model.output) 
    # x = base_model.output
    # x = Flatten()(x)
    x = Dense(30, activation="relu", 
              kernel_regularizer=regularizers.l2(L2_REG),
              activity_regularizer=regularizers.l2(L2_REG),
              bias_regularizer=regularizers.l2(L2_REG)              
              )(x)
    
    x = Dropout(MY_DROPOUT)(x)
    
    x_image_out = Dense(nb_classes, activation="softmax",
                        kernel_regularizer=regularizers.l2(L2_REG),
                        activity_regularizer=regularizers.l2(L2_REG),
                        bias_regularizer=regularizers.l2(L2_REG)                        
                        )(x)

    
    
    model = Model(base_model.input, x_image_out)
    
    if resume_previous_training:
        # weights = checkpoint_path
        # model.load_weights(weights, by_name=True)
        model = load_model(checkpoint_path)
        print("Previously trained model weights loaded.")


    # print(model.summary())
    
    return model