from sklearn import preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from mapgi import mapgi
from constants import *


def return_my_datagen(custom_preprocessing=True, mode="training"):

    if custom_preprocessing:
        preprocessing_function = mapgi
    else:
        preprocessing_function = None
          
    if mode == "training":    
        datagen=ImageDataGenerator(
            rescale=1/255,
            rotation_range=40,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest',
            samplewise_std_normalization=SAMPLEWISE_NORM,
            preprocessing_function=preprocessing_function
            )

    # TODO disable samplewise and try again for final paper
    # Monday evening put efficientnet without mapgi or samplewise norm for training, 
    # in same sh file train underwater mine without MAPGI, with efficientnet
    
    else:
        datagen = ImageDataGenerator(
            rescale=1/255,
            samplewise_std_normalization=SAMPLEWISE_NORM,
            preprocessing_function=preprocessing_function
            )
       
    return datagen