""" 
TODO list of todos for the entire project


Distant todos:
Write READMEs for all to avoid confusion
Docstring for all programs and functions/ remove unnecessary comments/todos, refactor vars, fns
From git commit history restore csv and svm for archived depthai

Once GPU is free:
Test sixclass video evaluator
Combine video evaluator output with that of annotator
Re evaluate all sixclass metrics to put into ppt and then research paper
Try grip assessment train and test for 1 epoch. If OK, run the shell script
Ask for more videos for further validation
"""


import argparse
import datetime
from pathlib import Path
import pickle
import warnings
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from sklearn.utils import shuffle
import os
import subprocess
from time import time
import math
from collections import Counter

from utils.constants import *
from utils.my_datagen import return_my_datagen
from utils.neural_networks import *

import numpy as np
import faulthandler
import shelve

faulthandler.enable()



def round_nearest(x, a):
    max_frac_digits = 100
    for i in range(max_frac_digits):
        if round(a, -int(math.floor(math.log10(a))) + i) == a:
            frac_digits = -int(math.floor(math.log10(a))) + i
            break
    return round(round(x / a) * a, frac_digits)



# TODO pass args separately
def train(args,resume_previous_training,checkpoint_filepath):

    num_classes = int(args["num_classes"])
    
    custom_preprocessing = int(args["custom_preprocess"])
    
    train_dir = str(args['train_dir'])
    val_dir = str(args['val_dir'])
    
    batch_size = BATCH_SIZE
    
    num_epochs = int(args['epochs'])
    
    network_name = args["network"]
    
    numerical_data_dir = os.path.join(Path(__file__).resolve().parent, 
                                      "saved_numerical_data", "")

    checkpoint_dir = os.path.join(numerical_data_dir, "checkpoints")
    
    checkpoint_filepath = os.path.join(checkpoint_dir, f"{network_name}_{num_epochs}epochs.h5")

    parameters_filepath = os.path.join(numerical_data_dir, "parameters.txt")

    metrics_pickle_filename = f"train_metrics_{num_epochs}epochs_{network_name}.pickle"
    metrics_pickle_filepath = os.path.join(numerical_data_dir, "training_metrics_pickle_files", 
                                           metrics_pickle_filename)

    

    if not os.path.isdir(train_dir):
        raise ValueError("Training directory not found!")
    
    if not os.path.isdir(val_dir):
        raise ValueError("Validation directory not found!")
    
    
    train_datagen = return_my_datagen(custom_preprocessing=custom_preprocessing, mode="training")
        
    val_datagen = return_my_datagen(custom_preprocessing=custom_preprocessing, mode="prediction")    


    train_generator = train_datagen.flow_from_directory(
                                                  train_dir,
                                                  target_size=(299,299),
                                                  class_mode="categorical", 
                                                  batch_size=batch_size
                                                  )
    
    
    val_gen = val_datagen.flow_from_directory(
                                          val_dir,
                                          target_size=(299,299),
                                          class_mode="categorical", 
                                          batch_size=batch_size                                          
                                          )    
    
    
    
    
    if network_name == "inceptionv4":
        model = create_inception_v4(
                                    num_classes,
                                    resume_previous_training,
                                    checkpoint_filepath
                                    )        

    elif network_name == "inceptionv3":
        model = create_pretrained_inceptionv3(
                                              num_classes,
                                              resume_previous_training,
                                              checkpoint_filepath
                                              )

        
    elif network_name == "efficientnet":
        model = create_pretrained_efficientnet(
                                               num_classes,
                                               resume_previous_training,
                                               checkpoint_filepath
                                               )   
    
    elif network_name == "nasnet":
        model = create_pretrained_nasnet(
                                        num_classes,
                                        resume_previous_training,
                                        checkpoint_filepath
                                        )
    
    else:
        raise ValueError("Please check your network name and try again")
    
    
    def lr_scheduler(epoch, lr):
        if epoch > (num_epochs/3):
            lr = LEARNING_RATE/10
        
        # print("Learning Rate = ", lr)
        return lr
    
    
    
    
    
    mc = keras.callbacks.ModelCheckpoint(
                                        checkpoint_filepath,
                                        save_weights_only=False,  
                                        monitor='val_accuracy',                                        
                                        save_best_only=True
                                        )
    
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor=f"val_loss",
    min_delta=0,
    patience=100,
    verbose=1,
    mode="auto",
    baseline=None,
    restore_best_weights=True,
    )
    
    lr_callback = keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)
    
    


    # Earlier option to use RMSProp has been removed, as results were unsatisfactory

    optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)
    
    model.compile(loss='categorical_crossentropy', 
                  optimizer= optimizer,
                  metrics=["accuracy"])

    counter = Counter(train_generator.classes)                          
    max_val = float(max(counter.values()))       
    class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()} 

    history = model.fit(train_generator,
                     epochs=num_epochs,
                     verbose=True,
                     validation_data=val_gen,
                     callbacks=[mc, early_stopping_callback, lr_callback], 
                     class_weight=class_weights)
    
    
    
    f = open(parameters_filepath, 'a')
    params = [num_epochs, MY_DROPOUT, LEARNING_RATE, L2_REG, custom_preprocessing, network_name]
    f.write(f"{params}\n")
    f.close()
    
    
    f = open(metrics_pickle_filepath, "wb")
    # Pickle dumps are FIFO
    pickle.dump(history.history, f)
    params.append(checkpoint_filepath)
    pickle.dump(params, f) 
    f.close()
        

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
 
    
    # Default arguments
    parser.add_argument("-network", default="inceptionv3", type=str, help="Name of the network")
        
    parser.add_argument("-train", "--train_dir", type=str, default="dataset/train", 
                        help="train directory")
    
    parser.add_argument("-val", "--val_dir", type=str, default="dataset/val", help="val directory")
    
    parser.add_argument("-c", "--checkpoint", type=str, default="no", 
                        help="Allow for pausing/resuming training")
    
    
    parser.add_argument("-log", "--log_dir", type=str, default="logs", help=" ")
       
   
    requiredNamed = parser.add_argument_group('required named arguments')
    # Required arguments
    requiredNamed.add_argument("-classes", "--num_classes", type=int, required=True, help=" ")
    
    requiredNamed.add_argument("-custom_preprocess", type=int, required=True, help=" ")
    
    requiredNamed.add_argument("-epochs", type=int, required=True, help=" ")
    
    args = vars(parser.parse_args())

    

    # Check whether continue training from pretrained model or not
    checkpoint_path = ""
    if str(args['checkpoint']) != 'no': # Default value is "no"
        checkpoint_path = str(args['checkpoint'])
        resume_previous_training = True
    else:
        resume_previous_training = False
    
    train(args,resume_previous_training,checkpoint_path)


if __name__ == "__main__":
    main()