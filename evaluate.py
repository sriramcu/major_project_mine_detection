import pickle
from sklearn.metrics import confusion_matrix
from utils.test_images_from_directory import test_images_from_directory
import os
import sys
import shelve
from pathlib import Path
from keras.models import load_model
import argparse
import csv
import tensorflow as tf
from sklearn.metrics import classification_report
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import efficientnet.keras as efn

from utils.compute_conf_mat_scores import *


def evaluate_model(args):
    root_val_dir = os.path.abspath(args["root_val_dir"])
    evaluate_model = int(args["evaluate_model"])
    verbose = int(args["verbose"])

    results = {}
    classes = sorted(os.listdir(root_val_dir))

    metrics_pickle_filepath = args["train_pickle_file"]

    f = open(metrics_pickle_filepath, 'rb')
    
    train_metrics = pickle.load(f)
    params = pickle.load(f)
    
    f.close()
    
    if verbose:
        print(params)
        
    custom_preprocessing = params[-3]

    checkpoints_filepath = params[-1]
    
    cp_basename = os.path.basename(checkpoints_filepath).split(".")[0]
        
    numerical_data_dir = os.path.join(Path(__file__).resolve().parent, 
                                      "saved_numerical_data")
    
    conf_pickle = os.path.join(numerical_data_dir, "confusion_matrix_pickle_files", 
                               f"{cp_basename}.pkl")
    
    conf_csv = os.path.join(numerical_data_dir, "confusion_matrix_tables", f"{cp_basename}.csv")
    
    
    if verbose:
        print(checkpoints_filepath)
    
    model = load_model(checkpoints_filepath)
    
    for actual_class_name in classes:
        test_dir = os.path.join(root_val_dir, actual_class_name)        
                
        results_dict = test_images_from_directory(model, 
                                                 test_dir=test_dir, 
                                                 custom_preprocessing=custom_preprocessing,
                                                 evaluate_model=evaluate_model, 
                                                 verbose=verbose
                                                 )
                


        for possible_class_name in classes:
            if possible_class_name not in results_dict.keys():
                results_dict[possible_class_name] = 0


        results[actual_class_name] = results_dict
    

    
    f = open(conf_pickle, 'wb')
    pickle.dump(results, f)
    f.close()

    
    # Columns-predicted class names, Rows- actual class names
    
    with open(conf_csv, 'w') as f_object:
        
        writer_object = csv.writer(f_object)            
        
        header_row = ["Class"]
        header_row.extend(classes)
        
        writer_object.writerow(header_row)
        
        for actual_class_name in classes:
            row = [actual_class_name]
            actual_class_dict = results[actual_class_name]
            for predicted_class in classes:
                row.append(actual_class_dict[predicted_class])

            writer_object.writerow(row)
            

    conf_mat = extract_conf_mat(conf_pickle)
    compute_conf_mat_scores(conf_mat, conf_pickle=conf_pickle)


def main():    

    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    ap.add_argument("-evaluate_model", type=int, help=" ", default = 0, choices = [0,1])
    
    ap.add_argument("-verbose", type=int, help="", default = 0, choices = [0,1])
    
    ap.add_argument("-root_val_dir", default="dataset/val", help=" ")


    requiredNamed = ap.add_argument_group('required named arguments')  
    
    requiredNamed.add_argument("-train_pickle_file", required=True, help=" ")
    
    args = vars(ap.parse_args())

    evaluate_model(args)




if __name__ == "__main__":
    main()
