import os
from pathlib import Path
import pickle
import cv2
import sys
from keras.models import load_model
import numpy as np
import efficientnet.keras as efn


from video_annotator import annotate_frame
from constants import *
from test_img import test_frame
from my_datagen import return_my_datagen
from time import time


# TODO merge vids from kvasir and test this program, annotate to get final rvce output
def evaluate_video(input_video_path, output_video_path, metrics_pickle_filepath):
    
    start_time = time()

    f = open(metrics_pickle_filepath, 'rb')
    
    train_metrics = pickle.load(f)
    params = pickle.load(f)
    
    f.close()   

        
    custom_preprocessing = params[-3]

    checkpoints_filepath = params[-1]
    
    model = load_model(checkpoints_filepath)

    vidcap = cv2.VideoCapture(input_video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH ))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT ))
    
    
    success, frame = vidcap.read()
    count = 0

    Path(output_video_path).unlink(missing_ok=True)
                                                    
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    output_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            
    
    datagen = return_my_datagen(custom_preprocessing=custom_preprocessing, mode="prediction")  

    while success:
        
        success, frame = vidcap.read()
        
        if not success:
            break
        
        original_frame = frame.copy()
        
        count += 1
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        frame = cv2.resize(frame, (299, 299))
        
        if frame.dtype == "uint8":
            frame = frame.astype("float64")
        
        
        class_names = CLASS_LABELS
    
        _, predicted_class, rounded_prob = test_frame(frame, model,
                                                      datagen, class_names, verbose=False) 
        
            
        pred_txt = f"{predicted_class}:{rounded_prob}%"
        
        # Because for output video we do not want to or need to show internal image processing
        annotated_frame = annotate_frame(original_frame, pred_txt, "green", position="top_right")

        output_writer.write(annotated_frame)
        
    print(f"Evaluation of video took {time()-start_time} seconds")

if __name__ == "__main__":

    evaluate_video(os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]), 
                   os.path.abspath(sys.argv[3]))