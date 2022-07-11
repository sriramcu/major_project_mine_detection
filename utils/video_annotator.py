import os
from pathlib import Path
import sys
import cv2
import argparse

import numpy as np
from pymediainfo import MediaInfo
from time import time

from constants import *


def annotate_frame(frame, text, color_str, position="top_left"):

    # Returns new, modified frame, not in-place
    
    height, width, _ = frame.shape
    
    pixels_needed  = len(text) * int(230/14 + 1) + 5

    
    if position == "top_right":   
        x = width - pixels_needed
        y = 40
        
    elif position == "top_left":
        x = 10
        y = 160  # Existing kvasir annotation occupies this place
    
    else:
        raise ValueError(f"Invalid frame position '{position}'")    
    
    
    x = int(x)
    y = int(y)

    
    modified_frame = frame.copy()
    
    
    font_size = 0.9
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_thickness = 2
    
    # BGR for cv2
    color_dict = COLOR_DICT
    
    font_color = color_dict[color_str]
    
    modified_frame = cv2.putText(modified_frame, 
                                text, 
                                (x,y), 
                                font, 
                                font_size, 
                                font_color, 
                                font_thickness)  
    
    return modified_frame


def annotate_video(input_video_path, timestamps, text_list, output_video_path):
    
    start_time = time()

    vidcap = cv2.VideoCapture(input_video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH ))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT ))
    success, frame = vidcap.read()
    count = 0

    
    timestamp_idx = 0                                                   
    
    Path(output_video_path).unlink(missing_ok=True)

    
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    output_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            
    
    while success:
        success, frame = vidcap.read()
        
        if not success:
            break
        
        count+=1
        lower_timestamp = timestamps[timestamp_idx][0]
        upper_timestamp = timestamps[timestamp_idx][1]
        
        current_ts = count/fps
        
        if current_ts >= lower_timestamp and current_ts <= upper_timestamp:
            modified_frame = annotate_frame(frame, text_list[timestamp_idx], "yellow")
            output_writer.write(modified_frame)
            # write modified to output
        
        elif current_ts > upper_timestamp:
            timestamp_idx += 1
            if timestamp_idx >= len(timestamps):
                break
            
    print(f"Annotation of video took {time()-start_time} seconds")



def verify_timestamps(timestamps, vid_duration):
    flattened_timestamps = list(np.array(timestamps).flatten())
    
    if sorted(flattened_timestamps) != flattened_timestamps:
        raise ValueError("Timestamps entered non-sequentially.")
    
    if max(flattened_timestamps) > vid_duration or min(flattened_timestamps) < 0:
        raise ValueError("All timestamps must be in seconds between 0 and video duration.")
        
        

def cli_annotation_input(input_video_path, output_video_path):
    timestamps = []
    text_list = []
    
    media_info = MediaInfo.parse(input_video_path)
    #duration in milliseconds
    duration_in_ms = media_info.tracks[0].duration
    
    vid_duration = int(duration_in_ms/1000)        
     
    print(f"Video duration = {vid_duration} seconds")       
    
    while True:
        start_ts = int(input("Enter starting timestamp in seconds (-1 for stopping): "))
        if start_ts == -1:
            break
        
        end_ts = int(input("Enter ending timestamp in seconds (-1 for end of video): "))
        
        if end_ts == -1:            
            end_ts = vid_duration
        
        text = input("Enter text to be annotated: ")
        timestamps.append([start_ts, end_ts])
        text_list.append(text)
    
    verify_timestamps(timestamps, vid_duration)    
     
    print(timestamps, text_list)   
    
    return input_video_path, timestamps, text_list, output_video_path                
        
        

def main():
    input_video_path, timestamps, text_list, output_video_path = cli_annotation_input(os.path.abspath(sys.argv[1]), 
                                                                                      os.path.abspath(sys.argv[2]))
    annotate_video(input_video_path, timestamps, text_list, output_video_path)
    
    
if __name__ == "__main__":
    main()