"""
preprocessing_function 	function that will be applied on each input. The function will run 
after the image is resized and augmented. 
The function should take one argument: one image (Numpy tensor with rank 3), 
and should output a Numpy tensor with the same shape. 
tf.rank([[1],[2]]).numpy()
2
"""

import math
import numpy as np
from PIL import Image
import os
import cv2
import datetime
import sys 
import time
import skimage
import argparse

def edge_removal(image, threshold=50, max_black_ratio=0.7, mode="hsv"):

    def pixels_should_be_conserved(pixels) -> bool:
        
        if mode == "hsv":
            black_pixel_count = (pixels[:,2] <= threshold).sum()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            
        elif mode == "yuv":
            black_pixel_count = (pixels[:,0] <= threshold).sum()
            
        elif mode == "grayscale":
            black_pixel_count = (pixels <= threshold).sum()
        else:
            black_pixel_count = (pixels <= threshold).all(axis=1).sum()
            
        pixel_count = len(pixels)
        
        return pixel_count > 0 and black_pixel_count/pixel_count <= max_black_ratio    

    
    num_rows, num_columns, _ = image.shape
    preserved_rows    = [r for r in range(num_rows)    if pixels_should_be_conserved(image[r, :, :])]
    preserved_columns = [c for c in range(num_columns) if pixels_should_be_conserved(image[:, c, :])]

    image = image[preserved_rows,:,:]
    image = image[:,preserved_columns,:]
        
   
    if mode == "hsv":
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        
    return image



def clahe(colorimage, clipLimit: float, tileGridSize: tuple):
    # colorimage is in BGR
    clahe_model = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    colorimage_b = clahe_model.apply(colorimage[:,:,0])
    colorimage_g = clahe_model.apply(colorimage[:,:,1])
    colorimage_r = clahe_model.apply(colorimage[:,:,2])
    colorimage = np.stack((colorimage_b,colorimage_g,colorimage_r), axis=2)
    return colorimage


def magva(image):
    # image is in BGR
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # to use skimage gamma correction function
    img_mean = np.mean(image)
    desired_mean = 90
    margin = 1
    
    lower_bound = desired_mean - margin
    upper_bound = desired_mean + margin
    
    while img_mean < lower_bound or img_mean > upper_bound:
        gamma = math.log(desired_mean/255) / math.log(img_mean/255)
        image = skimage.exposure.adjust_gamma(image, gamma)
        img_mean = np.mean(image)
        
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image
        
    

def apply_low_pass_filter(image):
    kernel = np.array([[0.1, 0.1, 0.1],[0.1, 1, 0.1],[0.1, 0.1, 0.1]])/1.8
    image = cv2.filter2D(image, cv2.CV_64F, kernel)
    return image

 

def mapgi(image, clipLimit: float = 1.0, tileGridSize: tuple = (2,2), image_format="rgb"):

    sq_flag = False
      
    # variable to check whether the dimensions of the input image were squeezed 
    # (happens in image evaluation but not training)
    
    if image_format == "rgb":

        if image.ndim == 4:
            image = np.squeeze(image, axis=0)
            sq_flag = True
        image = np.asarray(image)
        image = np.uint8(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)  
        
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)    
   
         
    image = edge_removal(image, threshold=37, mode="yuv")
    
    y_component = image[:,:,0]
    u_component = image[:,:,1]
    v_component = image[:,:,2]    
    
    # image = clahe(image, clipLimit, tileGridSize)
    
    y_component = magva(y_component)    

    y_component = apply_low_pass_filter(y_component)

    y_component = y_component.astype(u_component.dtype)   

    # cv2.merge needs all channels to have the same dtype  
     
    yuv_image = cv2.merge((y_component, u_component, v_component))

    
    image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)

    image = cv2.resize(image,(299,299), interpolation = cv2.INTER_CUBIC)

    if sq_flag:
        image = np.expand_dims(image, axis=0)

  
    image = np.float32(image)
    return image


def process_image_file_mapgi(img_path, clipLimit: float, tileGridSize: tuple):
    img_arr = cv2.imread(img_path)  # BGR
    img_arr = cv2.resize(img_arr,(299,299))
    img_path_without_ext = "".join(img_path.split(".")[:-1])
    resized_img_path = img_path_without_ext +  "_resized." + img_path.split(".")[-1]
    
    # cv2.imwrite(resized_img_path, img_arr)    
    
    img_arr = mapgi(img_arr, clipLimit, tileGridSize, image_format="bgr")
    clipLimit_str = str(clipLimit)
    img_path_without_ext += f"_{clipLimit_str}_{tileGridSize}"
    new_img_path = img_path_without_ext +  "." + img_path.split(".")[-1]

    cv2.imwrite(new_img_path, img_arr)    



def main():
    
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("-image_filepath", type=str, required=True, help="REQUIRED")
    ap.add_argument("-clip_limit", type=float, default=1.0, help=" ")
    ap.add_argument("-grid_size", type=int, default=2, help=" ")
    args = vars(ap.parse_args())
    
    grid_size = (int(args["grid_size"]), int(args["grid_size"]))
    process_image_file_mapgi(args["image_filepath"], args["clip_limit"], grid_size)    
    



if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Time taken = {time.time()-start_time}")
