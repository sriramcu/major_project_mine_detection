import numpy as np
from keras.preprocessing import image
import PIL
import sys
import cv2


def test_img(filename, img_width, img_height, model, datagen, class_names, verbose=False):
        
    try:
        img = image.load_img(filename, target_size=(img_width, img_height))
    except PIL.UnidentifiedImageError:
        print(f"{filename} couldn't be processed by load_img")
        return False, None    

    test_image = image.img_to_array(img)
    
    return test_frame(test_image, model, datagen, class_names, verbose)
    
    
    

def test_frame(test_image, model, datagen, class_names, verbose=False):
    
    original_test_image = test_image.copy()
    
    try:
        
        test_image = np.expand_dims(test_image, axis=0)
        test_image = datagen.standardize(test_image)
        # test_image.shape
        # test_image = test_image/255
        # images = np.vstack([test_image])
        predicted_classes = model.predict(test_image, batch_size=10) 
        
        
    except ValueError:
        print("Dimension of test image doesn't match input dimension of network")
        h = int(input("Enter input height: "))
        w = int(input("Enter input width: "))
        
        test_image = cv2.resize(original_test_image, (h, w))
        test_image = np.expand_dims(test_image, axis=0)
        test_image = datagen.standardize(test_image)
        
        try:
            predicted_classes = model.predict(test_image) 
        
        except ValueError:
            test_image = np.squeeze(test_image)

            test_image = cv2.resize(test_image, (h,w))

            test_image = np.expand_dims(test_image, axis=0)
            predicted_classes = model.predict(test_image) 
            
                
    predicted_class = class_names[np.argmax(predicted_classes)]
    rounded_prob = round(np.amax(predicted_classes)*100,2)
    
    if verbose:
        print(f"{predicted_class}:{rounded_prob}%")      
    
    return True, predicted_class, rounded_prob
    

