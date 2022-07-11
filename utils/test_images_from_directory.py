import pickle
from keras.models import load_model
from my_datagen import return_my_datagen
import sys
import os
import time
from pathlib import Path
from mapgi import mapgi
from test_img import test_img
import argparse


def test_images_from_directory(model, test_dir, class_names = [], evaluate_model=True, 
                               custom_preprocessing=True, verbose=False):

    test_dir_path = Path(test_dir)
    if class_names == []:
        class_names = sorted(os.listdir(test_dir_path.resolve().parent))

    val_datagen = return_my_datagen(custom_preprocessing=custom_preprocessing, mode="prediction")  
        

                    
    if evaluate_model:        

        
        val_dir = test_dir_path.resolve().parent
        val_gen = val_datagen.flow_from_directory(val_dir,target_size=(299,299),
                                                  class_mode="categorical")
        

        validation_metrics = model.evaluate(val_gen)
        for metric_index, metric_score in enumerate(validation_metrics):
            print(f"{model.metrics_names[metric_index].title()} = {metric_score}")       

    
    img_width, img_height = 299, 299

    print(f"Testing images located in {test_dir}")
    counter = 0
    results_dict = {}
    start_time = time.time()
    
    test_dir_size = len(os.listdir(test_dir))
    for filename_img in os.listdir(test_dir):
        
        filename = os.path.abspath(os.path.join(test_dir,filename_img))       
        
        successful, predicted_class, _ = test_img(filename, img_width, img_height, 
                                               model, val_datagen, class_names, verbose=verbose)
        
        if not successful:
            print(f"Could not process image {filename}")
            continue

        if predicted_class not in results_dict.keys():
            results_dict[predicted_class] = 1
        else:
            results_dict[predicted_class] += 1

        counter += 1    
        
        if counter % 100 == 0:
            print(f"{counter} out of {test_dir_size} files processed!")



    time_taken = time.time() - start_time
    time_taken = round(time_taken,2)
    
    print(f"{counter} images processed in {time_taken} seconds, at a rate of "
          f"{round(counter/time_taken,2)} images per second.")
    
    for predicted_class in results_dict.keys():
        num_predictions = results_dict[predicted_class]
        percentage_predicted = round(100*num_predictions/counter,2)
        print(f"{predicted_class} = {num_predictions} predictions ({percentage_predicted}%)")

    return results_dict

def main():
    #todo- use argparse
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    ap.add_argument("-evaluate_model", type=int, help=" ", default=0, choices=[0,1])
        
    requiredNamed = ap.add_argument_group('required named arguments')

    requiredNamed.add_argument("-test_directory", required=True, help=" ", type=str)
    
    requiredNamed.add_argument("-train_pickle_file", required=True, help=" ")
    
    
    args = vars(ap.parse_args())    

    class_names = CLASS_LABELS

    print(f"Classes being predicted by this model are: {class_names}")

    
    metrics_pickle_filepath = args["train_pickle_file"]

    f = open(metrics_pickle_filepath, 'rb')
    
    train_metrics = pickle.load(f)
    params = pickle.load(f)
    
    f.close()
    
    checkpoints_filepath = params[-1]
    
    print(checkpoints_filepath)
    
    model=load_model(checkpoints_filepath)
    
    test_images_from_directory(model, test_dir=args["test_directory"], 
                               evaluate_model=int(args["evaluate_model"]), 
                               class_names=class_names,
                               custom_preprocessing=params[-2]
                               )



if __name__ == "__main__":
    main()