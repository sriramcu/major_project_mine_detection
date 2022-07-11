import os
import sys
import shutil
import argparse


def main():
      
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    ap.add_argument("-train_dir", type=str, required=True, help="Train directory- REQUIRED")
    ap.add_argument("-val_split", type=float, required=True, help="Train-val split- REQUIRED")
    
    args = vars(ap.parse_args())        
    
    # Change working directory of the script to the folder containing 'train' and 'val' folders
    os.chdir(os.path.dirname(args["train_dir"]))
    os.makedirs("val", exist_ok=True)
    train_categories = list(os.listdir('train'))
    val_split = float(args["val_split"])
    sampling_rate = int(1/val_split)
    for category in train_categories:
        os.makedirs(os.path.join("val",category), exist_ok=True)
        all_images = list(os.listdir(os.path.join(os.getcwd(), 'train', category)))
        for img_num, img_name in enumerate(all_images):
            if img_num % sampling_rate != 0:
                continue
            shutil.move(os.path.join(os.getcwd(), 'train', category,img_name),os.path.join(os.getcwd(), 'val', category,img_name))




if __name__ == "__main__":
    main()