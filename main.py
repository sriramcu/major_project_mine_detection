import argparse
import datetime
import os

from utils.video_annotator import *
from utils.video_evaluate import *


def main():
    parser = argparse.ArgumentParser(argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("-i", help="Input video path", required=True)
    parser.add_argument("-o", help="Output video path", required=True)
    parser.add_argument("-tp", help="Training pickle file path", required=True)    
    
    args = vars(parser.parse_args())

    existing_file_parser_keys = ["i", "tp"]
    
    for key in existing_file_parser_keys:
        if not os.path.isfile(args[key]):
            raise FileNotFoundError(f"File '{args[key]}' not found!")
    
    input_video_path = os.path.abspath(args["i"])
    train_pickle_path = os.path.abspath(args["tp"])
    output_video_path = os.path.abspath(args["o"])
    
    if not input_video_path.endswith("mkv") or not output_video_path.endswith("mkv"):
        raise ValueError("Only mkv files supported by the program!")
    
    input_vid_dir = os.path.dirname(input_video_path)
    
    timestamp_str = datetime.datetime.now().strftime("%d_%m_%y_%H_%M_%S")
    
    evaluated_interim_path = os.path.join(input_vid_dir, f"evaluated_{timestamp_str}.mkv")
    
    input_video_path, timestamps, text_list, output_video_path = cli_annotation_input(input_video_path,
                                                                                      output_video_path)
    
    evaluate_video(input_video_path, evaluated_interim_path, train_pickle_path)
    
    annotate_video(evaluated_interim_path, timestamps, text_list, output_video_path)
    
    Path(evaluated_interim_path).unlink(missing_ok=True)
    
    

if __name__ == "__main__":
    main()