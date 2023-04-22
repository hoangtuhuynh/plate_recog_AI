import cv2
import argparse

from ultralytics import YOLO
import supervision as sv
import numpy as np

import extract
import os


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = 'YoloV8 Plate Detection')
    parser.add_argument(
        "--webcam-resolution",
        default=[1280,720],
        type =int,
        nargs= 2
    )
    parser.add_argument('--model', default= r".\model\best.pt", type=str)
    parser.add_argument("--source", default="0", type=str)
    args = parser.parse_args()
    return args
def load_sources(filename):
    img_format = ['jpg', 'png', 'jpeg', 'tif', 'tiff', 'dng', 'webp', 'mpo']
    key = 1 # 1 =  video, 0 = image
    frame = None
    cap = None

    # if filename is webcam
    if(filename == '0'):
        img_type = False
        filename = 0 
    else:
        img_type = filename.split('.')[-1].lower() in img_format

    # if filename is video or image
    if(img_type):
        key = 0
        frame = cv2.imread(filename)
    else:
        cap = cv2.VideoCapture(filename)
    return img_type, cap, frame, key

def main():
    # command line arguments
    args = parse_arguments()
    # Load the source file
    source_file = args.source
    img_type, cap, frame, key =  load_sources(source_file)
    if(key == 1):
        frame_width, frame_height = args.webcam_resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    model = args.model
    yolo_model = YOLO(model)
    box_annotator = sv.BoxAnnotator(
        thickness = 2,
        text_thickness = 2,
        text_scale=1
    )
    # create the csv file to hold the results if it is not existing
    if not os.path.exists('detected.csv'):
        extract.create_csv()
    list = []
    
    while True:
        if not img_type:
            ret, frame = cap.read()
        result = yolo_model(frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_yolov8(result)
        labels = [
            f"{yolo_model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, _
            in detections
        ]
        frame = box_annotator.annotate(
            scene = frame,
            detections = detections,
            labels = labels
        )
        
        # wait fot 30 mil seconds and 27 plays as the escape in ascii table
        cv2.imshow('Detected', frame)
        extract.read_text(frame, detections, list)
        
        
        if(cv2.waitKey(30) == 27):
            break

if __name__ == '__main__':
    main()