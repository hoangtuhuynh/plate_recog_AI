import easyocr
import cv2 
#import pandas as pd
import csv

reader = easyocr.Reader(['en'], gpu=False)
def read_text(frame, detections, list):
    
     
    for bbox in detections.xyxy:
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        area = frame[y1:y2, x1:x2]
        area = cv2.cvtColor(area, cv2.COLOR_RGB2GRAY)
        result = reader.readtext(area, detail=1, paragraph=False)
        for i, (bboxes, text, prob) in enumerate(result):
            text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
            if text not in list:
                list.append(text)
                write_to_csv(text, prob)

def create_csv():
    with open('detected.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["License Plate", "Probability"])
def write_to_csv(text, prob):
    with open('detected.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([text, prob])