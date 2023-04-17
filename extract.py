import easyocr
import cv2 

def read_text(frame, detections):
    reader = easyocr.Reader(['en'], gpu=False)

    for bbox in detections.xyxy:
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        area = frame[y1:y2, x1:x2]
        result = reader.readtext(area, detail=1, paragraph=False)
        for (bbox, text, prob) in result:
            text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
            print(text)

