import torch
import onnxruntime as ort
import numpy as np
import cv2

def main():
    model = r'.\model\best.onnx'
    session = ort.InferenceSession(model)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ret, frame = cap.read()
        

        # Preprocess the frame
        img = cv2.resize(frame, (640, 640))
        img = img.transpose((2, 0, 1))
        img = np.expand_dims(img, axis=0)
        
        # run the onnx model on the frame
        results = session.run([output_name], {input_name: img})

        #Get the class ID and confidence score for each object detected
        class_ids = results[0][0][:, 1]
        confidences = results[0][0][:,2]

        # Draw the bounding boxes
        for i in range(len(class_ids)):
            if confidences[i] > 0.2:
                x1, y1, x2, y2 = results[0][0][i][:4]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (225, 0, 0), 2)
        

        # Show the results of the frame
        cv2.imshow("Detected", frame)    
        if (cv2.waitKey(30) == 27):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
