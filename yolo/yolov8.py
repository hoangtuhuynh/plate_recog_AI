import onnxruntime as ort
import numpy as np
import cv2
import supervision as sv


class_name = ['licence-plate', 'license-plate', 'plate']

class YOLOV8:
    
    def __init__(self, model, conf_thres = 0.5, iou_thres = 0.5):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        # intitialize the model
        self.model_initialize(model)

    ## call itself with the frame as parameter
    def __call__(self, frame):
        self.object_detect(frame)
    
    def model_initialize(self, model):
        self.session = ort.InferenceSession(model)

        # get model information
        self.get_input_details()
        self.get_output_details()


    def object_detect(self, frame):
        self.img_height, self.img_width = (1280, 720)

        # resize the frame
        input_frame = cv2.resize(frame, (self.img_width, self.img_height))
        # scale the pixel to 0 and 1
        input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_frame = input_frame/ 255.0
        input_frame = input_frame.transpose(2, 0, 1)
        input_frame = input_frame[np.newaxis, :, :, :].astype(np.float32)

        # run the session
        output_frame = self.session.run(self.input_names, {self.output_names[0]: input_frame})
        self.scores, self.class_ids =self.process_output(output_frame)

    

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

    def process_output(self,output):
        predictions = np.squeeze(output[0]).T
        # Filter out object confidence scores below threshold

        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], []

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        return scores, class_ids


    
if __name__ == '__main__':
    model = r"../model/best.onnx"

    yolo = YOLOV8(model)
    filename = r"../test_img/Cars8.png"
    frame = cv2.imread(filename)
    box_annotator = sv.BoxAnnotator(
        thickness = 2,
        text_thickness = 2,
        text_scale=1
    )
    while True:
        confidence, class_id = yolo(frame=frame)

        
        if(cv2.waitKey(30) == 27):
            break
