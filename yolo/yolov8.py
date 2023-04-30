class YOLOV8:
    def __init__(self, conf_thres = 0.5, iou_thres = 0.5):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        # intitialize the model
        pass
    def __call__(self, frame):
        self.object_detect()
    def object_detect(self):
        pass