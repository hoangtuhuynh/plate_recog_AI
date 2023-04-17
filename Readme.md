## License Plate detection with YOLOv8 and Easy_ocr
Owner: **Hoang Tu Huynh**

### Overview:
The project use Yolov8 format, python to automatically detect the vehicle license plate through the video, image and real time webcam.
![detected](/test_img/product.png)  

### Approaches:
- Use Roboflow to get and annotate the images for training processing ([Roboflow](https://roboflow.com/))
- Training datasets with Google Colab and apply Yolov8 format to get the best custom model 
- Apply Easy_ocr to extract text from detected objects, specifically vehicle license plate
