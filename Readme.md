## License Plate detection with YOLOv8 and EasyOCR
Owner: **Hoang Tu Huynh**

### Overview:
The project uses Yolov8 format, python and EasyOCR to automatically detect the vehicle license plate through the video, image and real time webcam, and then save the result to csv file.
![detected](/test_img/result.png)  

### Approaches:
- Use Roboflow to get and annotate the images for training processing ([Roboflow](https://roboflow.com/))
- Training datasets with Google Colab and apply Yolov8 format to get the best custom model 
- Apply Easy_ocr to extract text from detected objects, specifically vehicle license plate
- Save the extracted text to csv file

### Running Processes:
- Clone the folder:<br><br>
`git clone https://github.com/hoangtuhuynh/plate_recog_AI.git` <br><br>
`cd plate_recog_AI`
- Create the environment:<br><br>
`python3 -m venv venv`
- Activate the environment:<br><br>
`venv\Scripts\activate`
- Install all the requirements required:<br><br>
`pip install -r requirements.txt`
- Run code: <br><br>
`python -m detect --source 0 --model .\model\best.pt` : Running with webcam <br><br>
`python -m detect --source .\test_img\Cars8.png` --model .\model\best.pt: Running with image<br><br>
`python -m detect --source .\test_img\videotest.mp4 --model .\model\best.pt`: Running with video

